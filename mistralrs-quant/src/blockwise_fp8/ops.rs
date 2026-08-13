use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Result, Tensor, WithDType};
use float8::F8E4M3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

struct Fp8BlockwiseDequantize {
    weight_block_size: Vec<usize>,
    out_ty: DType,
}

impl Fp8BlockwiseDequantize {
    fn dispatch_dequant_blockwise<T: WithDType>(
        &self,
        weight: &[F8E4M3],
        scale: &[f32],
        weight_l: &candle_core::Layout,
        scale_l: &candle_core::Layout,
    ) -> candle_core::Result<Vec<T>> {
        let grid_y = weight_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = weight_l.dim(1)?.div_ceil(self.weight_block_size[1]);

        let res = vec![T::zero(); weight.len()];

        (0..grid_y).into_par_iter().for_each(|y| {
            (0..grid_x).into_par_iter().for_each(|x| {
                let res_ptr = res.as_ptr() as *mut T;

                let scale = scale[y * scale_l.stride()[0] + x];

                let start_y = y * self.weight_block_size[0];
                let end_y = start_y + self.weight_block_size[0];

                let start_x = x * self.weight_block_size[1];
                let end_x = start_x + self.weight_block_size[1];

                for weight_y in start_y..end_y {
                    if weight_y >= weight_l.dims()[0] {
                        break;
                    }

                    let row_offset = weight_y * weight_l.stride()[0];
                    for weight_x in start_x..end_x {
                        if weight_x >= weight_l.dims()[1] {
                            break;
                        }

                        let weight_pos = row_offset + weight_x;

                        // SAFETY: We know each thread will only update indepedant values!
                        unsafe {
                            *res_ptr.wrapping_add(weight_pos) =
                                T::from_f64((weight[weight_pos].to_f32() * scale) as f64);
                        }
                    }
                }
            });
        });

        Ok(res)
    }
}

impl CustomOp2 for Fp8BlockwiseDequantize {
    fn name(&self) -> &'static str {
        "fp8-blockwise-dequantize"
    }

    fn cpu_fwd(
        &self,
        scale_s: &candle_core::CpuStorage,
        scale_l: &candle_core::Layout,
        weight_s: &candle_core::CpuStorage,
        weight_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let candle_core::CpuStorage::F8E4M3(weight) = weight_s else {
            candle_core::bail!("Expected F8E4M3 weight!");
        };
        let candle_core::CpuStorage::F32(scale) = scale_s else {
            candle_core::bail!("Expected F8E4M3 weight!");
        };
        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }
        if weight_l.dims().len() != 2 {
            candle_core::bail!("Expected weight to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scale to be rank 2");
        }

        match self.out_ty {
            DType::F32 => Ok((
                CpuStorage::F32(self.dispatch_dequant_blockwise(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            DType::BF16 => Ok((
                CpuStorage::BF16(
                    self.dispatch_dequant_blockwise(weight, scale, weight_l, scale_l)?,
                ),
                weight_l.shape().clone(),
            )),
            DType::F16 => Ok((
                CpuStorage::F16(self.dispatch_dequant_blockwise(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            other => candle_core::bail!("unexpected out type of fp8 blockwise dequant {other:?}"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        scale_s: &candle_core::CudaStorage,
        scale_l: &candle_core::Layout,
        weight_s: &candle_core::CudaStorage,
        weight_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, CudaStorage};
        use half::{bf16, f16};

        use crate::{blockwise_fp8::ffi, utils::slice_ptr};

        if !ffi::HAVE_BLOCKWISE_DEQUANT_KERNELS {
            candle_core::bail!("Do not have blockwise FP8 dequant kernels.");
        }

        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }
        if weight_l.dims().len() != 2 {
            candle_core::bail!("Expected weight to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scale to be rank 2");
        }

        let dev = weight_s.device();

        let (weight, _weight_guard) =
            slice_ptr(weight_s.as_cuda_slice::<F8E4M3>()?, weight_l.start_offset());
        let (scale, _scale_guard) =
            slice_ptr(scale_s.as_cuda_slice::<f32>()?, scale_l.start_offset());

        let weight_height = weight_l.dim(0)? as i32;
        let weight_block_size_x = self.weight_block_size[0] as i32;
        let weight_width = weight_l.dim(1)? as i32;
        let weight_block_size_y = self.weight_block_size[1] as i32;
        let scale_stride = scale_l.stride()[0] as i32;
        let weight_row_stride = weight_l.stride()[0] as i32;

        let res = match self.out_ty {
            DType::F32 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<f32>(weight_l.shape().elem_count())?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_f32(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            DType::F16 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<f16>(weight_l.shape().elem_count())?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_f16(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            DType::BF16 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<bf16>(weight_l.shape().elem_count())?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_bf16(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            other => candle_core::bail!("unexpected out type of fp8 blockwise dequant {other:?}"),
        };

        Ok((res, weight_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        scale_s: &candle_core::MetalStorage,
        scale_l: &candle_core::Layout,
        weight_s: &candle_core::MetalStorage,
        weight_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if weight_l.start_offset() != 0
            || !weight_l.is_contiguous()
            || weight_s.dtype() != DType::F8E4M3
        {
            candle_core::bail!("Expected f8e4m3 weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() || scale_s.dtype() != DType::F32
        {
            candle_core::bail!("Expected f32 scales to have start offset 0, continuous");
        }
        if weight_l.dims().len() != 2 {
            candle_core::bail!("Expected weight to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scale to be rank 2");
        }

        let encoder = weight_s.device().command_encoder()?;
        encoder.set_label("dequant-blockwise-fp8");

        let device = weight_s.device();

        let out_shape = weight_l.shape().clone();

        let output = device.new_buffer(
            out_shape.elem_count(),
            weight_s.dtype(),
            "dequant-blockwise-fp8",
        )?;

        let weight_height = weight_l.dim(0)? as u32;
        let weight_block_size_x = self.weight_block_size[0] as u32;
        let weight_width = weight_l.dim(1)? as u32;
        let weight_block_size_y = self.weight_block_size[1] as u32;
        let scale_stride = scale_l.stride()[0] as u32;
        let weight_row_stride = weight_l.stride()[0] as u32;

        crate::metal_kernels::call_dequant_blockwise_fp8(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            self.out_ty,
            weight_s.buffer(),
            scale_s.buffer(),
            &output,
            weight_height,
            weight_width,
            weight_row_stride,
            scale_stride,
            weight_block_size_y,
            weight_block_size_x,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            out_shape.elem_count(),
            self.out_ty,
        );
        Ok((newstorage, out_shape))
    }
}

/// FP8 blockwise dequantize.
/// - Expects weight to be fp8
/// - Expects inv_scales to be f32
/// - weight * inv_scale = dequantized
pub fn fp8_blockwise_dequantize(
    weight: &Tensor,
    inv_scales: &Tensor,
    weight_block_size: Vec<usize>,
    out_ty: DType,
) -> Result<Tensor> {
    inv_scales.apply_op2_no_bwd(
        weight,
        &Fp8BlockwiseDequantize {
            weight_block_size,
            out_ty,
        },
    )
}

// ── MXFP4 (E2M1, packed-as-I8) blockwise dequantization ──────────────────────
//
// DeepSeek V4 Flash stores routed-expert MoE weights as MXFP4 (OCP microscaling
// float4, E2M1) packed into I8, with an F8E8M0 block scale:
//   - Each I8 byte holds 2 sign-magnitude FP4 values. Per nibble: MSB = sign,
//     low 3 bits = E2M1 magnitude code -> {0,.5,1,1.5,2,3,4,6}.
//     Low nibble  = byte & 0x0F ; High nibble = (byte >> 4) & 0x0F.
//     (These are NOT two's-complement linear INT4 -- see decode note below.)
//   - Weight shape on disk: [rows, cols/2]  (half the logical column count)
//   - Scale shape: [rows/block_size[0], cols/block_size[1]]  (cols = unpacked count)
//   - Scales are F32 (after F8E8M0 → F32 decode at load time).
//
// Output: BF16 (or F16/F32) tensor of shape [rows, cols].

struct MxInt4BlockwiseDequantize {
    weight_block_size: Vec<usize>,
    out_ty: DType,
}

impl MxInt4BlockwiseDequantize {
    /// Dequantize INT4-packed-as-I32 weights with F32 block scales.
    ///
    /// `packed` contains i32 values, each being a sign-extended i8 byte that
    /// holds two INT4 values (low nibble and high nibble).
    fn dispatch_dequant_blockwise<T: WithDType>(
        &self,
        packed: &[i32],
        scale: &[f32],
        packed_l: &candle_core::Layout,
        scale_l: &candle_core::Layout,
    ) -> candle_core::Result<Vec<T>> {
        let rows = packed_l.dim(0)?;
        let packed_cols = packed_l.dim(1)?;
        let cols = packed_cols * 2; // unpacked column count

        let block_h = self.weight_block_size[0];
        let block_w = self.weight_block_size[1];
        let grid_y = rows.div_ceil(block_h);
        let grid_x = cols.div_ceil(block_w);

        let res = vec![T::zero(); rows * cols];

        (0..grid_y).into_par_iter().for_each(|by| {
            (0..grid_x).into_par_iter().for_each(|bx| {
                let res_ptr = res.as_ptr() as *mut T;
                let s = scale[by * scale_l.stride()[0] + bx];

                let start_y = by * block_h;
                let end_y = (start_y + block_h).min(rows);
                let start_x = bx * block_w;
                let end_x = (start_x + block_w).min(cols);

                for y in start_y..end_y {
                    for x in start_x..end_x {
                        // Each packed i32 at [y, x/2] is a sign-extended i8 byte
                        // holding two INT4 values.
                        let byte_col = x / 2;
                        let byte = packed[y * packed_l.stride()[0] + byte_col] as i8;

                        // MXFP4 (E2M1) sign-magnitude decode. The 4-bit element
                        // is float4 E2M1: MSB = sign, low 3 bits = E2M1 magnitude
                        // code mapping to {0,.5,1,1.5,2,3,4,6}. It is NOT
                        // two's-complement linear INT4 -- decoding it as such
                        // injects a per-code DC bias (signed-nibble mean ~-1.05 ->
                        // weight mean ~-0.015 -> row-sum ~-61), which amplifies the
                        // expert input's DC offset into an all-positive gate/up
                        // shift that explodes the MoE at the first compressed layer
                        // (RUN-161). The on-disk histogram is periodic with period
                        // 8 (count(n) ~= count(n+8)) -- the sign-magnitude
                        // signature -- and the magnitude counts are non-monotonic,
                        // matching E2M1's non-uniform levels.
                        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                        let ub = byte as u8;
                        let nib: u8 = if x % 2 == 0 { ub & 0x0F } else { ub >> 4 };
                        let sign = if nib & 0x08 != 0 { -1.0f32 } else { 1.0f32 };
                        let dequant_val = sign * E2M1[(nib & 0x07) as usize] * s;

                        // SAFETY: each thread writes to independent output positions
                        unsafe {
                            *res_ptr.wrapping_add(y * cols + x) = T::from_f64(dequant_val as f64);
                        }
                    }
                }
            });
        });

        Ok(res)
    }
}

impl CustomOp2 for MxInt4BlockwiseDequantize {
    fn name(&self) -> &'static str {
        "mx-int4-blockwise-dequantize"
    }

    fn cpu_fwd(
        &self,
        scale_s: &candle_core::CpuStorage,
        scale_l: &candle_core::Layout,
        packed_s: &candle_core::CpuStorage,
        packed_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let CpuStorage::I32(packed) = packed_s else {
            candle_core::bail!(
                "MX INT4 dequant expects I32 (sign-extended I8) packed weights, got {:?}",
                packed_s
            );
        };
        let CpuStorage::F32(scale) = scale_s else {
            candle_core::bail!("MX INT4 dequant expects F32 scales, got {:?}", scale_s);
        };
        if packed_l.start_offset() != 0 || !packed_l.is_contiguous() {
            candle_core::bail!("Expected packed weights to have start offset 0, contiguous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, contiguous");
        }
        if packed_l.dims().len() != 2 {
            candle_core::bail!("Expected packed weights to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scales to be rank 2 and weight_block_size len 2");
        }

        let rows = packed_l.dim(0)?;
        let cols = packed_l.dim(1)? * 2;
        let out_shape = candle_core::Shape::from_dims(&[rows, cols]);

        match self.out_ty {
            DType::F32 => Ok((
                CpuStorage::F32(self.dispatch_dequant_blockwise(packed, scale, packed_l, scale_l)?),
                out_shape,
            )),
            DType::BF16 => Ok((
                CpuStorage::BF16(
                    self.dispatch_dequant_blockwise(packed, scale, packed_l, scale_l)?,
                ),
                out_shape,
            )),
            DType::F16 => Ok((
                CpuStorage::F16(self.dispatch_dequant_blockwise(packed, scale, packed_l, scale_l)?),
                out_shape,
            )),
            other => candle_core::bail!("unexpected out type for MX INT4 dequant: {other:?}"),
        }
    }
}

/// MX INT4 blockwise dequantize.
/// - `packed`: I32 tensor of shape [rows, cols/2]. Each i32 is a sign-extended i8 byte
///   containing 2 signed INT4 values (low nibble + high nibble). This is how candle
///   loads I8 safetensor data (I8 → I32 cast in `convert()`).
/// - `scales`: F32 tensor of shape [rows/block_h, cols/block_w] (block scales, decoded from F8E8M0)
/// - `weight_block_size`: [block_h, block_w] (typically [128, 128])
/// - `out_ty`: output dtype (BF16, F16, or F32)
/// - Returns: tensor of shape [rows, cols] with dequantized values
pub fn mx_int4_blockwise_dequantize(
    packed: &Tensor,
    scales: &Tensor,
    weight_block_size: Vec<usize>,
    out_ty: DType,
) -> Result<Tensor> {
    scales.apply_op2_no_bwd(
        packed,
        &MxInt4BlockwiseDequantize {
            weight_block_size,
            out_ty,
        },
    )
}

#[allow(dead_code)]
struct Fp8BlockwiseQuantize {
    weight_block_size: Vec<usize>,
}

impl Fp8BlockwiseQuantize {
    #[allow(dead_code)]
    fn dispatch_quant_blockwise<T: WithDType>(
        &self,
        input: &[T],
        input_l: &candle_core::Layout,
    ) -> candle_core::Result<(Vec<F8E4M3>, Vec<f32>)> {
        let grid_y = input_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = input_l.dim(1)?.div_ceil(self.weight_block_size[1]);

        let weight = vec![F8E4M3::from_f32(0.0); input.len()];
        let scale = vec![0f32; grid_y * grid_x];

        (0..grid_y).into_par_iter().for_each(|y| {
            (0..grid_x).into_par_iter().for_each(|x| {
                let weight_ptr = weight.as_ptr() as *mut F8E4M3;
                let scale_ptr = scale.as_ptr() as *mut f32;

                let start_y = y * self.weight_block_size[0];
                let end_y = start_y + self.weight_block_size[0];

                let start_x = x * self.weight_block_size[1];
                let end_x = start_x + self.weight_block_size[1];

                // Find max absolute value in block
                let mut max_abs = 0f32;
                for weight_y in start_y..end_y {
                    if weight_y >= input_l.dims()[0] {
                        break;
                    }

                    let row_offset = weight_y * input_l.stride()[0];
                    for weight_x in start_x..end_x {
                        if weight_x >= input_l.dims()[1] {
                            break;
                        }

                        let pos = row_offset + weight_x;
                        let val = input[pos].to_f64() as f32;
                        let abs_val = val.abs();
                        if abs_val > max_abs {
                            max_abs = abs_val;
                        }
                    }
                }

                // Calculate scale
                let block_scale = if max_abs > 0.0 {
                    max_abs / 448.0
                } else {
                    1e-12
                };

                // SAFETY: We know each thread will only update independent values!
                unsafe {
                    *scale_ptr.wrapping_add(y * grid_x + x) = block_scale;
                }

                // Quantize values
                for weight_y in start_y..end_y {
                    if weight_y >= input_l.dims()[0] {
                        break;
                    }

                    let row_offset = weight_y * input_l.stride()[0];
                    for weight_x in start_x..end_x {
                        if weight_x >= input_l.dims()[1] {
                            break;
                        }

                        let pos = row_offset + weight_x;
                        let val = input[pos].to_f64() as f32;
                        let scaled_val = (val / block_scale).clamp(-448.0, 448.0);

                        // SAFETY: We know each thread will only update independent values!
                        unsafe {
                            *weight_ptr.wrapping_add(pos) = F8E4M3::from_f32(scaled_val);
                        }
                    }
                }
            });
        });

        Ok((weight, scale))
    }
}

impl CustomOp1 for Fp8BlockwiseQuantize {
    fn name(&self) -> &'static str {
        "fp8-blockwise-quantize"
    }

    fn cpu_fwd(
        &self,
        input_s: &candle_core::CpuStorage,
        input_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }
        if input_l.dims().len() != 2 {
            candle_core::bail!("Expected input to be rank 2");
        }
        if self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected weight_block_size to have length 2");
        }

        let grid_y = input_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = input_l.dim(1)?.div_ceil(self.weight_block_size[1]);

        let (weight, scale) = match input_s {
            CpuStorage::F32(input) => self.dispatch_quant_blockwise(input, input_l)?,
            CpuStorage::F16(input) => self.dispatch_quant_blockwise(input, input_l)?,
            CpuStorage::BF16(input) => self.dispatch_quant_blockwise(input, input_l)?,
            other => candle_core::bail!("unexpected input type for fp8 blockwise quant: {other:?}"),
        };

        // Return both weight and scale tensors packed into a single storage
        // We'll need to unpack them after the op
        let mut packed = Vec::with_capacity(weight.len() + scale.len());
        packed.extend_from_slice(&weight);

        // Convert scale to F8E4M3 for storage (will convert back when unpacking)
        for &s in &scale {
            packed.push(F8E4M3::from_f32(s));
        }

        Ok((
            CpuStorage::F8E4M3(packed),
            candle_core::Shape::from_dims(&[
                input_l.dims()[0] + grid_y,
                input_l.dims()[1].max(grid_x),
            ]),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input_s: &candle_core::CudaStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, CudaStorage};
        use half::{bf16, f16};

        use crate::{blockwise_fp8::ffi, utils::slice_ptr};

        if !ffi::HAVE_BLOCKWISE_QUANT_KERNELS {
            candle_core::bail!("Do not have blockwise FP8 quant kernels.");
        }

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }
        if input_l.dims().len() != 2 {
            candle_core::bail!("Expected input to be rank 2");
        }
        if self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected weight_block_size to have length 2");
        }

        let dev = input_s.device();

        let weight_height = input_l.dim(0)? as i32;
        let weight_block_size_y = self.weight_block_size[0] as i32;
        let weight_width = input_l.dim(1)? as i32;
        let weight_block_size_x = self.weight_block_size[1] as i32;
        let weight_row_stride = input_l.stride()[0] as i32;

        let grid_y = input_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = input_l.dim(1)?.div_ceil(self.weight_block_size[1]);
        let scale_stride = grid_x as i32;

        // Allocate output buffers
        let weight_output = dev.alloc_zeros::<F8E4M3>(input_l.shape().elem_count())?;
        let scale_output = dev.alloc_zeros::<f32>(grid_y * grid_x)?;

        let (weight_ptr, weight_guard) = slice_ptr(&weight_output, 0);
        let (scale_ptr, scale_guard) = slice_ptr(&scale_output, 0);

        match input_s.dtype() {
            DType::F32 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<f32>()?, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f32(
                        input as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            DType::F16 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<f16>()?, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f16(
                        input as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            DType::BF16 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<bf16>()?, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_bf16(
                        input as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            other => candle_core::bail!("unexpected input type for fp8 blockwise quant: {other:?}"),
        }

        drop(weight_guard);
        drop(scale_guard);

        // Return just the weight tensor - we'll handle scale separately
        let res = CudaStorage::wrap_cuda_slice(weight_output, input_s.device().clone());
        Ok((res, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        _input_s: &candle_core::MetalStorage,
        _input_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        candle_core::bail!("FP8 blockwise quantization not yet implemented for Metal");
    }
}

/// FP8 blockwise quantize.
/// - Expects input to be f32, f16, or bf16
/// - Returns a tuple of (quantized_weight, scales)
/// - quantized_weight is fp8
/// - scales is f32
pub fn fp8_blockwise_quantize(
    #[allow(unused_variables)] input: &Tensor,
    #[allow(unused_variables)] weight_block_size: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    // Since CustomOp1 only returns a single tensor, we need a different approach
    // Let's implement this using the CUDA kernels directly
    #[cfg(feature = "cuda")]
    {
        use candle_core::{CudaStorage, Device, Storage};
        use half::{bf16, f16};

        use crate::{blockwise_fp8::ffi, utils::slice_ptr};

        if !matches!(input.device(), Device::Cuda(_)) {
            candle_core::bail!("FP8 blockwise quantization only supported on CUDA for now");
        }

        if !ffi::HAVE_BLOCKWISE_QUANT_KERNELS {
            candle_core::bail!("Do not have blockwise FP8 quant kernels.");
        }

        let input_l = input.layout();
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }
        if input.dims().len() != 2 {
            candle_core::bail!("Expected input to be rank 2");
        }
        if weight_block_size.len() != 2 {
            candle_core::bail!("Expected weight_block_size to have length 2");
        }

        let dev = match input.device() {
            Device::Cuda(dev) => dev,
            _ => unreachable!(),
        };

        let weight_height = input.dim(0)? as i32;
        let weight_block_size_y = weight_block_size[0] as i32;
        let weight_width = input.dim(1)? as i32;
        let weight_block_size_x = weight_block_size[1] as i32;
        let weight_row_stride = input_l.stride()[0] as i32;

        let grid_y = input.dim(0)?.div_ceil(weight_block_size[0]);
        let grid_x = input.dim(1)?.div_ceil(weight_block_size[1]);
        let scale_stride = grid_x as i32;

        // Allocate output buffers
        let weight_output = dev.alloc_zeros::<F8E4M3>(input.shape().elem_count())?;
        let scale_output = dev.alloc_zeros::<f32>(grid_y * grid_x)?;

        let (weight_ptr, _weight_guard) = slice_ptr(&weight_output, 0);
        let (scale_ptr, _scale_guard) = slice_ptr(&scale_output, 0);

        match input.dtype() {
            DType::F32 => {
                let input_storage = input.storage_and_layout().0;
                let input_s = match &*input_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
                    _ => candle_core::bail!("Expected CUDA storage"),
                };
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f32(
                        input_ptr as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            DType::F16 => {
                let input_storage = input.storage_and_layout().0;
                let input_s = match &*input_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                    _ => candle_core::bail!("Expected CUDA storage"),
                };
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f16(
                        input_ptr as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            DType::BF16 => {
                let input_storage = input.storage_and_layout().0;
                let input_s = match &*input_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                    _ => candle_core::bail!("Expected CUDA storage"),
                };
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_bf16(
                        input_ptr as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            other => candle_core::bail!("unexpected input type for fp8 blockwise quant: {other:?}"),
        }

        // Drop guards before moving the buffers
        drop(_weight_guard);
        drop(_scale_guard);

        // Create weight tensor by wrapping the CUDA storage
        let weight_storage = CudaStorage::wrap_cuda_slice(weight_output, dev.clone());
        let weight = Tensor::from((Storage::Cuda(weight_storage), input.shape().clone()));

        // Create scale tensor
        let scale_storage = CudaStorage::wrap_cuda_slice(scale_output, dev.clone());
        let scale = Tensor::from((
            Storage::Cuda(scale_storage),
            candle_core::Shape::from_dims(&[grid_y, grid_x]),
        ));

        Ok((weight, scale))
    }

    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("FP8 blockwise quantization requires CUDA feature");
    }
}

/// Decode-path dispatch threshold for the blockwise-FP8 GEMV kernel.
///
/// `M <= this` routes to the warp-per-row GEMV instead of the tiled GEMM
/// (which at M == 1 wastes 31/32 of every 32x32 tile). Defaults to 4;
/// `ARC_FP8_GEMV_MAX_M=<n>` overrides, `ARC_NO_FP8_GEMV` disables (0).
/// Read once (LazyLock) so the decode hot loop never touches the env.
#[cfg(feature = "cuda")]
fn fp8_gemv_max_m() -> usize {
    use std::sync::LazyLock;
    static MAX_M: LazyLock<usize> = LazyLock::new(|| {
        if std::env::var("ARC_NO_FP8_GEMV").is_ok() {
            return 0;
        }
        std::env::var("ARC_FP8_GEMV_MAX_M")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
    });
    *MAX_M
}

/// FP8 blockwise matmul.
/// Computes output = input @ weight.T where weight is FP8 blockwise quantized.
/// - input: [M, K] in fp16/bf16
/// - weight: [N, K] in FP8 with blockwise scales
/// - scales: [N/block_y, K/block_x] in f32
/// - output: [M, N] in fp16/bf16
///
/// Dispatch: `M <= fp8_gemv_max_m()` (default 4, decode regime) routes to a
/// dedicated warp-per-row GEMV kernel that dequantizes FP8 blocks in
/// registers; larger `M` keeps the tiled GEMM.
#[cfg(feature = "cuda")]
pub fn fp8_blockwise_matmul(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    weight_block_size: &[usize],
) -> Result<Tensor> {
    fp8_blockwise_matmul_impl(input, weight, scales, weight_block_size, None)
}

/// Internal worker for [`fp8_blockwise_matmul`] with an explicit kernel
/// override for tests: `force_gemv = Some(true)` forces the GEMV kernel,
/// `Some(false)` forces the tiled GEMM, `None` uses the M-based dispatch.
/// The GEMV alignment preconditions (`K % 4 == 0`, `block_size_x % 4 == 0`)
/// are always enforced regardless of the override.
#[cfg(feature = "cuda")]
fn fp8_blockwise_matmul_impl(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    weight_block_size: &[usize],
    force_gemv: Option<bool>,
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Storage};
    use half::{bf16, f16};

    use crate::{blockwise_fp8::ffi, utils::slice_ptr};

    if !ffi::HAVE_BLOCKWISE_GEMM_KERNELS {
        candle_core::bail!("Do not have blockwise FP8 GEMM kernels.");
    }

    if !matches!(input.device(), Device::Cuda(_)) {
        candle_core::bail!("FP8 blockwise matmul only supported on CUDA");
    }

    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let scales = scales.contiguous()?;

    if input.dims().len() != 2 {
        candle_core::bail!("Expected input to be rank 2, got {:?}", input.dims());
    }
    if weight.dims().len() != 2 {
        candle_core::bail!("Expected weight to be rank 2, got {:?}", weight.dims());
    }
    if weight.dtype() != DType::F8E4M3 {
        candle_core::bail!("Expected FP8 weight, got {:?}", weight.dtype());
    }

    let m = input.dim(0)? as i32;
    let k = input.dim(1)? as i32;
    let n = weight.dim(0)? as i32;

    if weight.dim(1)? as i32 != k {
        candle_core::bail!(
            "Weight K dimension {} doesn't match input K dimension {}",
            weight.dim(1)?,
            k
        );
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };

    let block_size_y = weight_block_size[0] as i32;
    let block_size_x = weight_block_size[1] as i32;
    let scale_row_stride = scales.dim(1)? as i32;

    // Decode-regime GEMV dispatch. The alignment preconditions guarantee the
    // kernel's 32-bit vectorized FP8 loads and that a 4-wide K-group never
    // straddles a scale block; shapes that fail them keep the tiled GEMM.
    let gemv_aligned = k % 4 == 0 && block_size_x % 4 == 0;
    let use_gemv = force_gemv.unwrap_or_else(|| (m as usize) <= fp8_gemv_max_m()) && gemv_aligned;

    let input_l = input.layout();
    let weight_l = weight.layout();
    let scales_l = scales.layout();

    let input_storage = input.storage_and_layout().0;
    let weight_storage = weight.storage_and_layout().0;
    let scales_storage = scales.storage_and_layout().0;

    let weight_s = match &*weight_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<F8E4M3>()?,
        _ => candle_core::bail!("Expected CUDA storage for weight"),
    };
    let scales_s = match &*scales_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("Expected CUDA storage for scales"),
    };

    let (weight_ptr, _weight_guard) = slice_ptr(weight_s, weight_l.start_offset());
    let (scales_ptr, _scales_guard) = slice_ptr(scales_s, scales_l.start_offset());

    match input.dtype() {
        DType::F16 => {
            let output = dev.alloc_zeros::<f16>((m * n) as usize)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                if use_gemv {
                    unsafe {
                        ffi::launch_fp8_gemv_f16(
                            input_ptr as *const _,
                            weight_ptr as *const _,
                            scales_ptr as *const _,
                            output_ptr as *mut _,
                            m,
                            n,
                            k,
                            scale_row_stride,
                            block_size_y,
                            block_size_x,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                } else {
                    unsafe {
                        ffi::launch_fp8_matmul_f16(
                            input_ptr as *const _,
                            weight_ptr as *const _,
                            scales_ptr as *const _,
                            output_ptr as *mut _,
                            m,
                            n,
                            k,
                            scale_row_stride,
                            block_size_y,
                            block_size_x,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                }
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[m as usize, n as usize]),
            )))
        }
        DType::BF16 => {
            let output = dev.alloc_zeros::<bf16>((m * n) as usize)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                if use_gemv {
                    unsafe {
                        ffi::launch_fp8_gemv_bf16(
                            input_ptr as *const _,
                            weight_ptr as *const _,
                            scales_ptr as *const _,
                            output_ptr as *mut _,
                            m,
                            n,
                            k,
                            scale_row_stride,
                            block_size_y,
                            block_size_x,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                } else {
                    unsafe {
                        ffi::launch_fp8_matmul_bf16(
                            input_ptr as *const _,
                            weight_ptr as *const _,
                            scales_ptr as *const _,
                            output_ptr as *mut _,
                            m,
                            n,
                            k,
                            scale_row_stride,
                            block_size_y,
                            block_size_x,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                }
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[m as usize, n as usize]),
            )))
        }
        other => candle_core::bail!("Unsupported input dtype for FP8 matmul: {:?}", other),
    }
}

/// FP8 indexed MoE GEMM for gather_forward.
/// Computes indexed matmul for MoE where each token selects specific experts.
/// - input: [num_tokens, 1, K] or [num_tokens, topk, K] in fp16/bf16
/// - weights: [num_experts, N, K] in FP8 with blockwise scales
/// - scales: [num_experts, N/block_y, K/block_x] in f32
/// - indices: [num_tokens, topk] in i32
/// - output: [num_tokens, topk, N] in fp16/bf16
#[cfg(feature = "cuda")]
pub fn fp8_indexed_moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    scales: &Tensor,
    indices: &Tensor,
    weight_block_size: &[usize],
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Storage};
    use half::{bf16, f16};

    use crate::{blockwise_fp8::ffi, utils::slice_ptr};

    if !ffi::HAVE_BLOCKWISE_GEMM_KERNELS {
        candle_core::bail!("Do not have blockwise FP8 GEMM kernels.");
    }

    if !matches!(input.device(), Device::Cuda(_)) {
        candle_core::bail!("FP8 indexed MoE GEMM only supported on CUDA");
    }

    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let scales = scales.contiguous()?;
    let indices = indices.contiguous()?;

    // Determine input shape
    // Input can be [num_tokens, 1, K] or [num_tokens, topk, K]
    let (num_tokens, input_has_topk_dim, k) = if input.dims().len() == 3 {
        let dims = input.dims3()?;
        (dims.0, dims.1 > 1, dims.2)
    } else if input.dims().len() == 2 {
        let dims = input.dims2()?;
        (dims.0, false, dims.1)
    } else {
        candle_core::bail!("Expected input to be rank 2 or 3, got {:?}", input.dims());
    };

    // Get topk from indices
    let (indices_tokens, topk) = indices.dims2()?;
    if indices_tokens != num_tokens {
        candle_core::bail!(
            "Indices num_tokens {} doesn't match input num_tokens {}",
            indices_tokens,
            num_tokens
        );
    }

    // Weights shape: [num_experts, N, K]
    if weights.dims().len() != 3 {
        candle_core::bail!("Expected weights to be rank 3, got {:?}", weights.dims());
    }
    let (num_experts, n, weight_k) = weights.dims3()?;
    if weight_k != k {
        candle_core::bail!(
            "Weights K dimension {} doesn't match input K dimension {}",
            weight_k,
            k
        );
    }

    if weights.dtype() != DType::F8E4M3 {
        candle_core::bail!("Expected FP8 weights, got {:?}", weights.dtype());
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };

    let block_size_y = weight_block_size[0] as i32;
    let block_size_x = weight_block_size[1] as i32;

    // Scales shape should be [num_experts, N/block_y, K/block_x]
    let scale_row_stride = scales.dim(2)? as i32; // K/block_x

    let input_l = input.layout();
    let weights_l = weights.layout();
    let scales_l = scales.layout();
    let indices_l = indices.layout();

    let input_storage = input.storage_and_layout().0;
    let weights_storage = weights.storage_and_layout().0;
    let scales_storage = scales.storage_and_layout().0;
    let indices_storage = indices.storage_and_layout().0;

    let weights_s = match &*weights_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<F8E4M3>()?,
        _ => candle_core::bail!("Expected CUDA storage for weights"),
    };
    let scales_s = match &*scales_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("Expected CUDA storage for scales"),
    };
    let indices_s = match &*indices_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("Expected CUDA storage for indices"),
    };

    let (weights_ptr, _weights_guard) = slice_ptr(weights_s, weights_l.start_offset());
    let (scales_ptr, _scales_guard) = slice_ptr(scales_s, scales_l.start_offset());
    let (indices_ptr, _indices_guard) = slice_ptr(indices_s, indices_l.start_offset());

    match input.dtype() {
        DType::F16 => {
            let output = dev.alloc_zeros::<f16>(num_tokens * topk * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                unsafe {
                    ffi::launch_fp8_indexed_moe_gemm_f16(
                        input_ptr as *const _,
                        weights_ptr as *const _,
                        scales_ptr as *const _,
                        indices_ptr as *const _,
                        output_ptr as *mut _,
                        num_tokens as i32,
                        topk as i32,
                        num_experts as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride,
                        block_size_y,
                        block_size_x,
                        input_has_topk_dim,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[num_tokens, topk, n]),
            )))
        }
        DType::BF16 => {
            let output = dev.alloc_zeros::<bf16>(num_tokens * topk * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                unsafe {
                    ffi::launch_fp8_indexed_moe_gemm_bf16(
                        input_ptr as *const _,
                        weights_ptr as *const _,
                        scales_ptr as *const _,
                        indices_ptr as *const _,
                        output_ptr as *mut _,
                        num_tokens as i32,
                        topk as i32,
                        num_experts as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride,
                        block_size_y,
                        block_size_x,
                        input_has_topk_dim,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[num_tokens, topk, n]),
            )))
        }
        other => candle_core::bail!(
            "Unsupported input dtype for FP8 indexed MoE GEMM: {:?}",
            other
        ),
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{Linear, Module};
    use half::bf16;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

    use crate::{blockwise_fp8::ops, safetensors::MmapedSafetensors};

    #[test]
    fn test_fp8_blockwise_dequant() -> Result<()> {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
        let weight_block_size = vec![2, 2];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

        let res = dequant.to_vec2::<f32>()?;
        assert_eq!(
            res,
            vec![
                vec![0., 0., 1., 1., 2.],
                vec![0., 0., 1., 1., 2.],
                vec![3., 3., 4., 4., 5.],
                vec![3., 3., 4., 4., 5.],
                vec![6., 6., 7., 7., 8.],
            ]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_cuda() -> Result<()> {
        let truth = {
            let dev = &Device::Cpu;
            let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant =
                ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

            dequant.to_vec2::<f32>()?
        };
        let test = {
            let dev = &Device::new_cuda(0)?;
            // Create FP8 weight by first creating on CPU then moving to CUDA
            let weight_cpu = Tensor::ones((5, 5), DType::F8E4M3, &Device::Cpu)?;
            let weight = weight_cpu.to_device(dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant =
                ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

            dequant.to_vec2::<f32>()?
        };

        assert_eq!(test, truth);
        assert_eq!(
            test,
            vec![
                vec![0., 0., 1., 1., 2.],
                vec![0., 0., 1., 1., 2.],
                vec![3., 3., 4., 4., 5.],
                vec![3., 3., 4., 4., 5.],
                vec![6., 6., 7., 7., 8.],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_fp8_blockwise_dequant_bf16() -> Result<()> {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
        let weight_block_size = vec![2, 2];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::BF16)?;

        let res = dequant.to_vec2::<bf16>()?;
        assert_eq!(
            res,
            vec![
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(6.),
                    bf16::from_f32(6.),
                    bf16::from_f32(7.),
                    bf16::from_f32(7.),
                    bf16::from_f32(8.)
                ],
            ]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_cuda_bf16() -> Result<()> {
        let truth = {
            let dev = &Device::Cpu;
            let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant = ops::fp8_blockwise_dequantize(
                &weight,
                &inv_scales,
                weight_block_size,
                DType::BF16,
            )?;

            dequant.to_vec2::<bf16>()?
        };
        let test = {
            let dev = &Device::new_cuda(0)?;
            // Create FP8 weight by first creating on CPU then moving to CUDA
            let weight_cpu = Tensor::ones((5, 5), DType::F8E4M3, &Device::Cpu)?;
            let weight = weight_cpu.to_device(dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant = ops::fp8_blockwise_dequantize(
                &weight,
                &inv_scales,
                weight_block_size,
                DType::BF16,
            )?;

            dequant.to_vec2::<bf16>()?
        };

        assert_eq!(test, truth);
        assert_eq!(
            test,
            vec![
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(6.),
                    bf16::from_f32(6.),
                    bf16::from_f32(7.),
                    bf16::from_f32(7.),
                    bf16::from_f32(8.)
                ],
            ]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_quant_dequant_roundtrip() -> Result<()> {
        let dev = &Device::new_cuda(0)?;

        // Create test input
        let input = Tensor::randn(0f32, 2f32, (8, 8), dev)?;
        let weight_block_size = vec![4, 4];

        // Quantize
        let (quantized, scales) = ops::fp8_blockwise_quantize(&input, weight_block_size.clone())?;

        // Verify shapes
        assert_eq!(quantized.shape(), input.shape());
        assert_eq!(scales.dims2()?, (2, 2)); // 8/4 = 2 blocks in each dimension

        // Dequantize
        let dequantized =
            ops::fp8_blockwise_dequantize(&quantized, &scales, weight_block_size, input.dtype())?;

        // Check that shapes match
        assert_eq!(dequantized.shape(), input.shape());

        // The values won't be exactly the same due to quantization loss,
        // but they should be reasonably close
        let input_vec = input.to_vec2::<f32>()?;
        let dequant_vec = dequantized.to_vec2::<f32>()?;

        let mut max_error = 0f32;
        for (row_in, row_out) in input_vec.iter().zip(dequant_vec.iter()) {
            for (val_in, val_out) in row_in.iter().zip(row_out.iter()) {
                let error = (val_in - val_out).abs();
                max_error = max_error.max(error);
            }
        }

        // FP8 E4M3 has limited precision, so we expect some error
        // but it should be reasonable
        assert!(max_error < 0.16, "Max error {} is too large", max_error);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_blockwise_fp8_gemm() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;

        let api = ApiBuilder::new().with_progress(true).build().unwrap();
        let api = api.repo(Repo::with_revision(
            "EricB/mistralrs_tests".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let filename = api.get("test_fp8.safetensors").unwrap();
        let vb = unsafe { MmapedSafetensors::new(filename)? };

        let weight = vb.load("weight", &dev, None)?;
        assert_eq!((7168, 2048), weight.dims2()?);
        assert_eq!(DType::F8E4M3, weight.dtype());

        let scale = vb.load("scale", &dev, None)?;
        assert_eq!((56, 16), scale.dims2()?);
        assert_eq!(DType::F32, scale.dtype());

        let weight_block_size = vec![128, 128];

        // in dim is 2048.
        let xs = Tensor::randn(0f32, 1f32, (32, 2048), &dev)?.to_dtype(DType::BF16)?;

        let truth = {
            let weight_dq =
                ops::fp8_blockwise_dequantize(&weight, &scale, weight_block_size, DType::BF16)?;

            let lin_dq = Linear::new(weight_dq, None);
            lin_dq.forward(&xs)?
        };

        // TODO: will be adding real blockwise fp8 gemm shortly ;)
        assert_eq!((32, 7168), truth.dims2()?);

        Ok(())
    }

    /// Build a random blockwise-FP8 weight + scales pair on `dev` and return
    /// `(weight_fp8, scales)`. Shared by the GEMV parity tests.
    #[cfg(feature = "cuda")]
    fn make_fp8_weight(
        dev: &Device,
        n: usize,
        k: usize,
        block: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let w_f32 = Tensor::randn(0f32, 1f32, (n, k), dev)?;
        ops::fp8_blockwise_quantize(&w_f32, block.to_vec())
    }

    /// Max elementwise deviation |a - b| / max(|b|, 1) between two flattened
    /// outputs.
    #[cfg(feature = "cuda")]
    fn max_rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a: Vec<f32> = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        assert_eq!(a.len(), b.len());
        let mut max_err = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let err = (x - y).abs() / y.abs().max(1.0);
            max_err = max_err.max(err);
        }
        Ok(max_err)
    }

    /// GEMV kernel (M=1 decode shape) must match both the tiled GEMM kernel
    /// and the dequantize+matmul reference within FP8 tolerance.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_gemv_matches_gemm_m1() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping test_fp8_gemv_matches_gemm_m1");
                return Ok(());
            }
        };
        let (n, k) = (512, 1024);
        let block = vec![128usize, 128usize];
        let (weight, scales) = make_fp8_weight(&dev, n, k, &block)?;

        let x = Tensor::randn(0f32, 1f32, (1, k), &dev)?.to_dtype(DType::BF16)?;

        let gemv = ops::fp8_blockwise_matmul_impl(&x, &weight, &scales, &block, Some(true))?;
        let gemm = ops::fp8_blockwise_matmul_impl(&x, &weight, &scales, &block, Some(false))?;
        assert_eq!(gemv.dims2()?, (1, n));

        // GEMV vs tiled GEMM: same dequant math, different accumulation
        // order + BF16 rounding of the output.
        let err_kernels = max_rel_err(&gemv, &gemm)?;
        assert!(
            err_kernels < 1e-2,
            "GEMV vs GEMM max rel err {err_kernels} exceeds 1e-2"
        );

        // GEMV vs dequantize + matmul reference.
        let w_dq = ops::fp8_blockwise_dequantize(&weight, &scales, block.clone(), DType::BF16)?;
        let truth = Linear::new(w_dq, None).forward(&x)?;
        let err_ref = max_rel_err(&gemv, &truth)?;
        assert!(
            err_ref < 1e-2,
            "GEMV vs dequant reference max rel err {err_ref} exceeds 1e-2"
        );

        // The default dispatch must route M=1 to the GEMV automatically.
        let auto = ops::fp8_blockwise_matmul(&x, &weight, &scales, &block)?;
        let err_auto = max_rel_err(&auto, &gemv)?;
        assert!(
            err_auto < 1e-2,
            "auto dispatch (M=1) vs forced GEMV max rel err {err_auto} exceeds 1e-2"
        );

        Ok(())
    }

    /// Small-batch shapes (M = 2..4, grid.y > 1) and a remainder-loop K
    /// (K % 128 != 0 but K % 4 == 0) must also match the tiled GEMM.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_gemv_matches_gemm_small_m() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping test_fp8_gemv_matches_gemm_small_m");
                return Ok(());
            }
        };
        let block = vec![128usize, 128usize];
        for (m, n, k) in [(2usize, 384usize, 512usize), (4, 256, 324)] {
            let (weight, scales) = make_fp8_weight(&dev, n, k, &block)?;
            let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(DType::BF16)?;

            let gemv = ops::fp8_blockwise_matmul_impl(&x, &weight, &scales, &block, Some(true))?;
            let gemm = ops::fp8_blockwise_matmul_impl(&x, &weight, &scales, &block, Some(false))?;
            assert_eq!(gemv.dims2()?, (m, n));

            let err = max_rel_err(&gemv, &gemm)?;
            assert!(
                err < 1e-2,
                "GEMV vs GEMM (m={m}, n={n}, k={k}) max rel err {err} exceeds 1e-2"
            );
        }
        Ok(())
    }

    /// The GEMV kernel must be deterministic: identical inputs produce
    /// bitwise-identical outputs across runs (fixed reduction order, no
    /// atomics).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_gemv_deterministic() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping test_fp8_gemv_deterministic");
                return Ok(());
            }
        };
        let (n, k) = (512, 1024);
        let block = vec![128usize, 128usize];
        let (weight, scales) = make_fp8_weight(&dev, n, k, &block)?;
        let x = Tensor::randn(0f32, 1f32, (1, k), &dev)?.to_dtype(DType::BF16)?;

        let first: Vec<bf16> =
            ops::fp8_blockwise_matmul_impl(&x, &weight, &scales, &block, Some(true))?
                .flatten_all()?
                .to_vec1()?;
        for run in 1..3 {
            let again: Vec<bf16> =
                ops::fp8_blockwise_matmul_impl(&x, &weight, &scales, &block, Some(true))?
                    .flatten_all()?
                    .to_vec1()?;
            assert_eq!(first, again, "GEMV output diverged on run {run}");
        }
        Ok(())
    }
}
