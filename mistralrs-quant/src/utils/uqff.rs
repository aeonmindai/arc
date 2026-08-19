use byteorder::{LittleEndian, ReadBytesExt};

use candle_core::{DType, Device, Result, Tensor, WithDType};
use float8::F8E4M3;
use half::{bf16, f16};

// v0.1.0: initial release
// v0.1.1: add i16 dtype
// v0.1.2: add F8E4M3
// v0.1.3: add AFQ
// v0.2.0: add f4/f6e3m2/f6e2m3/f8e8m0 type handling
// v0.2.1: QTIP 3-D stacked-expert (MoE) payloads. Tensor payloads are
//         self-describing (rank + dims), so 2-D QTIP payloads are unchanged;
//         readers older than this version mis-decode rank-3 QTIP payloads
//         rather than failing cleanly, so the bump records provenance.
// v0.3.0: QTIP search stamp. Both QTIP rungs append a trailing provenance byte
//         (1 = trellis, 2 = greedy) naming the search that produced the blocks,
//         and refuse a greedy-stamped payload at load (DOCTRINE D4).
//         This is a MINOR bump, not a patch, on purpose: `version_is_compatible`
//         only gates on major/minor, so a 0.2.x reader would happily accept a
//         0.2.2 payload and then ignore the stamp — i.e. exactly the silent
//         mis-decode that the v0.2.1 note warns about. With 0.3.0 an older
//         binary fails cleanly ("newer than this build supports") instead of
//         serving weights whose provenance it cannot check, while this build
//         still reads every 0.1.x/0.2.x artifact (see
//         `QtipSearchStamp::enforce_at_load` for the legacy policy).

const UQFF_VERSION_MAJOR: u32 = 0;
const UQFF_VERSION_MINOR: u32 = 3;
const UQFF_VERSION_PATCH: u32 = 0;

/// Format 4 bytes, little endian: [ UNSPECIFIED ] [ MAJOR ] [ MINOR ] [ PATCH ]
pub(crate) const UQFF_VERSION: u32 =
    (UQFF_VERSION_MAJOR << (8 * 2)) | (UQFF_VERSION_MINOR << 8) | UQFF_VERSION_PATCH;
/// Offset for the quant type. UQFF always serializes the version first.
pub const UQFF_QUANT_TYPE_OFFSET: usize = std::mem::size_of::<u32>();

/// Check if major version matches: is backwards compatible
pub(crate) fn version_is_compatible(version: u32) -> Result<()> {
    let major = version >> (8 * 2);
    let minor = (version >> 8) & 0xFF;
    let patch = version & 0xFF;

    if major != UQFF_VERSION_MAJOR {
        candle_core::bail!("Major version of ISQ artifact file ({major}) does not match the implementation in this build ({UQFF_VERSION_MAJOR})");
    }

    // Check minor version for forward compatibility
    if minor > UQFF_VERSION_MINOR {
        candle_core::bail!("Minor version of ISQ artifact file ({major}.{minor}.{patch}) is newer than this build supports ({UQFF_VERSION_MAJOR}.{UQFF_VERSION_MINOR}.{UQFF_VERSION_PATCH}). Please update mistral.rs.");
    }

    Ok(())
}

// -----------------------
// Tensor dtype, u32, little endian
// -----------------------
pub(crate) fn write_dtype(dtype: DType, buffer: &mut Vec<u8>) {
    let dtype: u32 = match dtype {
        DType::U8 => 0,
        DType::U32 => 1,
        DType::I32 => 2,
        DType::I64 => 3,
        DType::F16 => 4,
        DType::BF16 => 5,
        DType::F32 => 6,
        DType::F64 => 7,
        DType::I16 => 8,
        DType::F8E4M3 => 9,
        DType::F6E2M3 => 10,
        DType::F6E3M2 => 11,
        DType::F4 => 12,
        DType::F8E8M0 => 13,
    };
    buffer.extend(&dtype.to_le_bytes());
}

pub(crate) fn read_dtype<R: std::io::Read>(buffer: &mut R) -> Result<DType> {
    let dtype = buffer.read_u32::<LittleEndian>()?;
    let dtype = match dtype {
        0 => DType::U8,
        1 => DType::U32,
        2 => DType::I32,
        3 => DType::I64,
        4 => DType::F16,
        5 => DType::BF16,
        6 => DType::F32,
        7 => DType::F64,
        8 => DType::I16,
        9 => DType::F8E4M3,
        10 => DType::F6E2M3,
        11 => DType::F6E3M2,
        12 => DType::F4,
        13 => DType::F8E8M0,
        _ => candle_core::bail!("unknown dtype for quantized tensor {dtype}"),
    };
    Ok(dtype)
}

// -----------------------
// Tensor data length, u32, little endian
// -----------------------
// Tensor dtype, u32, little endian
// -----------------------
// Num shape dims, u32, little endian
// -----------------------
// ...
// Array (in original order): shape dims, u32, little endian
// ...
// -----------------------
// ...
// Array: tensor data, u8s
// ...
// -----------------------

/// Write the fixed-layout tensor header: data length, dtype, rank, dims.
fn write_tensor_header(
    buffer: &mut Vec<u8>,
    data_len: usize,
    dtype: DType,
    b_shape: &[usize],
) -> Result<()> {
    // Check for potential overflow when converting usize to u32
    if data_len > u32::MAX as usize {
        candle_core::bail!(
            "Tensor data too large for UQFF format: {} bytes exceeds u32::MAX",
            data_len
        );
    }
    buffer.extend(&(data_len as u32).to_le_bytes());

    // DType
    write_dtype(dtype, buffer);

    // Shape
    let shape_len = b_shape.len();
    if shape_len > u32::MAX as usize {
        candle_core::bail!(
            "Tensor has too many dimensions for UQFF format: {} exceeds u32::MAX",
            shape_len
        );
    }
    buffer.extend((shape_len as u32).to_le_bytes());
    for dim in b_shape {
        if *dim > u32::MAX as usize {
            candle_core::bail!(
                "Tensor dimension too large for UQFF format: {} exceeds u32::MAX",
                dim
            );
        }
        buffer.extend((*dim as u32).to_le_bytes());
    }
    Ok(())
}

pub(crate) fn serialize_tensor(buffer: &mut Vec<u8>, tensor: &Tensor) -> Result<()> {
    let b_shape = tensor.dims();
    let tensor = tensor.flatten_all()?;
    let dtype = tensor.dtype();

    /// Pull the tensor to host as `Vec<T>`, then write header + data by
    /// BORROWING its bytes.
    ///
    /// The borrow is the point. The previous implementation rebuilt the
    /// `Vec<T>` as a `Vec<u8>` via `Vec::from_raw_parts` so it could hand back
    /// an owned `Vec<u8>`; see `data_to_bytes`'s removal for why that was
    /// undefined behaviour. Keeping the `Vec<T>` alive and lending `&[u8]` is
    /// both sound and copy-free.
    macro_rules! serialize_as {
        ($t:ty) => {{
            let vs: Vec<$t> = tensor.to_vec1()?;
            let bytes = as_byte_slice(vs.as_slice());
            write_tensor_header(buffer, bytes.len(), dtype, b_shape)?;
            buffer.extend_from_slice(bytes);
        }};
    }

    match dtype {
        DType::U8 => serialize_as!(u8),
        DType::U32 => serialize_as!(u32),
        DType::I16 => serialize_as!(i16),
        DType::I32 => serialize_as!(i32),
        DType::I64 => serialize_as!(i64),
        DType::F16 => serialize_as!(half::f16),
        DType::BF16 => serialize_as!(half::bf16),
        DType::F32 => serialize_as!(f32),
        DType::F64 => serialize_as!(f64),
        DType::F8E4M3 => serialize_as!(F8E4M3),
        DType::F4 | DType::F6E3M2 | DType::F6E2M3 | DType::F8E8M0 => {
            candle_core::bail!("f4/f6e3m2/f6e2m3/f8e8m0 tensors cannot be serialized.")
        }
    }

    Ok(())
}

pub(crate) fn deserialize_tensor<R: std::io::Read>(
    buffer: &mut R,
    device: &Device,
) -> Result<Tensor> {
    let data_len = buffer.read_u32::<LittleEndian>()? as usize;

    // DType
    let dtype = read_dtype(buffer)?;

    let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        dims.push(buffer.read_u32::<LittleEndian>()? as usize)
    }

    let mut tensor_data = vec![0; data_len];
    buffer.read_exact(&mut tensor_data)?;

    match dtype {
        DType::F16 => bytes_to_data::<f16>(&tensor_data, &dims, device),
        DType::BF16 => bytes_to_data::<bf16>(&tensor_data, &dims, device),
        DType::F32 => bytes_to_data::<f32>(&tensor_data, &dims, device),
        DType::F64 => bytes_to_data::<f64>(&tensor_data, &dims, device),
        DType::I32 => bytes_to_data::<i32>(&tensor_data, &dims, device),
        DType::I64 => bytes_to_data::<i64>(&tensor_data, &dims, device),
        DType::I16 => bytes_to_data::<i16>(&tensor_data, &dims, device),
        DType::U32 => bytes_to_data::<u32>(&tensor_data, &dims, device),
        DType::U8 => bytes_to_data::<u8>(&tensor_data, &dims, device),
        DType::F8E4M3 => bytes_to_data::<F8E4M3>(&tensor_data, &dims, device),
        DType::F4 | DType::F6E3M2 | DType::F6E2M3 | DType::F8E8M0 => {
            candle_core::bail!("f4/f6e3m2/f6e2m3/f8e8m0 tensors cannot be deserialized.")
        }
    }
}

/// Just seek the reader ahead.
pub(crate) fn fake_deserialize_tensor<R: std::io::Read + std::io::Seek>(
    buffer: &mut R,
) -> Result<()> {
    let data_len = buffer.read_u32::<LittleEndian>()? as usize;

    // DType
    let _dtype = read_dtype(buffer)?;

    let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        dims.push(buffer.read_u32::<LittleEndian>()? as usize)
    }

    // Fake read the data in bytes
    buffer.seek_relative(data_len as i64)?;

    Ok(())
}

/// View a `&[T]` as its raw bytes, without taking ownership.
///
/// This replaces a `data_to_bytes` that rebuilt the `Vec<T>` as a `Vec<u8>`
/// with `Vec::from_raw_parts(ptr as *mut u8, len_bytes, cap_bytes)` after
/// `mem::forget`. Its safety comment read "Every T is larger than u8, so there
/// is no issue regarding alignment" — which reasons about *reading* the bytes,
/// and reading was never the problem.
///
/// The problem was DEALLOCATION. `Vec::from_raw_parts` requires the pointer to
/// have been allocated with the exact layout the resulting `Vec` will free it
/// with. The buffer was allocated as `Layout::array::<T>(cap)` — alignment 4
/// for `f32`, 8 for `f64`/`i64` — and the reconstructed `Vec<u8>` freed it as
/// `Layout::array::<u8>(cap_bytes)`, alignment 1. Passing a layout to `dealloc`
/// that differs from the one used for `alloc` is undefined behaviour, and it is
/// the kind that corrupts the allocator's own bookkeeping rather than failing
/// loudly. Every UQFF *write* of an f32/f64/i64/i32/u32/f16/bf16 tensor did it.
///
/// Borrowing the bytes is sound for the reason the old comment gestured at but
/// misapplied: `u8` has alignment 1, so any `T` slice is validly aligned when
/// viewed as bytes. The `Vec<T>` keeps its own layout and frees itself
/// correctly. Nothing is copied.
///
/// SAFETY: `T: WithDType` is one of the plain numeric types candle supports
/// (integers, floats, `f16`/`bf16`/`F8E4M3`); all are `Copy`, have no padding
/// and no uninitialised bytes, so every byte of the slice is initialised. `u8`
/// has alignment 1, so the cast pointer is always well-aligned, and the length
/// is exactly the source's byte length.
fn as_byte_slice<T: WithDType>(vs: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(vs.as_ptr() as *const u8, std::mem::size_of_val(vs)) }
}

fn bytes_to_data<T: WithDType>(
    data: &[u8],
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize).is_multiple_of(size_in_bytes) {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Round-trip every serializable dtype through the UQFF tensor encoding.
    ///
    /// This is the regression test for the `data_to_bytes` deallocation bug:
    /// serializing used to rebuild the host `Vec<T>` as a `Vec<u8>` and free an
    /// align-4/8 allocation with an align-1 layout — undefined behaviour on
    /// every UQFF *write* of an f32/f64/i64/i32/u32/f16/bf16 tensor. The bytes
    /// happened to come out right, which is exactly why it survived: this test
    /// pins the observable contract so the sound implementation cannot silently
    /// change the format, while Miri covers the UB itself (a reduction of the
    /// old pattern reports "incorrect layout on deallocation: ... alignment 4,
    /// but gave ... alignment 1"; the replacement is clean).
    #[test]
    fn every_dtype_round_trips_through_serialize_deserialize() {
        let dev = Device::Cpu;
        // Shapes deliberately non-square and rank-3 so a transposed or
        // flattened round-trip cannot pass by coincidence.
        let shape = (2usize, 3usize, 4usize);
        let n = shape.0 * shape.1 * shape.2;
        let base: Vec<f32> = (0..n).map(|i| i as f32 - 7.0).collect();
        let src = Tensor::from_vec(base, shape, &dev).unwrap();

        for dtype in [
            DType::U8,
            DType::U32,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let t = src.to_dtype(dtype).unwrap();

            let mut buf = Vec::new();
            serialize_tensor(&mut buf, &t).unwrap();

            let mut cursor = std::io::Cursor::new(&buf);
            let back = deserialize_tensor(&mut cursor, &dev).unwrap();

            assert_eq!(back.dtype(), dtype, "dtype changed for {dtype:?}");
            assert_eq!(back.dims(), t.dims(), "shape changed for {dtype:?}");
            assert_eq!(
                cursor.position() as usize,
                buf.len(),
                "{dtype:?}: reader did not consume exactly the bytes written -- \
                 the declared data length disagrees with the payload"
            );

            let lhs = t.flatten_all().unwrap().to_dtype(DType::F64).unwrap();
            let rhs = back.flatten_all().unwrap().to_dtype(DType::F64).unwrap();
            let diff = (lhs - rhs)
                .unwrap()
                .abs()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
            assert_eq!(diff, 0.0, "values changed for {dtype:?}");
        }
    }

    /// Pin the on-disk header layout. UQFF is a shipped byte format, so a change
    /// here is a compatibility break and must be a deliberate one.
    #[test]
    fn serialized_header_layout_is_stable() {
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], (2usize, 3usize), &dev).unwrap();
        let mut buf = Vec::new();
        serialize_tensor(&mut buf, &t).unwrap();

        // data_len : u32 LE == 6 bytes of payload (6 x u8)
        assert_eq!(&buf[0..4], &6u32.to_le_bytes(), "data_len field moved");
        // dtype : u32 LE, then rank : u32 LE == 2
        assert_eq!(&buf[8..12], &2u32.to_le_bytes(), "rank field moved");
        // dims : u32 LE each
        assert_eq!(&buf[12..16], &2u32.to_le_bytes(), "dim0 moved");
        assert_eq!(&buf[16..20], &3u32.to_le_bytes(), "dim1 moved");
        // payload follows the header, unpadded
        assert_eq!(&buf[20..], &[1u8, 2, 3, 4, 5, 6], "payload moved or padded");
        assert_eq!(buf.len(), 26, "total encoded size changed");
    }

    /// `as_byte_slice` must expose the source's exact bytes, not a copy of the
    /// wrong length. Guards the replacement for the removed unsafe helper.
    #[test]
    fn as_byte_slice_exposes_the_whole_source() {
        let vs: Vec<u32> = vec![0x0403_0201, 0x0807_0605];
        let bytes = as_byte_slice(vs.as_slice());
        assert_eq!(bytes.len(), 8, "byte length must be len * size_of::<T>()");
        assert_eq!(bytes, &[1u8, 2, 3, 4, 5, 6, 7, 8], "byte content differs");
        // The source is still owned and usable -- i.e. we borrowed, not moved.
        assert_eq!(vs.len(), 2);
    }
}
