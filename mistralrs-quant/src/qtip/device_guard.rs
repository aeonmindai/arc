//! Device-ordinal validation shared by the QTIP CUDA kernel wrappers.
//!
//! Every wrapper in [`super::cuda_ops`] takes its launch stream from ONE tensor
//! (usually `blocks`) and then launches a kernel that dereferences the raw
//! device pointers of *all* the operands. Checking only that the other operands
//! are `Device::Cuda(_)` is not enough: under a multi-GPU map, activations on
//! `cuda:0` and expert weights on `cuda:1` pass a device-KIND check, the kernel
//! launches on `cuda:1`'s stream, and it reads a pointer that is not mapped in
//! that context. That is silent corruption (or an async fault attributed to an
//! unrelated launch later), not a clean error.
//!
//! The predicate below therefore compares device ORDINALS. It is deliberately a
//! pure function over [`candle_core::DeviceLocation`] so it can be unit-tested
//! on a host with no CUDA device and no CUDA toolchain.
//!
//! Only the CUDA wrappers call the helpers here, but the module itself is not
//! feature-gated so the pure predicate stays testable in a CPU-only build.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use candle_core::{Device, DeviceLocation, Result};

/// Render a [`DeviceLocation`] the way the CUDA driver / `CUDA_VISIBLE_DEVICES`
/// name it, so an error message can be matched against `nvidia-smi` output.
pub(crate) fn describe_location(loc: DeviceLocation) -> String {
    match loc {
        DeviceLocation::Cpu => "cpu".to_string(),
        DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
    }
}

/// Pure core of the guard: are `a` and `b` the SAME CUDA device?
///
/// Returns the shared ordinal on success, or the exact message the wrapper
/// should bail with. Being pure (and returning the message rather than a
/// `candle_core::Error`) is what makes this testable without a GPU.
///
/// `op` is the kernel-wrapper prefix already used by the surrounding bails
/// (e.g. `"QTIP gather gemv CUDA"`); `a_name` / `b_name` are the operand names
/// as they appear in the wrapper signature.
pub(crate) fn same_cuda_device_message(
    op: &str,
    a_name: &str,
    a: DeviceLocation,
    b_name: &str,
    b: DeviceLocation,
) -> std::result::Result<usize, String> {
    match (a, b) {
        (DeviceLocation::Cuda { gpu_id: lhs }, DeviceLocation::Cuda { gpu_id: rhs }) => {
            if lhs == rhs {
                Ok(lhs)
            } else {
                Err(format!(
                    "{op}: {a_name} is on cuda:{lhs} but {b_name} is on cuda:{rhs}; \
                     every operand of one kernel launch must live on the same device"
                ))
            }
        }
        (a_loc, DeviceLocation::Cuda { .. }) => Err(format!(
            "{op}: {a_name} must live on CUDA, got {}",
            describe_location(a_loc)
        )),
        (_, b_loc) => Err(format!(
            "{op}: {b_name} must live on CUDA, got {}",
            describe_location(b_loc)
        )),
    }
}

/// Wrapper-facing guard: bail unless `a` and `b` are on the same CUDA ordinal.
///
/// Pass the operand whose stream is NOT being used as `a` and the operand the
/// stream was taken from as `b`, so the message reads in the order a reader
/// expects (`x_rotated is on cuda:0 but blocks is on cuda:1`).
pub(crate) fn ensure_same_cuda_device(
    op: &str,
    a_name: &str,
    a: &Device,
    b_name: &str,
    b: &Device,
) -> Result<()> {
    match same_cuda_device_message(op, a_name, a.location(), b_name, b.location()) {
        Ok(_) => Ok(()),
        Err(msg) => Err(candle_core::Error::Msg(msg).bt()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_cuda_ordinal_is_accepted() {
        let got = same_cuda_device_message(
            "QTIP gather gemv CUDA",
            "x_rotated",
            DeviceLocation::Cuda { gpu_id: 3 },
            "blocks",
            DeviceLocation::Cuda { gpu_id: 3 },
        );
        assert_eq!(got, Ok(3));
    }

    #[test]
    fn different_cuda_ordinal_is_rejected_and_names_both() {
        let err = same_cuda_device_message(
            "QTIP gather gemv CUDA",
            "x_rotated",
            DeviceLocation::Cuda { gpu_id: 0 },
            "blocks",
            DeviceLocation::Cuda { gpu_id: 1 },
        )
        .expect_err("cuda:0 vs cuda:1 must not be accepted");
        assert!(
            err.contains("x_rotated is on cuda:0"),
            "message must name the left ordinal: {err}"
        );
        assert!(
            err.contains("blocks is on cuda:1"),
            "message must name the right ordinal: {err}"
        );
        assert!(
            err.starts_with("QTIP gather gemv CUDA:"),
            "message must carry the op prefix: {err}"
        );
    }

    /// The bug this guard exists for: a KIND check (`matches!(.., Cuda(_))`)
    /// passes for this pair, so the guard must not be reducible to one.
    #[test]
    fn kind_check_would_pass_where_ordinal_check_fails() {
        let a = DeviceLocation::Cuda { gpu_id: 0 };
        let b = DeviceLocation::Cuda { gpu_id: 7 };
        assert!(matches!(a, DeviceLocation::Cuda { .. }));
        assert!(matches!(b, DeviceLocation::Cuda { .. }));
        assert!(same_cuda_device_message("op", "a", a, "b", b).is_err());
    }

    #[test]
    fn non_cuda_left_operand_keeps_must_live_on_cuda_error() {
        let err = same_cuda_device_message(
            "QTIP fused gemv CUDA",
            "x_rotated",
            DeviceLocation::Cpu,
            "blocks",
            DeviceLocation::Cuda { gpu_id: 0 },
        )
        .expect_err("a cpu operand must be rejected");
        assert_eq!(
            err,
            "QTIP fused gemv CUDA: x_rotated must live on CUDA, got cpu"
        );
    }

    #[test]
    fn non_cuda_right_operand_keeps_must_live_on_cuda_error() {
        let err = same_cuda_device_message(
            "QTIP fused gemv CUDA",
            "x_rotated",
            DeviceLocation::Cuda { gpu_id: 0 },
            "blocks",
            DeviceLocation::Metal { gpu_id: 0 },
        )
        .expect_err("a metal operand must be rejected");
        assert_eq!(
            err,
            "QTIP fused gemv CUDA: blocks must live on CUDA, got metal:0"
        );
    }

    #[test]
    fn two_non_cuda_operands_are_rejected() {
        assert!(
            same_cuda_device_message("op", "a", DeviceLocation::Cpu, "b", DeviceLocation::Cpu)
                .is_err()
        );
    }

    #[test]
    fn describe_location_renders_ordinals() {
        assert_eq!(describe_location(DeviceLocation::Cpu), "cpu");
        assert_eq!(
            describe_location(DeviceLocation::Cuda { gpu_id: 2 }),
            "cuda:2"
        );
        assert_eq!(
            describe_location(DeviceLocation::Metal { gpu_id: 1 }),
            "metal:1"
        );
    }

    #[test]
    fn ensure_same_cuda_device_accepts_matching_cpu_free_path() {
        // Two CPU devices are NOT a valid kernel launch pair: the guard must
        // reject them even though `Device::same_device` would call them equal.
        let cpu = Device::Cpu;
        assert!(ensure_same_cuda_device("op", "a", &cpu, "b", &cpu).is_err());
    }
}
