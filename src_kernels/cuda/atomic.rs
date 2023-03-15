#![allow(unreachable_code)]

use sparkl_core::math::Vector;
use sparkl_core::na::Scalar;

// TODO: this is needed untile Rust-GPU supports atomics.
pub trait AtomicAdd {
    unsafe fn shared_red_add(&mut self, rhs: Self);
    unsafe fn global_red_add(&mut self, rhs: Self);
    unsafe fn global_atomic_add(&mut self, rhs: Self) -> Self;
}

pub trait AtomicInt {
    unsafe fn global_red_or(&mut self, rhs: Self);
    unsafe fn global_red_min(&mut self, rhs: Self);
    unsafe fn global_atomic_exch(&mut self, val: Self) -> Self;
    unsafe fn global_atomic_cas(&mut self, cmp: Self, val: Self) -> Self;
    unsafe fn shared_atomic_exch_acq(&mut self, val: Self) -> Self;
    unsafe fn shared_atomic_exch_rel(&mut self, val: Self) -> Self;
    unsafe fn global_atomic_dec(&mut self) -> Self;
}

impl AtomicAdd for u32 {
    unsafe fn shared_red_add(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut shared_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            red.shared.add.u32 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_red_add(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.add.u32 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_atomic_add(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.add.u32 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs,
            old = out(reg32) old,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }
}

impl AtomicInt for u32 {
    unsafe fn global_red_or(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.or.b32 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_red_min(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.min.u32 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_atomic_exch(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.exch.b32 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs,
            old = out(reg32) old,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn global_atomic_cas(&mut self, _cmp: Self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
                atom.global.cas.b32 {old}, [{gbl_ptr}], {cmp}, {rhs};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            cmp = in(reg32) _cmp,
            rhs = in(reg32) _rhs,
            old = out(reg32) old,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn shared_atomic_exch_acq(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut shared_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            atom.acquire.shared.exch.b32 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs,
            old = out(reg32) old,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn shared_atomic_exch_rel(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut shared_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            atom.release.shared.exch.b32 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg32) _rhs,
            old = out(reg32) old,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn global_atomic_dec(&mut self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let max = u32::MAX;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.dec.u32 {old}, [{gbl_ptr}], {max};",
                gbl_ptr = out(reg64) global_integer_addr,
                org_ptr = in(reg64) integer_addr,
                old = out(reg32) old,
                max = in(reg32) max,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }
}

impl AtomicAdd for u64 {
    unsafe fn shared_red_add(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut shared_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            red.shared.add.u64 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_red_add(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.add.u64 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_atomic_add(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.add.u64 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs,
            old = out(reg64) old
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }
}

impl AtomicInt for u64 {
    unsafe fn global_red_or(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.or.b64 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_red_min(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.min.u64 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_atomic_exch(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.exch.b64 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs,
            old = out(reg64) old
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn global_atomic_cas(&mut self, _cmp: Self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.cas.b64 {old}, [{gbl_ptr}], {cmp}, {rhs};",
            gbl_ptr = out(reg64) global_integer_addr,
            org_ptr = in(reg64) integer_addr,
            cmp = in(reg64) _cmp,
            rhs = in(reg64) _rhs,
            old = out(reg64) old
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn shared_atomic_exch_acq(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut shared_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            atom.acquire.shared.exch.b64 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs,
            old = out(reg64) old
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn shared_atomic_exch_rel(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let integer_addr = self as *mut _;
            let mut shared_integer_addr: *mut u64 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            atom.release.shared.exch.b64 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_integer_addr,
            org_ptr = in(reg64) integer_addr,
            number = in(reg64) _rhs,
            old = out(reg64) old
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }

    unsafe fn global_atomic_dec(&mut self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0;
            let max = u64::MAX;
            let integer_addr = self as *mut _;
            let mut global_integer_addr: *mut u32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.dec.u64 {old}, [{gbl_ptr}], {max};",
                gbl_ptr = out(reg64) global_integer_addr,
                org_ptr = in(reg64) integer_addr,
                old = out(reg64) old,
                max = in(reg64) max,
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        return unimplemented!();
    }
}

impl AtomicAdd for f32 {
    unsafe fn shared_red_add(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let float_addr = self as *mut _;
            let mut shared_float_addr: *mut f32 = core::ptr::null_mut();

            asm!(
            "cvta.to.shared.u64 {gbl_ptr}, {org_ptr};\
            red.shared.add.f32 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) shared_float_addr,
            org_ptr = in(reg64) float_addr,
            number = in(reg32) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_red_add(&mut self, _rhs: Self) {
        #[cfg(target_os = "cuda")]
        {
            let float_addr = self as *mut _;
            let mut global_float_addr: *mut f32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            red.global.add.f32 [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_float_addr,
            org_ptr = in(reg64) float_addr,
            number = in(reg32) _rhs
            );
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }

    unsafe fn global_atomic_add(&mut self, _rhs: Self) -> Self {
        #[cfg(target_os = "cuda")]
        {
            let mut old = 0.0;
            let float_addr = self as *mut _;
            let mut global_float_addr: *mut f32 = core::ptr::null_mut();

            asm!(
            "cvta.to.global.u64 {gbl_ptr}, {org_ptr};\
            atom.global.add.f32 {old}, [{gbl_ptr}], {number};",
            gbl_ptr = out(reg64) global_float_addr,
            org_ptr = in(reg64) float_addr,
            number = in(reg32) _rhs,
            old = out(reg32) old
            );

            old
        }

        #[cfg(not(target_os = "cuda"))]
        unimplemented!();
    }
}

impl<T: Scalar + AtomicAdd> AtomicAdd for Vector<T> {
    unsafe fn shared_red_add(&mut self, rhs: Self) {
        self.zip_apply(&rhs, |a, b| a.shared_red_add(b))
    }

    unsafe fn global_red_add(&mut self, rhs: Self) {
        self.zip_apply(&rhs, |a, b| a.global_red_add(b))
    }

    unsafe fn global_atomic_add(&mut self, _rhs: Self) -> Self {
        unimplemented!()
    }
}
