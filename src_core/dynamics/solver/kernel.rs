use crate::math::{Point, Real, Vector, DIM};
use na::vector;
#[cfg(target_os = "cuda")]
use na::ComplexField;

pub struct QuadraticKernel;

impl QuadraticKernel {
    #[inline(always)]
    pub fn inv_d(cell_width: Real) -> Real {
        4.0 / (cell_width * cell_width)
    }

    #[inline(always)]
    pub fn eval_all(x: Real) -> [Real; 3] {
        [
            0.5 * (1.5 - x).powi(2),
            0.75 - (x - 1.0).powi(2),
            0.5 * (x - 0.5).powi(2),
        ]
    }

    // #[inline(always)]
    // pub fn eval_all_simd(x: SimdReal) -> [SimdReal; 3] {
    //     let a = SimdReal::splat(1.5) - x;
    //     let b = x - SimdReal::splat(1.0);
    //     let c = x - SimdReal::splat(0.5);
    //
    //     [
    //         SimdReal::splat(0.5) * a * a,
    //         SimdReal::splat(0.75) * b * b,
    //         SimdReal::splat(0.5) * c * c,
    //     ]
    // }

    #[inline(always)]
    pub fn eval(x: Real) -> Real {
        let x_abs = x.abs();

        if x_abs < 0.5 {
            3.0 / 4.0 - x_abs.powi(2)
        } else if x_abs < 3.0 / 2.0 {
            0.5 * (3.0 / 2.0 - x_abs).powi(2)
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn eval_derivative(x: Real) -> Real {
        let x_abs = x.abs();

        if x_abs < 0.5 {
            -2.0 * x.signum() * x_abs
        } else if x_abs < 3.0 / 2.0 {
            -x.signum() * (3.0 / 2.0 - x_abs)
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn precompute_weights(
        ref_elt_pos_minus_particle_pos: Vector<Real>,
        h: Real,
    ) -> [[Real; 3]; DIM] {
        [
            Self::eval_all(-ref_elt_pos_minus_particle_pos.x / h),
            Self::eval_all(-ref_elt_pos_minus_particle_pos.y / h),
            #[cfg(feature = "dim3")]
            Self::eval_all(-ref_elt_pos_minus_particle_pos.z / h),
        ]
    }

    // #[inline(always)]
    // pub fn precompute_weights_simd(
    //     ref_elt_pos_minus_particle_pos: Vector<SimdReal>,
    //     h: SimdReal,
    // ) -> [[SimdReal; 3]; DIM] {
    //     [
    //         Self::eval_all_simd(-ref_elt_pos_minus_particle_pos.x / h),
    //         Self::eval_all_simd(-ref_elt_pos_minus_particle_pos.y / h),
    //         #[cfg(feature = "dim3")]
    //         Self::eval_all_simd(-ref_elt_pos_minus_particle_pos.z / h),
    //     ]
    // }

    #[inline(always)]
    pub fn stencil_with_dir(elt_pos_minus_particle_pos: Vector<Real>, h: Real) -> Real {
        let dpt = -elt_pos_minus_particle_pos / h;
        #[cfg(feature = "dim2")]
        return Self::eval(dpt.x) * Self::eval(dpt.y);
        #[cfg(feature = "dim3")]
        return Self::eval(dpt.x) * Self::eval(dpt.y) * Self::eval(dpt.z);
    }

    #[inline(always)]
    pub fn stencil(elt_pos: Point<Real>, particle_pos: Point<Real>, h: Real) -> Real {
        Self::stencil_with_dir(elt_pos - particle_pos, h)
    }

    #[inline(always)]
    pub fn stencil_gradient_with_dir(
        elt_pos_minus_particle_pos: Vector<Real>,
        h: Real,
    ) -> Vector<Real> {
        let inv_h = 1.0 / h;
        let dpt = -elt_pos_minus_particle_pos * inv_h;
        let val_x = Self::eval(dpt.x);
        let val_y = Self::eval(dpt.y);

        #[cfg(feature = "dim3")]
        let val_z = Self::eval(dpt.z);

        #[cfg(feature = "dim2")]
        return vector![
            inv_h * Self::eval_derivative(dpt.x) * val_y,
            inv_h * val_x * Self::eval_derivative(dpt.y)
        ];
        #[cfg(feature = "dim3")]
        return vector![
            inv_h * Self::eval_derivative(dpt.x) * val_y * val_z,
            inv_h * val_x * Self::eval_derivative(dpt.y) * val_z,
            inv_h * val_x * val_y * Self::eval_derivative(dpt.z)
        ];
    }

    #[inline(always)]
    pub fn stencil_gradient(
        elt_pos: Point<Real>,
        particle_pos: Point<Real>,
        h: Real,
    ) -> Vector<Real> {
        Self::stencil_gradient_with_dir(elt_pos - particle_pos, h)
    }
}
