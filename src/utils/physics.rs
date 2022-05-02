use crate::math::{DecomposedTensor, Matrix, Real, DIM};
use na::SMatrix;

pub fn inv_exact(e: Real) -> Real {
    // We don't want to use any threshold here.
    if e == 0.0 {
        0.0
    } else {
        1.0 / e
    }
}

/// Computes the Lamé parameters (lambda, mu) from the young modulus and poisson ratio.
pub fn lame_lambda_mu(young_modulus: Real, poisson_ratio: Real) -> (Real, Real) {
    (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        shear_modulus(young_modulus, poisson_ratio),
    )
}

pub fn shear_modulus(young_modulus: Real, poisson_ratio: Real) -> Real {
    young_modulus / (2.0 * (1.0 + poisson_ratio))
}

pub fn bulk_modulus(young_modulus: Real, poisson_ratio: Real) -> Real {
    young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
}

pub fn shear_modulus_from_lame(_lambda: Real, mu: Real) -> Real {
    mu
}

pub fn bulk_modulus_from_lame(lambda: Real, mu: Real) -> Real {
    lambda + 2.0 * mu / 3.0
}

pub fn solve_quadratic(a: Real, b: Real, c: Real) -> (Real, Real) {
    let discr_sqr = (b * b - 4.0 * a * c).sqrt();
    ((-b + discr_sqr) / (2.0 * a), (-b - discr_sqr) / (2.0 * a))
}

pub fn min_componentwise_quadratic_solve<const R: usize, const C: usize>(
    a: &SMatrix<Real, R, C>,
    b: &SMatrix<Real, R, C>,
    c: Real,
    sol_half_range: (Real, Real),
) -> Real {
    a.zip_map(b, |a, b| {
        let (mut s0, mut s1) = solve_quadratic(a, b, c);
        if s0 <= sol_half_range.0 {
            s0 = Real::MAX;
        }
        if s1 <= sol_half_range.0 {
            s1 = Real::MAX;
        }

        s0.min(s1)
    })
    .min()
    .min(sol_half_range.1)
}

pub fn spin_tensor(velocity_gradient: &Matrix<Real>) -> Matrix<Real> {
    (velocity_gradient - velocity_gradient.transpose()) * 0.5
}

pub fn strain_rate(velocity_gradient: &Matrix<Real>) -> Matrix<Real> {
    (velocity_gradient + velocity_gradient.transpose()) * 0.5
}

pub fn deviatoric_part(tensor: &Matrix<Real>) -> Matrix<Real> {
    DecomposedTensor::decompose(tensor).deviatoric_part
}
pub fn spherical_part(tensor: &Matrix<Real>) -> Real {
    tensor.trace() / (DIM as Real)
}
