use crate::math::{Matrix, Real, Vector, DIM};

#[cfg(not(feature = "std"))]
use na::ComplexField;

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct DruckerPragerPlasticity {
    pub h0: Real,
    pub h1: Real,
    pub h2: Real,
    pub h3: Real,
    pub lambda: Real,
    pub mu: Real,
    pub only_active_when_failed: bool,
    pub volume_correction: Real,
}

impl DruckerPragerPlasticity {
    pub fn new(young_modulus: Real, poisson_ration: Real) -> Self {
        let (lambda, mu) = crate::utils::lame_lambda_mu(young_modulus, poisson_ration);
        DruckerPragerPlasticity {
            h0: (35.0 as Real).to_radians(),
            h1: (9.0 as Real).to_radians(),
            h2: 0.2,
            h3: (10.0 as Real).to_radians(),
            lambda,
            mu,
            only_active_when_failed: false,
            volume_correction: 1.,
        }
    }

    pub fn project_deformation_gradient(
        &self,
        singular_values: Vector<Real>,
        log_vol_gain: Real,
        alpha: Real,
    ) -> Option<(Vector<Real>, Real)> {
        let d = DIM as Real;
        let strain = singular_values.map(|e| e.ln()) + Vector::repeat(log_vol_gain / d);
        let strain_trace = strain.sum();
        let deviatoric_strain = strain - Vector::repeat(strain_trace / d);

        if deviatoric_strain == Vector::zeros() || strain_trace > 0.0 {
            return Some((Vector::repeat(1.0), strain.norm()));
        }

        let gamma = deviatoric_strain.norm()
            + (d * self.lambda + 2.0 * self.mu) / (2.0 * self.mu) * strain_trace * alpha;
        if gamma <= 0.0 {
            return None;
        }

        let h = strain - gamma * deviatoric_strain.normalize();
        Some((h.map(|e| e.exp()), gamma))
    }

    fn alpha(&self, q: Real) -> Real {
        let angle = self.h0 + (self.h1 * q - self.h3) * (-self.h2 * q).exp();
        let s_angle = angle.sin();

        (2.0 as Real / 3.0).sqrt() * (2.0 * s_angle) / (3.0 - s_angle)
    }

    pub fn update_particle(
        &self,
        particle_phase: Real,
        particle_deformation_gradient: &mut Matrix<Real>,
        particle_plastic_deformation_gradient_det: &mut Real,
        particle_plastic_hardening: &mut Real,
        particle_log_vol_gain: &mut Real,
    ) {
        if self.only_active_when_failed && particle_phase != 0.0 {
            return;
        }

        let mut svd = particle_deformation_gradient.svd_unordered(true, true);

        let alpha = self.alpha(*particle_plastic_hardening);

        if let Some((new_singular_values, dq)) =
            self.project_deformation_gradient(svd.singular_values, *particle_log_vol_gain, alpha)
        {
            let prev_det = svd.singular_values.product();

            let new_det = new_singular_values.product();
            let diff = new_singular_values.product() - prev_det;
            let new_det = if diff > 0. {
                new_det
            } else {
                prev_det + diff * self.volume_correction
            };

            *particle_plastic_deformation_gradient_det *= prev_det / new_det;
            *particle_log_vol_gain += prev_det.ln() - new_det.ln();

            svd.singular_values = new_singular_values;
            *particle_plastic_hardening += dq; // TODO: Is it OK to store this here?
            *particle_deformation_gradient = svd.recompose().unwrap();
        } // Else, do nothing because we are inside of the yield surface.
    }
}
