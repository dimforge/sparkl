use crate::math::{Matrix, Real, DIM};
use core::cmp::Ordering;

#[cfg(not(feature = "std"))]
use na::ComplexField;

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RankinePlasticity {
    mu: Real,
    lambda: Real,
    tensile_strength: Real,
    softening_rate: Real,
}

impl RankinePlasticity {
    pub fn new(
        young_modulus: Real,
        poisson_ratio: Real,
        tensile_strength: Real,
        softening_rate: Real,
    ) -> Self {
        let (lambda, mu) = crate::utils::lame_lambda_mu(young_modulus, poisson_ratio);

        Self {
            lambda,
            mu,
            tensile_strength,
            softening_rate,
        }
    }

    pub fn update_particle(
        &self,
        particle_deformation_gradient: &mut Matrix<Real>,
        particle_plastic_hardening: &mut Real,
    ) {
        let lambda = self.lambda;
        let mu = self.mu;

        let mut svd = particle_deformation_gradient.svd_unordered(true, true);
        let mut eigv = svd.singular_values.map(|s| s.ln()); // Hencky strain eigenvalues.

        let prev_eigv = eigv;
        let mut idx = [0usize, 1, DIM - 1]; // We use DIM - 1 so that it works in 2D too.
        idx.sort_unstable_by(|a, b| eigv[*a].partial_cmp(&eigv[*b]).unwrap_or(Ordering::Equal));
        let [e3, e2, e1] = idx;
        // TODO: The -1 is to account for the initial hardening value equal to 1.0.
        //       Having to do this sounds error-prone (what if the user modifies that initial value?
        //       Should we let the user modify that initial value?
        let soft_tensile_strength = self.tensile_strength - (*particle_plastic_hardening - 1.0);

        if lambda * eigv.sum() + 2.0 * mu * eigv[e1] <= soft_tensile_strength {
            // No plastic deformation.
            return;
        } else if (2.0 * mu + lambda) * eigv[e2] + lambda * (eigv.sum() - eigv[e1])
            <= soft_tensile_strength
        {
            let new_eigv =
                (soft_tensile_strength - lambda * (eigv.sum() - eigv[e1])) / (2.0 * mu + lambda);
            eigv[e1] = new_eigv;
        } else if DIM == 3 && (2.0 * mu + 3.0 * lambda) * eigv[e3] <= soft_tensile_strength {
            let new_eigv = (soft_tensile_strength - lambda * (eigv.sum() - eigv[e1] - eigv[e2]))
                / (2.0 * mu + 2.0 * lambda);
            eigv[e1] = new_eigv;
            eigv[e2] = new_eigv;
        } else {
            let new_eigv = soft_tensile_strength / (2.0 * self.mu + 3.0 * self.lambda);
            eigv.fill(new_eigv);
        }

        *particle_plastic_hardening += self.softening_rate * (prev_eigv - eigv).norm();
        *particle_plastic_hardening = particle_plastic_hardening.min(self.tensile_strength);
        svd.singular_values = eigv.map(|e| e.exp());
        *particle_deformation_gradient = svd.recompose().unwrap();
    }
}
