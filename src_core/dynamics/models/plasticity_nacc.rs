use crate::math::{Matrix, Real, Vector, DIM};
use na::vector;

#[cfg(not(feature = "std"))]
use na::ComplexField;

/// The Non-Associated-Cam-Clay plasticity model.
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct NaccPlasticity {
    mu: Real,
    kappa: Real,
    hardening_enabled: bool,
    hardening_factor: Real, // xi
    cohesion: Real,         // beta
    friction: Real,         // M
}

impl NaccPlasticity {
    pub fn new(
        young_modulus: Real,
        poisson_ratio: Real,
        cohesion: Real,
        hardening_enabled: bool,
        hardening_factor: Real,
        friction_angle: Real,
    ) -> Self {
        let sin_f = friction_angle.sin();
        let d = DIM as Real;
        Self {
            mu: crate::utils::shear_modulus(young_modulus, poisson_ratio),
            kappa: crate::utils::bulk_modulus(young_modulus, poisson_ratio),
            hardening_enabled,
            hardening_factor,
            cohesion,
            friction: (2.0 as Real / 3.0).sqrt() * 2.0 * sin_f / (3.0 - sin_f) * d
                / (2.0 / (6.0 - d)).sqrt(),
        }
    }

    pub fn with_m(
        young_modulus: Real,
        poisson_ratio: Real,
        cohesion: Real,
        hardening_enabled: bool,
        hardening_factor: Real,
        m: Real,
    ) -> Self {
        Self {
            mu: crate::utils::shear_modulus(young_modulus, poisson_ratio),
            kappa: crate::utils::bulk_modulus(young_modulus, poisson_ratio),
            hardening_enabled,
            hardening_factor,
            cohesion,
            friction: m,
        }
    }

    pub fn project_deformation_gradient(
        &self,
        deformation_gradient: Matrix<Real>,
        mut alpha: Real,
    ) -> (Matrix<Real>, Real) {
        let xi = self.hardening_factor;
        let beta = self.cohesion;
        let m = self.friction;
        let d = DIM as Real;

        let mut svd = deformation_gradient.svd_unordered(true, true);

        let square_eigv = svd.singular_values.component_mul(&svd.singular_values);
        let square_eigv_trace = square_eigv.sum();

        let p0 = self.kappa * (1.0e-5 + ((xi * (-alpha).max(0.0)).sinh()));
        #[cfg(feature = "dim2")]
        let j_e_tr = svd.singular_values[0] * svd.singular_values[1];
        #[cfg(feature = "dim3")]
        let j_e_tr = svd.singular_values[0] * svd.singular_values[1] * svd.singular_values[2];
        let s_tr =
            self.mu * j_e_tr.powf(-2.0 / d) * (square_eigv - Vector::repeat(square_eigv_trace / d));
        let psi_kappa = self.kappa / 2.0 * (j_e_tr - 1.0 / j_e_tr);
        let p_tr = -psi_kappa * j_e_tr;

        if p_tr > p0 {
            // Project to max tip of the yield surface.
            let j_e_n1 = (-2.0 * p0 / self.kappa + 1.0).sqrt();
            svd.singular_values.fill(j_e_n1.powf(1.0 / d));

            if self.hardening_enabled {
                alpha += (j_e_tr / j_e_n1).ln();
            }
            // info!("Return a: {} > {}", p_tr, p0);
            return (svd.recompose().unwrap(), alpha);
        }

        if p_tr < -beta * p0 {
            // Project to min tip of the yield surface.
            let j_e_n1 = (2.0 * beta * p0 / self.kappa + 1.0).sqrt();
            svd.singular_values.fill(j_e_n1.powf(1.0 / d));

            if self.hardening_enabled {
                alpha += (j_e_tr / j_e_n1).ln();
            }
            // info!("Retrun B");
            return (svd.recompose().unwrap(), alpha);
        }

        let y0 = (1.0 + 2.0 * beta) * ((6.0 - d) / 2.0);
        let y1 = m * m * (p_tr + beta * p0) * (p_tr - p0);
        let y = y0 * s_tr.norm_squared() + y1;

        if y < 1.0e-4 {
            // Inside the yield surface.
            return (deformation_gradient, alpha);
        }

        if self.hardening_enabled && p0 > 1.0e-4 && p_tr < p0 - 1.0e-4 && p_tr > -beta * p0 + 1.0e-4
        {
            // Hardening routine.
            let p_c = (1.0 - beta) * p0 / 2.0;
            let q_tr = ((6.0 - d) / 2.0).sqrt() * s_tr.norm();
            let direction = vector![p_c - p_tr, 0.0 - q_tr];
            let direction = direction.normalize();
            let c = m * m * (p_c + beta * p0) * (p_c - p0);
            let b = m * m * direction[0] * (2.0 * p_c - p0 + beta * p0);
            let a = m * m * direction[0] * direction[0]
                + (1.0 + 2.0 * beta) * direction[1] * direction[1];
            let discr = (b * b - 4.0 * a * c).sqrt();
            let l1 = (-b + discr) / (2.0 * a);
            let l2 = (-b - discr) / (2.0 * a);
            let p1 = p_c + l1 * direction[0];
            let p2 = p_c + l2 * direction[0];
            let p_x = if (p_tr - p_c) * (p1 - p_c) > 0.0 {
                p1
            } else {
                p2
            };
            let j_e_x = (-2.0 * p_x / self.kappa + 1.0).abs().sqrt();

            if j_e_x > 1.0e-4 {
                alpha += (j_e_tr / j_e_x).ln();
            }
        }

        // Yield surface projection.
        let b_e_n1 = (-y1 / y0).sqrt() * (j_e_tr.powf(2.0 / d) / self.mu) * s_tr.normalize()
            + Vector::repeat(square_eigv_trace / d);

        svd.singular_values = b_e_n1.map(|e| e.sqrt());
        // info!("Retrun D");
        return (svd.recompose().unwrap(), alpha);
    }

    pub fn update_particle(
        &self,
        particle_deformation_gradient: &mut Matrix<Real>,
        particle_nacc_alpha: &mut Real,
    ) {
        let (new_def, new_alpha) =
            self.project_deformation_gradient(*particle_deformation_gradient, *particle_nacc_alpha);
        *particle_deformation_gradient = new_def;
        *particle_nacc_alpha = new_alpha;
    }
}
