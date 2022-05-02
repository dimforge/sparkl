#[cfg(all(feature = "dim2", feature = "f32"))]
pub extern crate parry2d as parry;
#[cfg(all(feature = "dim2", feature = "f64"))]
pub extern crate parry2d_f64 as parry;
#[cfg(all(feature = "dim3", feature = "f32"))]
pub extern crate parry3d as parry;
#[cfg(all(feature = "dim3", feature = "f64"))]
pub extern crate parry3d_f64 as parry;

#[cfg(all(feature = "dim2", feature = "f32"))]
pub extern crate rapier2d as rapier;
#[cfg(all(feature = "dim2", feature = "f64"))]
pub extern crate rapier2d_f64 as rapier;
#[cfg(all(feature = "dim3", feature = "f32"))]
pub extern crate rapier3d as rapier;
#[cfg(all(feature = "dim3", feature = "f64"))]
pub extern crate rapier3d_f64 as rapier;

#[cfg(all(feature = "dim2", feature = "rapier-testbed"))]
extern crate rapier_testbed2d as rapier_testbed;
#[cfg(all(feature = "dim3", feature = "rapier-testbed"))]
extern crate rapier_testbed3d as rapier_testbed;

#[cfg(feature = "dim2")]
pub extern crate sparkl2d_core;
#[cfg(all(feature = "dim2", feature = "cuda"))]
pub extern crate sparkl2d_kernels as kernels;

#[cfg(feature = "cuda")]
pub extern crate cust;
#[cfg(feature = "dim3")]
pub extern crate sparkl3d_core;
#[cfg(all(feature = "dim3", feature = "cuda"))]
pub extern crate sparkl3d_kernels as kernels;

pub extern crate nalgebra as na;

#[macro_use]
extern crate log;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

#[cfg(feature = "dim2")]
pub use sparkl2d_core as core;
#[cfg(feature = "dim3")]
pub use sparkl3d_core as core;

pub mod prelude {
    pub use crate::dynamics::models::*;
    pub use crate::dynamics::solver::*;
    pub use crate::dynamics::*;
    pub use crate::geometry::*;
    pub use crate::math::*;
}

pub mod math {
    pub use super::parry::math::*;
    pub type Kernel = crate::core::dynamics::solver::QuadraticKernel;

    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct DecomposedTensor {
        pub deviatoric_part: Matrix<Real>,
        pub spherical_part: Real,
    }

    impl DecomposedTensor {
        pub fn decompose(tensor: &Matrix<Real>) -> Self {
            let spherical_part = tensor.trace() / (DIM as Real);
            let mut deviatoric_part = *tensor;

            for i in 0..DIM {
                deviatoric_part[(i, i)] -= spherical_part;
            }

            Self {
                deviatoric_part,
                spherical_part,
            }
        }

        pub fn zero() -> Self {
            Self {
                deviatoric_part: Matrix::zeros(),
                spherical_part: 0.0,
            }
        }

        pub fn recompose(&self) -> Matrix<Real> {
            self.deviatoric_part + Matrix::identity() * self.spherical_part
        }
    }

    impl std::ops::Add<DecomposedTensor> for DecomposedTensor {
        type Output = DecomposedTensor;

        #[inline]
        fn add(self, rhs: DecomposedTensor) -> DecomposedTensor {
            DecomposedTensor {
                deviatoric_part: self.deviatoric_part + rhs.deviatoric_part,
                spherical_part: self.spherical_part + rhs.spherical_part,
            }
        }
    }

    impl std::ops::AddAssign<DecomposedTensor> for DecomposedTensor {
        #[inline]
        fn add_assign(&mut self, rhs: DecomposedTensor) {
            self.deviatoric_part += rhs.deviatoric_part;
            self.spherical_part += rhs.spherical_part;
        }
    }

    impl std::ops::Mul<Vector<Real>> for DecomposedTensor {
        type Output = Vector<Real>;

        #[inline]
        fn mul(self, rhs: Vector<Real>) -> Self::Output {
            self.deviatoric_part * rhs + self.spherical_part * rhs
        }
    }
}

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dynamics;
pub mod geometry;
pub mod pipelines;
pub mod third_party;
pub mod utils;
