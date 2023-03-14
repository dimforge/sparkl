#![no_std]

pub extern crate nalgebra as na;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

pub mod prelude {
    pub use crate::dynamics::models::*;
    pub use crate::dynamics::solver::*;
    pub use crate::dynamics::*;
    pub use crate::math::*;
}

pub mod math {
    // pub use super::parry::math::*; // FIXME: make parry no-std friendly and import its math module instead.
    pub use super::parry_math::math::*;
    pub type Kernel = crate::dynamics::solver::QuadraticKernel;

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

    impl core::ops::Add<DecomposedTensor> for DecomposedTensor {
        type Output = DecomposedTensor;

        #[inline]
        fn add(self, rhs: DecomposedTensor) -> DecomposedTensor {
            DecomposedTensor {
                deviatoric_part: self.deviatoric_part + rhs.deviatoric_part,
                spherical_part: self.spherical_part + rhs.spherical_part,
            }
        }
    }

    impl core::ops::AddAssign<DecomposedTensor> for DecomposedTensor {
        #[inline]
        fn add_assign(&mut self, rhs: DecomposedTensor) {
            self.deviatoric_part += rhs.deviatoric_part;
            self.spherical_part += rhs.spherical_part;
        }
    }

    impl core::ops::Mul<Vector<Real>> for DecomposedTensor {
        type Output = Vector<Real>;

        #[inline]
        fn mul(self, rhs: Vector<Real>) -> Self::Output {
            self.deviatoric_part * rhs + self.spherical_part * rhs
        }
    }
}

mod parry_math {

    mod real {
        /// The scalar type used throughout this crate.
        #[cfg(feature = "f64")]
        pub type Real = f64;

        /// The scalar type used throughout this crate.
        #[cfg(feature = "f32")]
        pub type Real = f32;
    }

    /// Compilation flags dependent aliases for mathematical types.
    #[cfg(feature = "dim3")]
    pub mod math {
        pub use super::real::*;
        use na::{
            Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, UnitVector3, Vector3,
            Vector6, U3, U6,
        };

        /// The default tolerance used for geometric operations.
        pub const DEFAULT_EPSILON: Real = Real::EPSILON;

        /// The dimension of the space.
        pub const DIM: usize = 3;

        /// The dimension of the ambient space.
        pub type Dim = U3;

        /// The dimension of a spatial vector.
        pub type SpatialDim = U6;

        /// The dimension of the rotations.
        pub type AngDim = U3;

        /// The point type.
        pub type Point<N> = Point3<N>;

        /// The angular vector type.
        pub type AngVector<N> = Vector3<N>;

        /// The vector type.
        pub type Vector<N> = Vector3<N>;

        /// The unit vector type.
        pub type UnitVector<N> = UnitVector3<N>;

        /// The matrix type.
        pub type Matrix<N> = Matrix3<N>;

        /// The vector type with dimension `SpatialDim × 1`.
        pub type SpatialVector<N> = Vector6<N>;

        /// The orientation type.
        pub type Orientation<N> = Vector3<N>;

        /// The transformation matrix type.
        pub type Isometry<N> = Isometry3<N>;

        /// The rotation matrix type.
        pub type Rotation<N> = UnitQuaternion<N>;

        /// The translation type.
        pub type Translation<N> = Translation3<N>;

        /// The principal angular inertia of a rigid body.
        pub type PrincipalAngularInertia<N> = Vector3<N>;

        /// A matrix that represent the cross product with a given vector.
        pub type CrossMatrix<N> = Matrix3<N>;

        /// A vector with a dimension equal to the maximum number of degrees of freedom of a rigid body.
        pub type SpacialVector<N> = Vector6<N>;
    }

    /// Compilation flags dependent aliases for mathematical types.
    #[cfg(feature = "dim2")]
    pub mod math {
        pub use super::real::*;
        use na::{
            Isometry2, Matrix2, Point2, Translation2, UnitComplex, UnitVector2, Vector1, Vector2,
            Vector3, U2,
        };

        /// The default tolerance used for geometric operations.
        pub const DEFAULT_EPSILON: Real = Real::EPSILON;

        /// The dimension of the space.
        pub const DIM: usize = 2;

        /// The dimension of the ambient space.
        pub type Dim = U2;

        /// The point type.
        pub type Point<N> = Point2<N>;

        /// The angular vector type.
        pub type AngVector<N> = N;

        /// The vector type.
        pub type Vector<N> = Vector2<N>;

        /// The unit vector type.
        pub type UnitVector<N> = UnitVector2<N>;

        /// The matrix type.
        pub type Matrix<N> = Matrix2<N>;

        /// The orientation type.
        pub type Orientation<N> = Vector1<N>;

        /// The transformation matrix type.
        pub type Isometry<N> = Isometry2<N>;

        /// The rotation matrix type.
        pub type Rotation<N> = UnitComplex<N>;

        /// The translation type.
        pub type Translation<N> = Translation2<N>;

        /// The angular inertia of a rigid body.
        pub type AngularInertia<N> = N;

        /// The principal angular inertia of a rigid body.
        pub type PrincipalAngularInertia<N> = N;

        /// A matrix that represent the cross product with a given vector.
        pub type CrossMatrix<N> = Vector2<N>;

        /// A vector with a dimension equal to the maximum number of degrees of freedom of a rigid body.
        pub type SpacialVector<N> = Vector3<N>;
    }
}

pub mod dynamics;
pub mod rigid_particles;
pub mod utils;
