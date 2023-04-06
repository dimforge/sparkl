use crate::{cuda::AtomicInt, DevicePointer};
use core::marker::PhantomData;
use na::ComplexField;
use parry::{
    math::{AngVector, Isometry, Matrix, Point, Real, Vector},
    query::{PointProjection, PointQueryWithLocation},
    shape::{Cuboid, CudaHeightFieldPtr, CudaTriMeshPtr, Segment, SegmentPointLocation, Triangle},
    utils::CudaArrayPointer1,
};
use sparkl_core::dynamics::solver::BoundaryCondition;

// Todo: remove this after transitioning to CDF

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub enum GpuColliderShape {
    Cuboid(Cuboid),
    HeightField {
        heightfield: CudaHeightFieldPtr,
        bellow_heightfield_is_solid: bool,
        flip_interior: bool,
    },
    TriMesh {
        trimesh: CudaTriMeshPtr,
        flip_interior: bool,
    },
    Polyline {
        vertices: CudaArrayPointer1<Point<Real>>,
        flip_interior: bool,
    },
    Any, // this is no longer required as soon as we switch to CDF completely
}

impl GpuColliderShape {
    #[cfg(not(target_os = "cuda"))]
    pub fn project_point_with_max_dist(
        &self,
        _position: &Isometry<Real>,
        _point: &Point<Real>,
        _solid: bool,
        _max_dist: Real,
    ) -> Option<PointProjection> {
        unreachable!()
    }

    #[cfg(target_os = "cuda")]
    pub fn project_point_with_max_dist(
        &self,
        position: &Isometry<Real>,
        point: &Point<Real>,
        solid: bool,
        max_dist: Real,
    ) -> Option<PointProjection> {
        use parry::query::PointQuery;

        match self {
            Self::Cuboid(s) => Some(s.project_point(position, point, solid)),
            Self::Polyline {
                vertices,
                flip_interior,
            } => polyline_project_point(vertices, position, point, solid, *flip_interior),
            Self::TriMesh {
                trimesh,
                flip_interior,
            } => trimesh
                .project_point_with_max_dist(position, point, solid, Real::MAX) // max_dist)
                .map(|mut proj| {
                    if *flip_interior {
                        proj.is_inside = !proj.is_inside;
                    }
                    proj
                }),
            Self::HeightField {
                heightfield,
                bellow_heightfield_is_solid,
                flip_interior,
            } => {
                let local_point = position.inverse_transform_point(point);
                let mut local_aabb = heightfield.local_aabb();
                local_aabb.mins.y = -Real::MAX;
                local_aabb.maxs.y = Real::MAX;
                let mut local_proj =
                    heightfield.project_local_point_with_max_dist(&local_point, solid, max_dist)?;

                if *bellow_heightfield_is_solid {
                    if !flip_interior {
                        local_proj.is_inside = local_aabb.contains_local_point(&local_point)
                            && local_point.y <= local_proj.point.y;
                    } else {
                        local_proj.is_inside = local_aabb.contains_local_point(&local_point)
                            && local_point.y >= local_proj.point.y;
                    }
                }

                Some(local_proj.transform_by(position))
            }
            _ => None,
        }
    }
}

#[cfg(target_os = "cuda")]
pub fn polyline_project_point(
    vertices: &CudaArrayPointer1<Point<Real>>,
    position: &Isometry<Real>,
    point: &Point<Real>,
    solid: bool,
    flip_interior: bool,
) -> Option<PointProjection> {
    #[derive(Copy, Clone)]
    struct BestProjection {
        proj: PointProjection,
        location: SegmentPointLocation,
        dist: Real,
        id: usize,
        segment: Segment,
    }

    let local_point = position.inverse_transform_point(point);

    // 1) First, identify the closest segment.
    let mut best_proj: Option<BestProjection> = None;

    let ith_segment = |i| Segment::new(vertices.get(i), vertices.get((i + 1) % vertices.len()));

    for i in 0..vertices.len() {
        let segment = ith_segment(i);
        let (proj, location) = segment.project_local_point_and_get_location(&local_point, false);
        let candidate_dist = na::distance(&local_point, &proj.point);

        if best_proj.map(|p| candidate_dist < p.dist).unwrap_or(true) {
            best_proj = Some(BestProjection {
                proj,
                location,
                dist: candidate_dist,
                id: i,
                segment,
            });
        }
    }

    if let Some(best_proj) = &mut best_proj {
        // 2) Inside/outside test (copied from Polyline::project_local_point_assuming_solid_interior_ccw)
        #[cfg(feature = "dim2")]
        let normal1 = best_proj.segment.normal();
        #[cfg(feature = "dim3")]
        let normal1 = best_proj.segment.planar_normal(2);

        if let Some(normal1) = normal1 {
            best_proj.proj.is_inside = match best_proj.location {
                SegmentPointLocation::OnVertex(i) => {
                    let dir2 = if i == 0 {
                        let adj_seg = if best_proj.id == 0 {
                            vertices.len() - 1
                        } else {
                            best_proj.id - 1
                        };

                        assert_eq!(best_proj.segment.a, ith_segment(adj_seg).b);
                        -ith_segment(adj_seg).scaled_direction()
                    } else {
                        assert_eq!(i, 1);
                        let adj_seg = (best_proj.id + 1) % vertices.len();
                        assert_eq!(best_proj.segment.b, ith_segment(adj_seg).a);

                        ith_segment(adj_seg).scaled_direction()
                    };

                    let dot = normal1.dot(&dir2);
                    // TODO: is this threshold too big? This corresponds to an angle equal to
                    //       abs(acos(1.0e-3)) = (90 - 0.057) degrees.
                    //       We did encounter some cases where this was needed, but perhaps the
                    //       actual problem was an issue with the SegmentPointLocation (which should
                    //       perhaps have been Edge instead of Vertex)?
                    let threshold = 1.0e-3 * dir2.norm();
                    if dot.abs() > threshold {
                        // If the vertex is a reentrant vertex, then the point is
                        // inside. Otherwise, it is outside.
                        dot >= 0.0
                    } else {
                        // If the two edges are collinear, we can’t classify the vertex.
                        // So check against the edge’s normal instead.
                        (point - best_proj.proj.point).dot(&normal1) <= 0.0
                    }
                }
                SegmentPointLocation::OnEdge(_) => {
                    (point - best_proj.proj.point).dot(&normal1) <= 0.0
                }
            };
        }
    }

    best_proj.map(|mut p| {
        #[cfg(feature = "dim3")]
        {
            p.proj.point.z = local_point.z;
        }

        if flip_interior {
            p.proj.is_inside = !p.proj.is_inside;
        }

        p.proj.transform_by(position)
    })
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GpuRigidBody {
    pub position: Isometry<Real>,
    pub linvel: Vector<Real>,
    pub angvel: AngVector<Real>,
    pub mass: Real,
    pub center_of_mass: Point<Real>,
    pub effective_inv_mass: Vector<Real>,
    #[cfg(feature = "dim2")]
    pub effective_world_inv_inertia_sqrt: Real,
    #[cfg(feature = "dim3")]
    pub effective_world_inv_inertia_sqrt: Matrix<Real>,
    pub linvel_update: Vector<Real>,
    pub angvel_update: AngVector<Real>,
    // pub impulse: Vector<Real>,
    // pub torque: AngVector<Real>,
    pub lock: u32,
}

impl Default for GpuRigidBody {
    fn default() -> Self {
        Self {
            position: Isometry::default(),
            linvel: na::zero(),
            angvel: na::zero(),
            mass: na::zero(),
            center_of_mass: Point::origin(),
            effective_inv_mass: na::zero(),
            effective_world_inv_inertia_sqrt: na::zero(),
            linvel_update: na::zero(),
            angvel_update: na::zero(),
            // impulse: na::zero(),
            // torque: na::zero(),
            lock: na::zero(),
        }
    }
}

impl GpuRigidBody {
    pub fn apply_particle_impulse(
        &mut self,
        impulse: Vector<Real>,
        particle_position: Point<Real>,
    ) {
        #[cfg(feature = "dim2")]
        let torque_impulse = {
            let difference = particle_position - self.center_of_mass;

            difference.x * impulse.y - difference.y + impulse.x
        };
        #[cfg(feature = "dim3")]
        let torque_impulse = (particle_position - self.center_of_mass).cross(&impulse);

        let linvel = self.effective_inv_mass.component_mul(&impulse);
        let angvel = self.effective_world_inv_inertia_sqrt
            * (self.effective_world_inv_inertia_sqrt * torque_impulse);

        unsafe {
            while self.lock.global_atomic_exch_acq(1) == 1 {}
            self.linvel_update += linvel;
            self.angvel_update += angvel;
            // self.impulse += impulse;
            // self.torque += torque_impulse;
            self.lock.global_atomic_exch_rel(0);
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GpuCollider {
    pub shape: GpuColliderShape,
    pub position: Isometry<Real>,
    pub friction: Real,
    pub boundary_condition: BoundaryCondition,
    pub rigid_body_index: Option<u32>,
    pub enable_cdf: bool,
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RigidParticle {
    pub position: Point<Real>,
    pub collider_index: u32, // Todo: consider packing both indices into a single u32
    pub segment_or_triangle_index: u32,
    pub color_index: u32, // Debug only, can be removed
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GpuRigidWorld {
    pub penalty_stiffness: Real,
    pub rigid_body_ptr: DevicePointer<GpuRigidBody>,
    pub collider_ptr: DevicePointer<GpuCollider>,
    pub rigid_particle_ptr: DevicePointer<RigidParticle>,
    pub vertex_ptr: DevicePointer<Point<Real>>,
    pub index_ptr: DevicePointer<u32>,
    pub collider_count: u32,
}

impl GpuRigidWorld {
    pub fn rigid_body(&self, i: u32) -> &GpuRigidBody {
        unsafe { &*self.rigid_body_ptr.as_ptr().add(i as usize) }
    }

    pub fn rigid_body_mut(&self, i: u32) -> &mut GpuRigidBody {
        unsafe { &mut *self.rigid_body_ptr.as_mut_ptr().add(i as usize) }
    }

    pub fn collider(&self, i: u32) -> &GpuCollider {
        unsafe { &*self.collider_ptr.as_ptr().add(i as usize) }
    }

    pub fn rigid_particle(&self, i: u32) -> &RigidParticle {
        unsafe { &*self.rigid_particle_ptr.as_ptr().add(i as usize) }
    }

    pub fn segment(&self, i: u32, position: &Isometry<Real>) -> Segment {
        unsafe {
            let index_a = *self.index_ptr.as_ptr().add(i as usize);
            let index_b = *self.index_ptr.as_ptr().add(i as usize + 1);

            let a = position * *self.vertex_ptr.as_ptr().add(index_a as usize);
            let b = position * *self.vertex_ptr.as_ptr().add(index_b as usize);

            Segment { a, b }
        }
    }

    pub fn triangle(&self, i: u32, position: &Isometry<Real>) -> Triangle {
        unsafe {
            let index_a = *self.index_ptr.as_ptr().add(i as usize);
            let index_b = *self.index_ptr.as_ptr().add(i as usize + 1);
            let index_c = *self.index_ptr.as_ptr().add(i as usize + 2);

            let a = position * *self.vertex_ptr.as_ptr().add(index_a as usize);
            let b = position * *self.vertex_ptr.as_ptr().add(index_b as usize);
            let c = position * *self.vertex_ptr.as_ptr().add(index_c as usize);

            Triangle { a, b, c }
        }
    }

    pub fn particle_collision(
        &self,
        particle_position: Point<Real>,
        particle_velocity: Vector<Real>,
        surface_normal: Vector<Real>,
        closest_collider_index: u32,
    ) -> Vector<Real> {
        let collider = self.collider(closest_collider_index);

        let collider_velocity = if let Some(rigid_body) = &mut collider
            .rigid_body_index
            .map(|index| self.rigid_body(index))
        {
            #[cfg(feature = "dim2")]
            {
                rigid_body.linvel
                    + rigid_body.angvel * (particle_position - rigid_body.center_of_mass)
            }
            #[cfg(feature = "dim3")]
            {
                rigid_body.linvel
                    + rigid_body
                        .angvel
                        .cross(&(particle_position - rigid_body.center_of_mass))
            }
        } else {
            na::zero()
        };

        collider_velocity
            + collider.boundary_condition.project(
                particle_velocity - collider_velocity,
                surface_normal,
                collider.friction,
            )
    }

    pub fn iter_colliders(&self) -> GpuColliderIter {
        GpuColliderIter {
            ptr: self.collider_ptr.as_ptr(),
            len: self.collider_count as usize,
            _marker: PhantomData,
        }
    }
}

pub struct GpuColliderIter<'a> {
    ptr: *const GpuCollider,
    len: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Iterator for GpuColliderIter<'a> {
    type Item = &'a GpuCollider;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            let curr = self.ptr;

            unsafe {
                self.ptr = self.ptr.offset(1);
                self.len -= 1;
                Some(&*curr)
            }
        }
    }
}
