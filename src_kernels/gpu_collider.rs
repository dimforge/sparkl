use crate::DevicePointer;
use core::marker::PhantomData;
use na::ComplexField;
use parry::query::{PointProjection, PointQueryWithLocation};
use parry::shape::{Cuboid, CudaHeightFieldPtr, CudaTriMeshPtr, Segment, SegmentPointLocation};
use parry::utils::CudaArrayPointer1;
use sparkl_core::dynamics::solver::BoundaryHandling;
use sparkl_core::math::{Isometry, Point, Real};

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
        vertices: parry::utils::CudaArrayPointer1<Point<Real>>,
        flip_interior: bool,
    },
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
pub struct GpuCollider {
    pub shape: GpuColliderShape,
    pub position: Isometry<Real>,
    pub friction: Real,
    pub penalty_stiffness: Real,
    pub grid_boundary_handling: BoundaryHandling,
}

pub struct GpuColliderSet {
    pub ptr: *const GpuCollider,
    pub len: usize,
}

impl GpuColliderSet {
    pub fn get(&self, i: usize) -> Option<&GpuCollider> {
        if i >= self.len {
            None
        } else {
            unsafe { Some(&*self.ptr.add(i)) }
        }
    }

    pub fn iter(&self) -> GpuColliderIter {
        GpuColliderIter {
            ptr: self.ptr,
            len: self.len,
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
