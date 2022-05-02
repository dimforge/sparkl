use crate::dynamics::{GridNode, Particle};
use crate::geometry::SpGrid;
use crate::math::DIM;
use parry::utils::hashmap::HashMap;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::atomic::Ordering;

#[derive(Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ParticleSet {
    pub(crate) particles: Vec<Particle>,
    pub(crate) order: Vec<usize>,
    pub(crate) regions: Vec<(u64, Range<usize>)>,
    pub(crate) active_regions: HashMap<u64, ()>,
    pub(crate) active_cells: Vec<u64>,
    pub(crate) particle_bins: Vec<[usize; 5]>,
}

impl ParticleSet {
    pub fn new() -> Self {
        Self {
            particles: vec![],
            order: vec![],
            regions: vec![],
            active_regions: HashMap::default(),
            active_cells: vec![],
            particle_bins: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    pub fn iter(&self) -> impl Iterator<Item = &Particle> {
        self.particles.iter()
    }

    pub fn set_particles(&mut self, particles: Vec<Particle>) {
        self.particles = particles;
        self.order = (0..self.particles.len()).collect();
    }

    pub fn retain(&mut self, p: impl Fn(&Particle) -> bool) {
        self.particles.retain(p);
        self.order = (0..self.particles.len()).collect();
    }

    pub fn remove_range(&mut self, range: Range<usize>) {
        drop(self.particles.drain(range));
        self.order = (0..self.particles.len()).collect();
    }

    pub fn insert(&mut self, particle: Particle) {
        self.order.push(self.order.len());
        self.particles.push(particle);
    }

    pub fn insert_batch(&mut self, mut particles: Vec<Particle>) {
        let index_range = self.order.len()..self.order.len() + particles.len();
        self.order.extend(index_range);
        self.particles.append(&mut particles);
    }

    pub fn get_sorted_mut(&mut self, i: usize) -> Option<&mut Particle> {
        self.particles.get_mut(*self.order.get(i)? as usize)
    }

    pub fn get_sorted_mut2(
        &mut self,
        i: usize,
        j: usize,
    ) -> Option<(&mut Particle, &mut Particle)> {
        if i == j || i > self.particles.len() || j > self.particles.len() {
            None
        } else {
            unsafe {
                let p1 = &mut self.particles[self.order[i]] as *mut _;
                let p2 = &mut self.particles[self.order[j]] as *mut _;
                Some((std::mem::transmute(p1), std::mem::transmute(p2)))
            }
        }
    }

    // pub fn iter_bodies(&self) -> impl Iterator<Item = (&ParticleModel, &[Particle])> {
    //     self.bodies
    //         .iter()
    //         .map(move |body| (body, &self.particles[body.particles.clone()]))
    // }
    //
    // pub fn map_bodies_mut(&mut self, mut f: impl FnMut(&ParticleModel, &mut [Particle])) {
    //     for body in &self.bodies {
    //         f(body, &mut self.particles[body.particles.clone()])
    //     }
    // }

    pub fn sorted_particle_indices(&self, range: Range<usize>) -> &[usize] {
        &self.order[range]
    }

    pub fn iter_sorted_in_range(&self, range: Range<usize>) -> impl Iterator<Item = &Particle> {
        self.order[range].iter().map(move |i| &self.particles[*i])
    }

    #[inline(always)]
    pub fn for_each_particles_mut(&mut self, f: impl Fn(&mut Particle) + Sync) {
        self.particles.par_iter_mut().for_each(|p| f(p))
        // let particles = std::sync::atomic::AtomicPtr::new(self as *mut _);
        // let order = &self.order;
        //
        // order.par_iter().for_each(|i| {
        //     let particles: &mut Self =
        //         unsafe { std::mem::transmute(particles.load(Ordering::Relaxed)) };
        //     f(&mut particles.particles[*i])
        // })
    }

    pub fn sort(&mut self, update_particle_range: bool, grid: &mut SpGrid<GridNode>) {
        let t0 = instant::now();
        self.particles.par_iter_mut().for_each(|particle| {
            particle.grid_index = grid.cell_associated_to_point(&particle.position);

            if !grid.is_neighborhood_valid(particle.grid_index) {
                particle.grid_index = u64::MAX;
                particle.failed = true;
            }
        });

        info!("- Index packing: {}", instant::now() - t0);

        let particles = &self.particles;
        let t0 = instant::now();
        self.order.par_sort_by_key(|i| particles[*i].grid_index);
        // self.particles.par_sort_by_key(|p| p.grid_index);
        info!("- Actual sort time: {}", instant::now() - t0);

        // Reset the previously active regions.
        let t0 = instant::now();
        {
            let grid = &std::sync::atomic::AtomicPtr::new(grid as *mut _);

            self.regions.par_iter().for_each(|region_id| {
                let grid: &mut SpGrid<GridNode> =
                    unsafe { std::mem::transmute(grid.load(Ordering::Relaxed)) };

                for i in 0..4u64.pow(DIM as u32) {
                    let cell_id = region_id.0 | (i << SpGrid::<()>::PACK_ALIGN);
                    let cell = grid.get_packed_mut(cell_id);
                    cell.reset();
                }
            });
        }
        info!("- Region reset: {}", instant::now() - t0);

        // Update the regions/cell associations.
        let t0 = instant::now();
        self.regions.clear();
        self.active_regions.clear();
        self.active_cells.clear();
        self.particle_bins.clear();

        let mut curr_region_range_start = 0;
        let mut curr_cell = self.particles[self.order[0]].grid_index;
        let mut curr_bin = [usize::MAX; 5];
        let mut curr_bin_len = 0;
        let mut curr_model = self.particles[self.order[0]].model;

        if update_particle_range {
            let mut curr_cell_range_start = 0;

            for (i, order_id) in self.order.iter().enumerate() {
                let particle = &self.particles[*order_id];
                let new_cell = particle.grid_index;
                let new_model = particle.model;

                if new_cell != curr_cell && curr_cell != u64::MAX {
                    grid.get_packed_mut(curr_cell).particles = (curr_cell_range_start, i as u32);

                    let curr_region_id = curr_cell & SpGrid::<GridNode>::REGION_ID_MASK;
                    if new_cell & SpGrid::<GridNode>::REGION_ID_MASK != curr_region_id {
                        self.regions
                            .push((curr_region_id, curr_region_range_start..i));
                        self.active_regions.insert(curr_region_id, ());
                        curr_region_range_start = i;
                    }

                    curr_cell_range_start = i as u32;
                }

                if new_cell != curr_cell || new_model != curr_model || curr_bin_len == 4 {
                    self.particle_bins.push(curr_bin);
                    curr_bin = [usize::MAX; 5];
                    curr_bin_len = 0;
                }

                curr_bin[curr_bin_len] = *order_id;
                curr_bin_len += 1;
                curr_cell = new_cell;
                curr_model = new_model;
            }

            // Set the last region/cell.
            if curr_cell != u64::MAX {
                grid.get_packed_mut(curr_cell).particles =
                    (curr_cell_range_start, self.order.len() as u32);
            }
        } else {
            for (i, order_id) in self.order.iter().enumerate() {
                let particle = self.particles[*order_id];
                let new_cell = particle.grid_index;
                let new_model = particle.model;
                let new_region_id = new_cell & SpGrid::<GridNode>::REGION_ID_MASK;
                let curr_region_id = curr_cell & SpGrid::<GridNode>::REGION_ID_MASK;

                if new_region_id != curr_region_id {
                    self.regions
                        .push((curr_region_id, curr_region_range_start..i));
                    self.active_regions.insert(curr_region_id, ());
                    curr_region_range_start = i;
                }

                if new_cell != curr_cell || new_model != curr_model || curr_bin_len == 4 {
                    self.particle_bins.push(curr_bin);
                    curr_bin = [usize::MAX; 5];
                    curr_bin_len = 0;
                }

                curr_bin[curr_bin_len] = *order_id;
                curr_bin_len += 1;
                curr_cell = new_cell;
                curr_model = new_model;
            }
        }

        if curr_cell != u64::MAX {
            let curr_region_id = curr_cell & SpGrid::<GridNode>::REGION_ID_MASK;
            self.regions
                .push((curr_region_id, curr_region_range_start..self.order.len()));
            self.active_regions.insert(curr_region_id, ());

            if curr_bin_len > 0 {
                self.particle_bins.push(curr_bin);
            }
        }
        info!("- Big loop: {}", instant::now() - t0);

        let mut bins_stats = [0; 4];
        for bin in &self.particle_bins {
            for ii in 0..4 {
                if bin[ii + 1] == usize::MAX {
                    bins_stats[ii] += 1;
                }
            }
        }

        info!("Bin stats: {:?}", bins_stats);

        let t0 = instant::now();
        for i in 0..self.regions.len() {
            let region_id = self.regions[i].0;
            for nbh_id in SpGrid::<()>::region_neighbors(region_id) {
                if self.active_regions.insert(nbh_id, ()).is_none() {
                    self.regions.push((nbh_id, 0..0)) // Empty region that we will touch.
                }
            }
        }
        info!("- Region expansion: {}", instant::now() - t0);
    }
}

impl<'a> IntoIterator for &'a ParticleSet {
    type Item = &'a Particle;
    type IntoIter = <&'a Vec<Particle> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.particles).into_iter()
    }
}
