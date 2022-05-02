use crate::dynamics::GridNode;
use crate::geometry::SpGrid;

pub trait MpmHooks: Send + Sync {
    fn post_grid_update_hook(&mut self, grid: &mut SpGrid<GridNode>);
}

impl MpmHooks for () {
    fn post_grid_update_hook(&mut self, _: &mut SpGrid<GridNode>) {
        /* nothing */
    }
}
