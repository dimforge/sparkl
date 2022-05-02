use crate::dynamics::models::{
    DruckerPragerPlasticity, NaccPlasticity, RankinePlasticity, SnowPlasticity,
};

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum CorePlasticModel {
    DruckerPrager(DruckerPragerPlasticity),
    Nacc(NaccPlasticity),
    Rankine(RankinePlasticity),
    Snow(SnowPlasticity),
    Custom(u32),
}
