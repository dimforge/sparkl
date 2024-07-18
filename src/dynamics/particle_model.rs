use crate::dynamics::models::{ConstitutiveModel, FailureModel, PlasticModel};
use rapier::data::{Arena, Index};
use std::sync::Arc;

#[cfg(feature = "serde-serialize")]
use {
    crate::dynamics::models::{CoreConstitutiveModel, CoreFailureModel, CorePlasticModel},
    serde::{Deserialize, Deserializer, Serialize, Serializer},
};

pub type ParticleModelHandle = Index;

#[derive(Clone)]
pub struct ParticleModel {
    pub constitutive_model: Arc<dyn ConstitutiveModel>,
    pub plastic_model: Option<Arc<dyn PlasticModel>>,
    pub failure_model: Option<Arc<dyn FailureModel>>,
}

#[cfg(feature = "serde-serialize")]
impl Serialize for ParticleModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let typed_data = (
            self.constitutive_model.to_core_model(),
            self.plastic_model.as_ref().map(|m| {
                m.to_core_model()
                    .expect("Unsupported model for serialization.")
            }),
            self.failure_model.as_ref().map(|m| {
                m.to_core_model()
                    .expect("Unsupported model for serialization.")
            }),
        );
        typed_data.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de> Deserialize<'de> for ParticleModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let typed_data: (
            CoreConstitutiveModel,
            Option<CorePlasticModel>,
            Option<CoreFailureModel>,
        ) = Deserialize::deserialize(deserializer)?;

        let constitutive_model = match typed_data.0 {
            CoreConstitutiveModel::EosMonaghanSph(m) => Arc::new(m) as Arc<dyn ConstitutiveModel>,
            CoreConstitutiveModel::NeoHookeanElasticity(m) => {
                Arc::new(m) as Arc<dyn ConstitutiveModel>
            }
            CoreConstitutiveModel::CorotatedLinearElasticity(m) => {
                Arc::new(m) as Arc<dyn ConstitutiveModel>
            }
            CoreConstitutiveModel::Custom(_) => todo!(),
        };
        let plastic_model = typed_data.1.map(|data| match data {
            CorePlasticModel::Snow(m) => Arc::new(m) as Arc<dyn PlasticModel>,
            CorePlasticModel::Rankine(m) => Arc::new(m) as Arc<dyn PlasticModel>,
            CorePlasticModel::Nacc(m) => Arc::new(m) as Arc<dyn PlasticModel>,
            CorePlasticModel::DruckerPrager(m) => Arc::new(m) as Arc<dyn PlasticModel>,
            CorePlasticModel::Custom(_) => todo!(),
        });

        let failure_model = typed_data.2.map(|data| match data {
            CoreFailureModel::MaximumStress(m) => Arc::new(m) as Arc<dyn FailureModel>,
            CoreFailureModel::Custom(_) => todo!(),
        });

        Ok(Self {
            constitutive_model,
            plastic_model,
            failure_model,
        })
    }
}

impl ParticleModel {
    pub fn new(constitutive_model: impl ConstitutiveModel + 'static) -> Self {
        Self {
            constitutive_model: Arc::new(constitutive_model),
            plastic_model: None,
            failure_model: None,
        }
    }

    pub fn with_plasticity(
        constitutive_model: impl ConstitutiveModel + 'static,
        plasticity: impl PlasticModel + 'static,
    ) -> Self {
        Self {
            constitutive_model: Arc::new(constitutive_model),
            plastic_model: Some(Arc::new(plasticity)),
            failure_model: None,
        }
    }

    pub fn with_failure(
        constitutive_model: impl ConstitutiveModel + 'static,
        failure: impl FailureModel + 'static,
    ) -> Self {
        Self {
            constitutive_model: Arc::new(constitutive_model),
            plastic_model: None,
            failure_model: Some(Arc::new(failure)),
        }
    }
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ParticleModelSet {
    models: Arena<ParticleModel>,
}

impl ParticleModelSet {
    pub fn new() -> Self {
        Self {
            models: Arena::new(),
        }
    }

    pub fn insert(&mut self, model: ParticleModel) -> ParticleModelHandle {
        self.models.insert(model)
    }

    pub fn get(&self, handle: ParticleModelHandle) -> Option<&ParticleModel> {
        self.models.get(handle)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Index, &ParticleModel)> {
        self.models.iter()
    }
}

impl std::ops::Index<ParticleModelHandle> for ParticleModelSet {
    type Output = ParticleModel;

    #[inline]
    fn index(&self, i: ParticleModelHandle) -> &ParticleModel {
        &self.models[i]
    }
}

impl std::ops::IndexMut<ParticleModelHandle> for ParticleModelSet {
    #[inline]
    fn index_mut(&mut self, i: ParticleModelHandle) -> &mut ParticleModel {
        &mut self.models[i]
    }
}
