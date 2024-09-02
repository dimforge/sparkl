use std::fs::read_to_string;

use bevy::{
    core_pipeline::core_3d::{Opaque3d, Opaque3dBinKey},
    ecs::system::{lifetimeless::*, SystemParamItem},
    math::prelude::*,
    pbr::{
        MeshPipeline, MeshPipelineKey, MeshTransforms, RenderMeshInstances, SetMeshBindGroup,
        SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{GpuBufferInfo, GpuMesh, MeshVertexBufferLayout, MeshVertexBufferLayoutRef},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, BinnedRenderPhase, BinnedRenderPhaseType, DrawFunctions, PhaseItem,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::{ExtractedView, Msaa, VisibleEntities},
        Render, RenderApp, RenderSet,
    },
};
use bevy_ecs::query::ROQueryItem;
use bytemuck::{Pod, Zeroable};

// From: https://discordapp.com/channels/691052431525675048/1170930650606489711/1170944743962849380
const PARTICLE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(10091001291240510013);

#[derive(Component, Clone)]
pub struct ParticleInstanceMaterialData(pub Vec<ParticleInstanceData>);
impl ExtractComponent for ParticleInstanceMaterialData {
    type QueryData = &'static ParticleInstanceMaterialData;
    type QueryFilter = ();

    fn extract_component(item: bevy::ecs::query::QueryItem<Self::QueryData>) -> Option<Self> {
        Some(ParticleInstanceMaterialData(item.0.clone()))
    }

    type Out = Self;
}

pub struct ParticleMaterialPlugin;

impl Plugin for ParticleMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ParticleInstanceMaterialData>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Opaque3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<ParticleRenderPipeline>>()
            .insert_resource(Msaa::default())
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSet::QueueMeshes),
                    prepare_instance_buffers.in_set(RenderSet::PrepareResources),
                ),
            );

        let mut shaders = app
            .world_mut()
            .get_resource_mut::<Assets<Shader>>()
            .unwrap();

        const WGSL_PATH: &'static str = "../src/third_party/rapier/shaders/instancing3d.wgsl";

        shaders.get_or_insert_with(&PARTICLE_SHADER_HANDLE, || {
            Shader::from_wgsl(
                read_to_string(WGSL_PATH).expect("Couldn't read particle shader"),
                WGSL_PATH,
            )
        });
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<ParticleRenderPipeline>();
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ParticleInstanceData {
    pub position: Vec3,
    pub scale: f32,
    pub color: [f32; 4],
}

fn queue_custom(
    opaque_3d_draw_functions: Res<DrawFunctions<Opaque3d>>,
    custom_pipeline: Res<ParticleRenderPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<ParticleRenderPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<GpuMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    material_meshes: Query<Entity, With<ParticleInstanceMaterialData>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    views: Query<(Entity, &ExtractedView)>,
) {
    let draw_custom = opaque_3d_draw_functions.read().id::<DrawCustom>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

    for (view_entity, view) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        //let rangefinder = view.rangefinder3d();
        for entity in &material_meshes {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(entity) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let key =
                view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();
            let bin_key = Opaque3dBinKey {
                draw_function: draw_custom,
                pipeline,
                asset_id: mesh_instance.mesh_asset_id.untyped(),
                material_bind_group_id: None,
                lightmap_image: None,
            };
            opaque_phase.add(
                bin_key,
                entity,
                BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
            );
            /*
            transparent_phase.add(Opaque3d {
                asset_id: mesh_instance.mesh_asset_id,
                entity,
                pipeline,
                draw_function: draw_custom,
                //distance: rangefinder.distance_translation(&mesh_instance.transforms.transform.translation),
                batch_range: 0..1,
                key: todo!(),
                representative_entity: todo!(),
                extra_index: todo!(),
            });*/
        }
    }
}

#[derive(Component)]
pub struct ParticleInstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &ParticleInstanceMaterialData)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in query.iter() {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("particle instance data buffer"),
            contents: bytemuck::cast_slice(instance_data.0.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(ParticleInstanceBuffer {
            buffer,
            length: instance_data.0.len(),
        });
    }
}

#[derive(Resource)]
pub struct ParticleRenderPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for ParticleRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.resource::<MeshPipeline>();

        ParticleRenderPipeline {
            shader: PARTICLE_SHADER_HANDLE,
            mesh_pipeline: mesh_pipeline.clone(),
        }
    }
}

impl SpecializedMeshPipeline for ParticleRenderPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        // meshes typically live in bind group 2. because we are using bindgroup 1
        // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
        // linked in the shader
        descriptor
            .vertex
            .shader_defs
            .push("MESH_BINDGROUP_1".into());

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleInstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                // i_pos_scale
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                // i_color
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        descriptor.label = Some("particles_pipeline".into());

        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawParticlesInstanced,
);

pub struct DrawParticlesInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawParticlesInstanced {
    type Param = (SRes<RenderAssets<GpuMesh>>, SRes<RenderMeshInstances>);
    type ViewQuery = ();
    type ItemQuery = (Option<Entity>, Read<ParticleInstanceBuffer>);

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        item_query: ROQueryItem<'w, Option<(Option<Entity>, &'w ParticleInstanceBuffer)>>,
        (meshes, render_mesh_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some((Some(entity), instance_buffer)) = item_query else {
            return RenderCommandResult::Failure;
        };
        let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(entity) else {
            return RenderCommandResult::Failure;
        };
        let gpu_mesh = match meshes.into_inner().get(mesh_instance.mesh_asset_id) {
            Some(gpu_mesh) => gpu_mesh,
            None => {
                return RenderCommandResult::Failure;
            }
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..gpu_mesh.vertex_count, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}

pub fn init_renderer(app: &mut App) {
    app.add_plugins(ParticleMaterialPlugin);
}
