use std::ops::Range;

use crate::texture;

/// The `Vertex` trait defines a common interface for vertex types.
///
/// It includes a single method `desc`, which returns a `wgpu::VertexBufferLayout`.
/// This method describes the layout of this vertex type in memory, which is used by the GPU to correctly read vertex data.
pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

/// The `ModelVertex` struct represents a vertex in a 3D model.
///
/// It contains fields for the position, texture coordinates, and normal vector of the vertex.
/// Each field is an array of `f32` values.
/// The `position` field represents the position of the vertex in 3D space.
/// The `tex_coords` field represents the texture coordinates of the vertex.
/// The `normal` field represents the normal vector of the vertex.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

/// Implementation of the `Vertex` trait for the `ModelVertex` struct.
///
/// This implementation provides a method `desc` that describes the layout of a `ModelVertex` in memory.
/// This layout description is used by the GPU to correctly read vertex data.
///
/// The `desc` method returns a `wgpu::VertexBufferLayout`, which includes the stride, step mode, and attributes of the vertex buffer.
/// The stride is the size of a `ModelVertex` in bytes.
/// The step mode is `Vertex`, which means that the stride is applied for each vertex.
/// The attributes describe the layout of each field in the `ModelVertex`.
impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// The `Material` struct represents a material in a 3D model.
///
/// It contains fields for the name of the material, the diffuse texture, and the bind group.
/// The `name` field is a `String` that holds the name of the material.
/// The `diffuse_texture` field is a `Texture` that holds the diffuse texture of the material.
/// The `bind_group` field is a `wgpu::BindGroup` that holds the bind group for the material.
pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

/// The `Mesh` struct represents a mesh in a 3D model.
///
/// It contains fields for the name of the mesh, the vertex buffer, the index buffer, the number of elements, and the material.
/// The `name` field is a `String` that holds the name of the mesh.
/// The `vertex_buffer` field is a `wgpu::Buffer` that holds the vertex buffer for the mesh.
/// The `index_buffer` field is a `wgpu::Buffer` that holds the index buffer for the mesh.
/// The `num_elements` field is a `u32` that holds the number of elements in the mesh.
/// The `material` field is a `usize` that holds the index of the material used by the mesh.
pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

/// The `Model` struct represents a 3D model.
///
/// It contains fields for the meshes and materials of the model.
/// The `meshes` field is a `Vec<Mesh>` that holds the meshes of the model.
/// The `materials` field is a `Vec<Material>` that holds the materials of the model.
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

/// The `DrawModel` trait defines methods for drawing 3D models.
///
/// It includes methods for drawing individual meshes, as well as entire models.
/// Each method takes a reference to a `Mesh` or `Model`, a `Material`, and bind groups for the camera and light.
/// There are also "instanced" versions of the methods that take a range of instances to draw.
pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

/// This block provides an implementation of the `DrawModel` trait for `wgpu::RenderPass`.
///
/// `wgpu::RenderPass` represents a render pass in a WebGPU application, which is a sequence of GPU commands for rendering.
/// By implementing `DrawModel` for `RenderPass`, we can use the drawing methods directly on a `RenderPass`.
impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(
                mesh,
                material,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}

/// The `DrawLight` trait defines methods for drawing light sources in a 3D scene.
///
/// It includes methods for drawing individual meshes and entire models with lighting.
/// Each method takes a reference to a `Mesh` or `Model`, and bind groups for the camera and light.
/// There are also "instanced" versions of the methods that take a range of instances to draw.
/// The `camera_bind_group` and `light_bind_group` parameters are used to pass data to the GPU about the camera and light source.
pub trait DrawLight<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

/// This block provides an implementation of the `DrawLight` trait for `wgpu::RenderPass`.
///
/// `wgpu::RenderPass` represents a render pass in a WebGPU application, which is a sequence of GPU commands for rendering.
/// By implementing `DrawLight` for `RenderPass`, we can use the drawing methods directly on a `RenderPass`.
impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(
        &mut self,
        mesh: &'b Mesh,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(
                mesh,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}
