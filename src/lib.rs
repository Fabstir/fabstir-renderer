// This Rust program renders 3D models (.obj or .gltf files) in real-time in a canvas using WebGPU.
// It is designed to be compiled to WebAssembly and called from JavaScript.
//
// Author: Jules Lai
//
// The code and rendering logic is based on the excellent tutorial
// "Render Pipelines in wgpu and Rust" by Ryosuke (https://sotrh.github.io/learn-wgpu/).

use std::iter;

use cgmath::prelude::*;
use serde::Deserialize;
use serde_wasm_bindgen::from_value;
//use std::time::Duration;
//use wasm_bindgen_futures::futures_0_3::FutureExt;
use futures::channel::mpsc;
use futures::StreamExt;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use winit::dpi::LogicalSize;
use winit::dpi::PhysicalSize;
use winit::event::KeyEvent;
use winit::platform::web::EventLoopExtWebSys;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

// Import the necessary crates and modules
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;
use winit::platform::web::WindowBuilderExtWebSys;

mod model;
mod resources;
mod texture;
mod utils;
use model::{DrawLight, DrawModel, Vertex};

// Constants for instances
const NUM_INSTANCES_PER_ROW: u32 = 1;
const MAX_TEXTURE_SIZE: u32 = 2048;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// The `Camera` struct represents a camera in the 3D space.
///
/// It contains fields for the following:
///
/// - `eye`: The position of the camera in the 3D space.
/// - `target`: The point in the 3D space that the camera is looking at.
/// - `up`: The up direction for the camera.
/// - `aspect`: The aspect ratio of the camera's field of view.
/// - `fovy`: The vertical field of view angle, in degrees.
/// - `znear`: The distance to the near clipping plane.
/// - `zfar`: The distance to the far clipping plane.
///
/// The `Camera` struct is typically used to define the viewpoint from which the 3D scene is rendered.
struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

/// Implementation block for the `Camera` struct.
///
/// This block contains methods that operate on the `Camera` struct. These methods can be called on instances of `Camera`.
///
/// Here's a brief overview of what each method does:
///
/// - `build_view_projection_matrix`: This method calculates the view projection matrix for the camera. The view projection matrix transforms points in the 3D world as seen from the camera's perspective.
impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        // We're using Vector4 because ofthe camera_uniform 16 byte spacing requirement
        self.view_position = camera.eye.to_homogeneous().into();
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

struct CameraController {
    speed: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        //                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the up/ down is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

// The `Instance` struct represents an instance of a 3D model in the scene.
// It contains fields for the position and rotation of the instance.
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

// Implementation block for the `Instance` struct.
impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

/// Implementation of the `model::Vertex` trait for the `InstanceRaw` struct.
///
/// This block contains methods that define how instances of `InstanceRaw` behave as vertices.
///
/// The `model::Vertex` trait is typically used to define the data for each vertex in a 3D model.
/// By implementing this trait for `InstanceRaw`, we're saying that instances of `InstanceRaw` can be used as vertices in a 3D model.
///
/// Here's a brief overview of what each method does:
///
/// - `desc`: This method returns a `wgpu::VertexBufferDescriptor` that describes the layout of this vertex type. This is used by the GPU to read data from the vertex buffer.
impl model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We don't have to do this in code though.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

/// The `State` struct represents the state of the WebGPU application.
///
/// It contains fields for the following:
///
/// - `surface`: The surface onto which the 3D model will be rendered.
/// - `device`: The logical device, which is the main interface to the GPU.
/// - `queue`: The command queue, which is used to submit commands to the GPU.
/// - `config`: The configuration for the surface, including the pixel format and dimensions.
/// - `clear_color`: The color used to clear the canvas before each render pass.
/// - `size`: The current size of the canvas.
/// - `render_pipeline`: The render pipeline, which encapsulates the GPU commands for rendering the 3D model.
/// - `depth_texture`: The depth texture, which is used for depth testing to correctly render overlapping objects.
/// - `camera`: The camera, which defines the viewpoint from which the 3D model is rendered.
/// - `camera_controller`: The camera controller, which handles user input for controlling the camera.
/// - `camera_buffer`: The uniform buffer for the camera, which sends the camera data to the GPU.
/// - `camera_bind_group`: The bind group for the camera, which binds the camera data to the GPU shaders.
/// - `camera_uniform`: The uniform data for the camera, which includes the view and projection matrices.
/// - `instances`: The instances of the 3D model to be rendered.
/// - `instance_buffer`: The vertex buffer for the instances, which sends the instance data to the GPU.
/// - `obj_model`: The 3D model to be rendered.
/// - `light_uniform`: The uniform data for the light, which includes the light's position and color.
/// - `light_buffer`: The uniform buffer for the light, which sends the light data to the GPU.
/// - `light_bind_group`: The bind group for the light, which binds the light data to the GPU shaders.
/// - `light_render_pipeline`: The render pipeline for the light, which encapsulates the GPU commands for rendering the light.
/// - `model_uri`: The model uri of the 3D model.
/// - `extension`: The file extension of the 3D model.
///
/// The `State` struct is typically instantiated once at the start of the application, and then passed by reference to any functions or methods that need to access or modify the shared state.
struct State {
    canvas: HtmlCanvasElement,
    // Graphic context
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Window size
    size: winit::dpi::PhysicalSize<u32>,
    // Clear color for mouse interactions
    clear_color: wgpu::Color,
    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    // Textures
    depth_texture: texture::Texture,
    // The layout for binding texture resources in a bind group.
    texture_bind_group_layout: wgpu::BindGroupLayout,
    // Camera
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Instances
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    // 3D Models
    obj_models: Vec<model::Model>,
    // Lighting
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline will be used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
    })
}

/// Implementation block for the `State` struct.
///
/// This block contains methods that operate on the `State` struct. These methods can be called on instances of `State`.
///
/// Here's a brief overview of what each method does:
///
/// - `new`: This is the constructor for the `State` struct. It initializes a new `State` with the given parameters.
///
/// - `resize`: This method is called when the window is resized. It updates the `size` field of the `State` and reconfigures the surface to match the new size.
///
/// - `input`: This method is called when there is user input. It updates the `camera_controller` field of the `State` based on the input.
///
/// - `update`: This method is called once per frame. It updates any state that needs to change from frame to frame, such as the `camera` field.
///
/// - `render`: This method is called once per frame. It renders the current frame to the surface.
impl State {
    // Initialize the state
    async fn new(
        canvas: HtmlCanvasElement,
        model_uris: Vec<String>,
        extensions: Vec<String>,
    ) -> Result<Self, String> {
        if model_uris.len() != extensions.len() {
            return Err(String::from(
                "model_uris and extensions vectors must have the same length",
            ));
        }

        let canvas = canvas.clone();
        let size = winit::dpi::PhysicalSize::new(canvas.width(), canvas.height());

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Version2,
        });

        let surface = instance
            .create_surface_from_canvas(canvas.clone())
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Select a device to use
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                // Some(&std::path::Path::new("trace")), // Trace path
                None,
            )
            .await
            .unwrap();

        // Config for surface
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb, // or wgpu::TextureFormat::Rgba8UnormSrgb
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        // Bind the texture to the renderer
        // This creates a general texture bind group
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // Bind the camera to the shaders

        let camera = Camera {
            eye: (0.0, 5.0, -10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let camera_controller = CameraController::new(0.2);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create instance buffer
        // We create a 2x2 grid of objects by doing 1 nested loop here
        // And use the "displacement" matrix above to offset objects with a gap
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position = cgmath::Vector3 { x, y: 0.0, z };

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                    };

                    Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        // We condense the matrix properties into a flat array (aka "raw data")
        // (which is how buffers work - so we can "stride" over chunks)
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        // Create the instance buffer with our data
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create a bind group for camera buffer
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Load model from disk or as a HTTP request (for web support)
        log::warn!("Load model");

        let mut obj_models = Vec::new();

        for (model_uri, extension) in model_uris.iter().zip(extensions.iter()) {
            let obj_model = resources::load_model(
                model_uri.as_str(),
                extension,
                &device,
                &queue,
                &texture_bind_group_layout,
            )
            .await
            .expect("Couldn't load model. Maybe path is wrong?");

            obj_models.push(obj_model);
        }

        // Lighting
        // Create light uniforms and setup buffer for them
        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind groups for lights
        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // Create depth texture
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Create the render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                // We add any bind groups here (texture and camera)
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        // Clear color used for mouse input interaction
        let clear_color = wgpu::Color::BLACK;

        Ok(Self {
            canvas,
            surface,
            device,
            queue,
            config,
            clear_color,
            size,
            render_pipeline,
            depth_texture,
            texture_bind_group_layout,
            camera,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instances,
            instance_buffer,
            obj_models,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
        })
    }

    pub async fn load_model(
        &mut self,
        model_uris: Vec<String>,
        extensions: Vec<String>,
    ) -> Result<(), String> {
        if model_uris.len() != extensions.len() {
            return Err(String::from(
                "load_model: model_uris and extensions vectors must have the same length",
            ));
        }

        web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: Load model"));
        log::warn!("Load model");

        let mut obj_models = Vec::new();

        web_sys::console::log_1(&JsValue::from_str(
            "wgpu_renderer: let mut obj_models = Vec::new();",
        ));

        for (model_uri, extension) in model_uris.iter().zip(extensions.iter()) {
            web_sys::console::log_1(&JsValue::from_str(
                "wgpu_renderer: before resources::load_model",
            ));

            let obj_model = resources::load_model(
                model_uri.as_str(),
                extension,
                &self.device,
                &self.queue,
                &self.texture_bind_group_layout,
            )
            .await
            .map_err(|_| format!("wgpu_renderer: Couldn't load model from uri: {}", model_uri))?;

            web_sys::console::log_1(&JsValue::from_str(
                "wgpu_renderer: after resources::load_model",
            ));

            obj_models.push(obj_model);

            web_sys::console::log_1(&JsValue::from_str(
                "wgpu_renderer: obj_models.push(obj_model);",
            ));
        }

        self.obj_models = obj_models;
        web_sys::console::log_1(&JsValue::from_str(
            "wgpu_renderer: self.obj_models = obj_models; ",
        ));

        Ok(())
    }

    // Keeps state in sync with window size when changed
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width.min(MAX_TEXTURE_SIZE);
            self.config.height = new_size.height.min(MAX_TEXTURE_SIZE);
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            log::info!("state resize: new_size = {:?}", new_size);
        }
    }

    // Handle input using WindowEvent
    fn input(&mut self, event: &WindowEvent) -> bool {
        // Send any input to camera controller
        self.camera_controller.process_events(event);

        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.clear_color = wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                };
                true
            }
            _ => false,
        }
    }

    /// This method is responsible for updating the state of the application for each frame.
    ///
    /// It's a part of the `State` struct implementation and is typically called once per frame, before the `render` method.
    ///
    /// The method doesn't take any parameters as it operates on the `State` struct, using its fields to perform the update.
    ///
    /// The method is marked as `mut` because it may change the state of the `State` struct, for example, by updating the position or orientation of objects in the scene, advancing animations, or processing user input.
    ///
    /// The method doesn't return a value, as it's assumed that any results of the update will be stored in the `State` struct for use by other methods.
    fn update(&mut self) {
        // Sync local app state with camera
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the light
        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
                * old_position)
                .into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    /// This method is responsible for rendering a frame.
    ///
    /// It's a part of the `State` struct implementation and is called once per frame.
    ///
    /// The method doesn't take any parameters as it operates on the `State` struct, using its fields to perform the rendering.
    ///
    /// The method returns a `Result` which is `Ok` if the frame was rendered successfully, and `Err` if there was an error during rendering. The error type is `wgpu::SurfaceError`, which represents errors that can occur when interacting with a `Surface`.
    ///
    /// The method is marked as `mut` because it may change the state of the `State` struct, for example, by updating the current frame number or other fields related to rendering state.
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        //        web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: inside"));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        //        web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: 1"));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        //        web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: 2"));

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Set the clear color during redraw
                        // This is basically a background color applied if an object isn't taking up space

                        // This sets it a color that changes based on mouse move
                        // load: wgpu::LoadOp::Clear(self.clear_color),

                        // A standard clear color
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // Create a depth stencil buffer using the depth texture
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            //            web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: 3"));

            // Setup our render pipeline with our config earlier in `new()`
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            // Setup lighting pipeline
            render_pass.set_pipeline(&self.light_render_pipeline);
            // Draw/calculate the lighting on models
            for obj_model in &self.obj_models {
                render_pass.draw_light_model(
                    obj_model,
                    &self.camera_bind_group,
                    &self.light_bind_group,
                );
            }

            //            web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: 4"));

            // Setup render pipeline
            render_pass.set_pipeline(&self.render_pipeline);
            // Draw the models
            for obj_model in &self.obj_models {
                render_pass.draw_model_instanced(
                    obj_model,
                    0..self.instances.len() as u32,
                    &self.camera_bind_group,
                    &self.light_bind_group,
                );
            }
        }

        //        web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: 5"));

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        //        web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: render: 6"));

        Ok(())
    }
}

#[wasm_bindgen]
extern "C" {
    fn get_canvas_size() -> JsValue;
}

#[derive(Deserialize)]
struct CanvasSize {
    width: u32,
    height: u32,
}

/// This function is responsible for getting the current size of the canvas from JavaScript.
///
/// It does not take any parameters.
///
/// The function returns a `LogicalSize<f64>` which represents the width and height of the canvas in logical pixels.
///
/// This function is called from the Rust side of the application, and fetches the size of the canvas from the JavaScript side.
/// This is useful in scenarios where the size of the canvas is controlled by JavaScript or CSS, and needs to be known on the Rust side for rendering.
///
/// Please note that this function assumes that a JavaScript function named `get_canvas_size` exists in the global scope, and that this function returns an object with `width` and `height` properties.
fn get_canvas_size_from_javascript() -> winit::dpi::PhysicalSize<u32> {
    let size = get_canvas_size();
    let size: CanvasSize = from_value(size).unwrap();
    winit::dpi::PhysicalSize::new(size.width, size.height)
}

// This flag indicates whether the rendering should continue or not.
// It's initially true, meaning the rendering should continue.
static mut CONTINUE_RENDERING: bool = true;

#[wasm_bindgen]
pub async fn stop_render() {
    // When this function is called from JavaScript, it sets the flag to false,
    // indicating that the rendering should stop.
    unsafe {
        CONTINUE_RENDERING = false;
        web_sys::console::log_1(&JsValue::from_str(
            "wgpu_renderer: set flag to stop rendering",
        ));
    }
}

lazy_static! {
    static ref SENDER: Mutex<mpsc::Sender<(Vec<String>, Vec<String>)>> = {
        let (sender, receiver) = mpsc::channel(1);
        *RECEIVER.lock().unwrap() = Some(receiver);
        Mutex::new(sender)
    };
    static ref RECEIVER: Mutex<Option<mpsc::Receiver<(Vec<String>, Vec<String>)>>> =
        Mutex::new(None);
}

#[wasm_bindgen]
pub fn load_model(model_uris: js_sys::Array, extensions: js_sys::Array) {
    web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: Inside load_model"));
    let new_model_uris: Vec<String> = model_uris
        .iter()
        .map(|js_value| js_value.as_string().expect("Expected string in array"))
        .collect();

    let new_extensions: Vec<String> = extensions
        .iter()
        .map(|js_value| js_value.as_string().expect("Expected string in array"))
        .collect();
    web_sys::console::log_1(&JsValue::from_str(
        "wgpu_renderer: new_extensions: Vec<String>",
    ));

    // Send the model_uris and extensions to the event loop
    let mut sender = SENDER.lock().unwrap();

    match sender.try_send((new_model_uris, new_extensions)) {
        Ok(_) => {
            // The message was sent successfully
        }
        Err(e) => {
            // The send operation failed
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "Failed to send message: {:?}",
                e
            )));
        }
    }
}

#[wasm_bindgen]
pub async fn render_model(
    canvas: HtmlCanvasElement,
    model_uris: js_sys::Array,
    extensions: js_sys::Array,
    callback: JsValue, // Accepting a JavaScript function as a callback
) {
    web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: Entered renderer"));
    // unsafe {
    //     CONTINUE_RENDERING = true;
    // }

    let model_uris: Vec<String> = model_uris
        .iter()
        .map(|js_value| js_value.as_string().expect("Expected string in array"))
        .collect();

    let extensions: Vec<String> = extensions
        .iter()
        .map(|js_value| js_value.as_string().expect("Expected string in array"))
        .collect();

    static mut LOGGER_INITIALIZED: bool = false;

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            unsafe {
                if !LOGGER_INITIALIZED {
                    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                    console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
                    LOGGER_INITIALIZED = true;
                }
            }
        } else {
            env_logger::init();
        }
    }

    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            log::warn!("Failed to create event loop: {}", e);
            return;
        }
    };

    log::info!("canvas width: {:?}", canvas.width());
    log::info!("canvas height: {:?}", canvas.height());

    let canvas_clone = canvas.clone();

    log::info!("canvas_clone width: {:?}", canvas_clone.width());
    log::info!("canvas_clone height: {:?}", canvas_clone.height());

    let size = LogicalSize::new(canvas.width() as f64, canvas.height() as f64);

    let window = WindowBuilder::new()
        .with_inner_size(size)
        .with_title("Fabstir renderer")
        .with_canvas(Some(canvas.clone()))
        .build(&event_loop)
        .unwrap();

    let canvas_clone = canvas.clone();
    log::info!("canvas_clone2 width: {:?}", canvas_clone.width());
    log::info!("canvas_clone2 height: {:?}", canvas_clone.height());

    // let mut next_frame_time = web_sys::window().unwrap().performance().unwrap().now();
    // let frame_duration = 100.0; // 10 frames per second

    // State::new uses async code, so we're going to wait for it to finish
    let state = match State::new(canvas, model_uris, extensions).await {
        Ok(state) => state,
        Err(e) => {
            // Handle error, log it to the console
            web_sys::console::error_1(&JsValue::from_str(&e));
            return;
        }
    };

    let future = async move {
        // your async code here
    };

    // Ensure the callback is a function
    if !callback.is_function() {
        web_sys::console::error_1(&JsValue::from_str("Callback is not a function"));
        return;
    }

    // Create a channel for sending model_uris and extensions from the closure to the event loop

    let callback = js_sys::Function::from(callback);
    let state = Arc::new(Mutex::new(state));
    //let canvas_clone = canvas.clone();

    event_loop.spawn(move |event, future| {
        // If the flag is false, exit the program.
        unsafe {
            if !CONTINUE_RENDERING {
                web_sys::console::log_1(&JsValue::from_str("wgpu_renderer: Rendering has stopped"));
                CONTINUE_RENDERING = true;
                std::process::exit(0);
            }
        }

        let mut receiver = RECEIVER.lock().unwrap();
        if let Some(receiver) = receiver.as_mut() {
            // Check if there are any new model_uris and extensions from the closure
            match receiver.try_next() {
                Ok(Some((new_model_uris, new_extensions))) => {
                    web_sys::console::log_1(&JsValue::from_str(
                        "wgpu_renderer: received channel message for load_model",
                    ));

                    let state_clone = Arc::clone(&state);
                    wasm_bindgen_futures::spawn_local(async move {
                        let mut state = state_clone.lock().unwrap();
                        match state.load_model(new_model_uris, new_extensions).await {
                            Ok(_) => {
                                web_sys::console::log_1(&JsValue::from_str(
                                    "wgpu_renderer: successfully loaded model",
                                ));
                            }
                            Err(e) => {
                                // Log the error
                                web_sys::console::error_1(&JsValue::from_str(&e));
                            }
                        }
                    });
                }
                Ok(None) => {
                    // The sender has been dropped, and no more messages will be received
                    web_sys::console::log_1(&JsValue::from_str(
                        "wgpu_renderer: channel sender dropped, no more messages will be received",
                    ));
                }
                Err(_) => {
                    // No message was available
                    // web_sys::console::log_1(&JsValue::from_str(
                    //     "wgpu_renderer: no message available in the channel",
                    // ));
                }
            }
        }

        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if let Ok(mut state) = state.try_lock() {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => return,
                            _ => {}
                        }
                    }
                } else {
                    // The state mutex is locked, so we skip this iteration
                }
            }
            _ => {}
        }

        {
            if let Ok(mut state) = state.try_lock() {
                let size = state.size;
                state.update();

                match state.render() {
                    Ok(_) => {
                        // Invoke the callback function after a successful render
                        let this = JsValue::NULL;
                        if let Err(e) = callback.call0(&this) {
                            web_sys::console::error_1(&JsValue::from_str(&format!(
                                "Callback error: {:?}",
                                e
                            )));
                        }
                    }
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(size)
                    }

                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => unsafe { CONTINUE_RENDERING = false },

                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }

                // Get the current size of the canvas from JavaScript
                let new_size = get_canvas_size_from_javascript();
                let size = LogicalSize::new(new_size.width as f64, new_size.height as f64);
                log::info!("Event::MainEventsCleared size = {:?}", size);

                let scale_factor = window.scale_factor();
                let physical_size: PhysicalSize<u32> = size.to_physical(scale_factor);
                log::info!(
                    "Event::MainEventsCleared physical_size = {:?}",
                    physical_size
                );

                state.resize(physical_size);

                let size_option = window.request_inner_size(physical_size);
                if let Some(size) = size_option {
                    log::info!("Inner size is now: {:?}", size);
                } else {
                    log::warn!("Failed to set inner size because window is not resizable");
                }
            }
        }

        // Request a redraw
        window.request_redraw();
    });
}
