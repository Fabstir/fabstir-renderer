use crate::texture::Texture;
use crate::utils::download_bytes;
use base64::{decode, encode};
use gltf::Gltf;
use std::io::{BufReader, Cursor};
use std::path::Path;
use std::sync::Arc;

use wgpu::util::DeviceExt;

extern crate image;

use crate::{model, texture};

/// This function is responsible for loading a texture from a file.
///
/// It takes a `wgpu::Device`, `wgpu::Queue`, and the file name as parameters, and returns a `Texture` struct.
///
/// The function reads the file, decodes the image data, creates a texture on the GPU, and copies the image data to the texture.
/// It also creates a texture view and a sampler for the texture.
///
/// The `Texture` struct, which includes the texture, its view, and its sampler, can then be used for rendering.
pub async fn load_texture(
    texture_uri: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = download_bytes(texture_uri).await?;

    texture::Texture::from_bytes(device, queue, &data, "model")
}

/// This function is responsible for loading a 3D model.
///
/// It takes three parameters:
/// - `device`: A reference to the `wgpu::Device` which represents the physical device (GPU) used for rendering.
/// - `queue`: A reference to the `wgpu::Queue` which is used to submit commands to the GPU.
/// - `layout`: A reference to the `wgpu::BindGroupLayout` which describes the layout of a bind group.
/// - `model_uri`: The uri of the file containing the 3D model.
/// - `extension`: The file extension of the 3D model file.
///
/// This function is asynchronous and returns a `Future` that resolves to a `Model` struct. The `Model` struct contains the meshes and materials of the 3D model.
///
/// The function uses the `extension` parameter to determine how to parse the 3D model file. If the extension is "obj", it uses the `tobj` crate to parse the file. If the extension is "gltf", it uses the `gltf` crate to parse the file.
///
/// The function also creates a texture and bind group for each material in the 3D model.
pub async fn load_model(
    model_uri: &str,
    extension: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    // ... rest of your code ...

    match extension {
        // This section of the code is responsible for loading a 3D model in .obj format.
        //
        // The process involves the following steps:
        // 1. Download the .obj file and store it in memory.
        // 2. Parse the .obj file using the `tobj` crate, which provides functions for loading .obj and .mtl files.
        // 3. Process the materials defined in the .obj file, creating a texture and bind group for each one.
        // 4. Process the meshes defined in the .obj file, creating a vertex buffer and index buffer for each one.
        // 5. Return a `Model` struct containing the meshes and materials.
        //
        // The `Model` struct can then be used to render the 3D model in the WebGPU application.
        "obj" => {
            let file_bytes = Arc::new(download_bytes(model_uri).await?);

            let file_bytes_clone = Arc::clone(&file_bytes); // Clone it here
            let obj_cursor = Cursor::new((*file_bytes_clone).as_slice());
            let mut obj_reader = BufReader::new(obj_cursor);

            // Clone file_bytes again for use inside the closure
            let file_bytes_clone2 = Arc::clone(&file_bytes); // Clone it here again

            let (models, obj_materials) = tobj::load_obj_buf_async(
                &mut obj_reader,
                &tobj::LoadOptions {
                    triangulate: true,
                    single_index: true,
                    ..Default::default()
                },
                |p| {
                    let file_bytes_clone3 = Arc::clone(&file_bytes_clone2); // Clone inside the closure
                    async move {
                        tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(
                            (*file_bytes_clone3).as_slice(),
                        )))
                    }
                },
            )
            .await?;

            let mut materials = Vec::new();
            for m in obj_materials? {
                //                let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;
                let diffuse_texture = load_texture(
                    m.diffuse_texture.as_deref().unwrap_or("default"),
                    device,
                    queue,
                )
                .await?;
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                        },
                    ],
                    label: None,
                });

                materials.push(model::Material {
                    name: m.name,
                    diffuse_texture,
                    bind_group,
                })
            }

            let meshes = models
                .into_iter()
                .map(|m| {
                    let vertices = (0..m.mesh.positions.len() / 3)
                        .map(|i| model::ModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                            normal: [
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ],
                        })
                        .collect::<Vec<_>>();

                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Vertex Buffer", model_uri)),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Index Buffer", model_uri)),
                            contents: bytemuck::cast_slice(&m.mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                    model::Mesh {
                        name: model_uri.to_string(),
                        vertex_buffer,
                        index_buffer,
                        num_elements: m.mesh.indices.len() as u32,
                        material: m.mesh.material_id.unwrap_or(0),
                    }
                })
                .collect::<Vec<_>>();
            Ok(model::Model { meshes, materials })
        }
        // This section of the code is responsible for loading a 3D model in .gltf format.
        //
        // The process involves the following steps:
        // 1. Download the .gltf file and store it in memory.
        // 2. Parse the .gltf file using the `gltf` crate, which provides functions for loading .gltf and .bin files.
        // 3. Load binary data from the buffers defined in the .gltf file.
        // 4. Process the materials defined in the .gltf file, creating a texture and bind group for each one.
        // 5. Process the meshes defined in the .gltf file, creating a vertex buffer and index buffer for each one.
        // 6. Return a `Model` struct containing the meshes and materials.
        //
        // The `Model` struct can then be used to render the 3D model in the WebGPU application.
        "gltf" => {
            // Load the GLTF file
            let gltf_path = Path::new(model_uri);
            let gltf_directory = Path::new("assets");

            let file_bytes = Arc::new(download_bytes(model_uri).await?);
            let gltf = Gltf::from_slice(&file_bytes).unwrap();

            // Load the GLTF file
            // let full_path = gltf_directory.join(gltf_path);
            // let gltf = Gltf::open(full_path).unwrap();

            // Load binary data from the buffers
            let buffers: Vec<Vec<u8>> = gltf
                .buffers()
                .map(|buffer| {
                    match buffer.source() {
                        gltf::buffer::Source::Uri(uri) => {
                            if uri.starts_with("data:") {
                                // Handle data URI
                                let encoded_data = uri.split(",").nth(1).unwrap();
                                let decoded_data = base64::decode(encoded_data).unwrap();
                                decoded_data
                            } else {
                                // Handle file URI
                                let full_path = gltf_directory.join(uri);
                                std::fs::read(full_path).unwrap()
                            }
                        }
                        gltf::buffer::Source::Bin => {
                            // Handle the Bin variant
                            unimplemented!()
                        }
                    }
                })
                .collect();

            let mut materials = Vec::new();
            for mat in gltf.materials() {
                let texture = if let Some(base_color_texture) =
                    mat.pbr_metallic_roughness().base_color_texture()
                {
                    let image = base_color_texture.texture().source();
                    match image.source() {
                        gltf::image::Source::View { view, mime_type } => {
                            let buffer = &buffers[view.buffer().index()];
                            let start = view.offset();
                            let end = start + view.length();
                            let image_bytes = &buffer[start..end];
                            Texture::from_bytes(device, queue, image_bytes, "Base Color").unwrap()
                        }
                        gltf::image::Source::Uri { uri, .. } => {
                            // Load image data from URI (either a file path or a data URI)

                            let image_path = gltf_directory.join(uri);
                            println!("Loading image from {:?}", image_path);
                            let img = image::open(&image_path)?.into_rgba8();
                            let dimensions = img.dimensions(); // Get the dimensions of the image

                            let image_bytes = img.into_raw();

                            // Print out some information about the image data
                            println!("Image data length: {}", image_bytes.len());
                            println!(
                                "First 10 bytes of image data: {:?}",
                                &image_bytes[0..10.min(image_bytes.len())]
                            );

                            let texture_result = Texture::from_raw(
                                device,
                                queue,
                                &image_bytes,
                                dimensions,
                                Some("Base Color"),
                            ); // Use from_raw instead of from_bytes
                            println!("{:?}", texture_result); // This will print the result (either Ok or Err)
                            texture_result? // This will return the texture, or propagate the error if there is one.
                        }
                    }
                } else {
                    // You can create a solid color texture like this:
                    let red: [u8; 4] = [255, 0, 0, 255]; // Red in RGBA
                    Texture::from_bytes(device, queue, &red, "Solid Red").unwrap()
                };

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&texture.sampler),
                        },
                    ],
                    label: Some("texture_bind_group"),
                });

                materials.push(model::Material {
                    name: mat.name().unwrap_or_default().to_string(),
                    diffuse_texture: texture,
                    bind_group,
                });
            }

            // Load meshes
            let meshes: Vec<model::Mesh> = gltf
                .meshes()
                .flat_map(|mesh_data| {
                    // Using flat_map to flatten the Vec<Vec<model::Mesh>> to Vec<model::Mesh>
                    mesh_data
                        .primitives()
                        .map(|primitive| {
                            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                            // Load vertices
                            let vertex_data: Vec<model::ModelVertex> = reader
                                .read_positions()
                                .unwrap()
                                .zip(reader.read_tex_coords(0).unwrap().into_f32())
                                .zip(reader.read_normals().unwrap())
                                .map(|((position, tex_coords), normal)| model::ModelVertex {
                                    position,
                                    tex_coords,
                                    normal,
                                })
                                .collect();

                            // Load indices
                            let index_data: Vec<u32> =
                                reader.read_indices().unwrap().into_u32().collect();

                            let vertex_buffer =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("Vertex Buffer"),
                                    contents: bytemuck::cast_slice(&vertex_data),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });

                            let index_buffer =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("Index Buffer"),
                                    contents: bytemuck::cast_slice(&index_data),
                                    usage: wgpu::BufferUsages::INDEX,
                                });

                            // Get the material index associated with this primitive
                            let material_index = primitive.material().index();

                            model::Mesh {
                                name: mesh_data.name().unwrap_or_default().to_string(),
                                vertex_buffer,
                                index_buffer,
                                num_elements: index_data.len() as u32,
                                material: material_index.unwrap(), // Set the material index here
                            }
                        })
                        .collect::<Vec<_>>() // Collect into a Vec<model::Mesh>
                })
                .collect(); // Collect into the outer Vec<model::Mesh>

            Ok(model::Model { meshes, materials })
        }
        _ => Err(anyhow::anyhow!("Unsupported file format")),
    }
}
