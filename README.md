# WGPU Fabstir Renderer

This Rust program renders 3D models (.obj or .gltf files) in real-time in a canvas using WebGPU.
It is designed to be compiled to WebAssembly and called from JavaScript.

Author: Jules Lai

The code and rendering logic is based on the tutorial
"Render Pipelines in wgpu and Rust" by Ryosuke (https://sotrh.github.io/learn-wgpu/).

## Getting Started

1. Clone this repo:
2. Compile the project:
   $env:RUSTFLAGS='--cfg=web_sys_unstable_apis'; cargo build --target wasm32-unknown-unknown --release
   wasm-bindgen target/wasm32-unknown-unknown/release/wgpu_fabstir_renderer.wasm --out-dir ./public --target web
