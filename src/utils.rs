use wasm_bindgen::prelude::*;

pub async fn download_bytes(url: &str) -> Result<Vec<u8>, anyhow::Error> {
    let mut opts = web_sys::RequestInit::new();
    opts.method("GET");

    let request = web_sys::Request::new_with_str_and_init(url, &opts)
        .map_err(|_| anyhow::anyhow!("Failed to create request"))?;

    let window = web_sys::window().unwrap();
    let promise = window.fetch_with_request(&request);

    let resp_value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| anyhow::anyhow!("Failed to fetch data"))?;

    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("Failed to convert response value"))?;

    let ab = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().unwrap())
        .await
        .map_err(|_| anyhow::anyhow!("Failed to get array buffer"))?;

    let js_array = js_sys::Uint8Array::new(&ab);
    let data: Vec<u8> = js_array.to_vec();

    Ok(data)
}
