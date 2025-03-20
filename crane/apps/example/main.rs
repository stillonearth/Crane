use crane_core::models::{DType, Device};

fn main() {
    // runing example to verify model result correctness
    test_siglip2();
}

fn test_siglip2() {
    use crane_core::models::siglip2;

    let dtype = DType::F16;
    let device = Device::Cpu;

    let model =
        siglip2::Siglip2VisionModel::from_pretrained("checkpoints/siglip2", &device, &dtype)
            .unwrap();

    // tensor pixel_values, pixel_attention_mask, spatial_shapes;
    model.forward(pixel_values, pixel_attention_mask, spatial_shapes);
}
