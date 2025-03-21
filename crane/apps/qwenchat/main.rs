// Qwen2.5 LLM Chat
// A simple design make you easily use Qwen model in Rust
// it supports streaming as well.
// For Multimodal usage, refer to Namo-R1

use anyhow::{Error as E, Result};
use clap::Parser;
use crane_core::{
    generation::{GenerationConfig, based::ModelForCausalLM, streamer::TextStreamer},
    models::{DType, Device, qwen25::Model as Qwen25Model},
};

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short('m'), long, default_value = "checkpoints/Qwen2.5-0.5B-Instruct")]
    model_path: String,
}

fn main() {
    let args = Args::parse();

    let dtype = DType::F16;
    let device = Device::Cpu;

    crane_core::utils::utils::print_candle_build_info();

    let mut model = Qwen25Model::new(&args.model_path, &device, &dtype).unwrap();

    let gen_config = GenerationConfig {
        max_new_tokens: 235,
        temperature: Some(0.67),
        top_p: Some(1.0),
        repetition_penalty: 1.1,
        repeat_last_n: 5,
        do_sample: false,
        pad_token_id: model.tokenizer.get_token("<|end_of_text|>"),
        eos_token_id: model.tokenizer.get_token("<|im_end|>"),
    };

    let prompt = "Who are you?";
    let input_ids = model.prepare_inputs(prompt).unwrap();

    // warmup
    let _ = model
        .generate(&input_ids, &GenerationConfig::with_max_tokens(20), None)
        .unwrap();

    let mut streamer = TextStreamer {
        tokenizer: model.tokenizer.tokenizer.clone(),
        buffer: String::new(),
    };
    let output_ids = model
        .generate(&input_ids, &gen_config, Some(&mut streamer))
        .map_err(|e| format!("Generation failed: {}", e))
        .unwrap();

    // decode output_ids
    let res = model.decode(&output_ids, false).unwrap();
    println!("Output: {}", res);
}
