#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use candle_transformers::models::qwen2_moe::{Config as ConfigMoe, Model as ModelMoe};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

pub enum Model {
    Base(ModelBase),
    Moe(ModelMoe),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Moe(ref mut m) => m.forward(xs, s),
            Self::Base(ref mut m) => m.forward(xs, s),
        }
    }
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

pub fn load_qwen25_model(model_path: &str) -> Model {
    let tokenizer_filename = std::path::PathBuf::from(model_path);

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg);

    let dtype = DType::F32;
    let device = Device::Cpu;

    let filenames = utils::get_safetensors_files(model_path).unwrap();

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device) };
    let config_file = format!("{}/config.json", model_path);
    // make sure config file exists
    let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();
    let model = Model::Base(ModelBase::new(&config, vb.unwrap()).expect(""));

    // model for foward
    model
}

fn test_main() -> Result<()> {
    // use tracing_chrome::ChromeLayerBuilder;
    // use tracing_subscriber::prelude::*;

    // let args = Args::parse();
    // let _guard = if args.tracing {
    //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    //     tracing_subscriber::registry().with(chrome_layer).init();
    //     Some(guard)
    // } else {
    //     None
    // };

    // println!(
    //     "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
    //     args.temperature.unwrap_or(0.),
    //     args.repeat_penalty,
    //     args.repeat_last_n
    // );

    // let start = std::time::Instant::now();
    // let api = Api::new()?;
    // let model_id = match args.model_id {
    //     Some(model_id) => model_id,
    //     None => {
    //         let (version, size) = match args.model {
    //             WhichModel::W2_0_5b => ("2", "0.5B"),
    //             WhichModel::W2_1_5b => ("2", "1.5B"),
    //             WhichModel::W2_7b => ("2", "7B"),
    //             WhichModel::W2_72b => ("2", "72B"),
    //             WhichModel::W0_5b => ("1.5", "0.5B"),
    //             WhichModel::W1_8b => ("1.5", "1.8B"),
    //             WhichModel::W4b => ("1.5", "4B"),
    //             WhichModel::W7b => ("1.5", "7B"),
    //             WhichModel::W14b => ("1.5", "14B"),
    //             WhichModel::W72b => ("1.5", "72B"),
    //             WhichModel::MoeA27b => ("1.5", "MoE-A2.7B"),
    //         };
    //         format!("Qwen/Qwen{version}-{size}")
    //     }
    // };
    // let repo = api.repo(Repo::with_revision(
    //     model_id,
    //     RepoType::Model,
    //     args.revision,
    // ));
    // let tokenizer_filename = match args.tokenizer_file {
    //     Some(file) => std::path::PathBuf::from(file),
    //     None => repo.get("tokenizer.json")?,
    // };
    // let filenames = match args.weight_files {
    //     Some(files) => files
    //         .split(',')
    //         .map(std::path::PathBuf::from)
    //         .collect::<Vec<_>>(),
    //     None => match args.model {
    //         WhichModel::W0_5b | WhichModel::W2_0_5b | WhichModel::W2_1_5b | WhichModel::W1_8b => {
    //             vec![repo.get("model.safetensors")?]
    //         }
    //         WhichModel::W4b
    //         | WhichModel::W7b
    //         | WhichModel::W2_7b
    //         | WhichModel::W14b
    //         | WhichModel::W72b
    //         | WhichModel::W2_72b
    //         | WhichModel::MoeA27b => {
    //             candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
    //         }
    //     },
    // };
    // println!("retrieved the files in {:?}", start.elapsed());
    // let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // let start = std::time::Instant::now();
    // let config_file = repo.get("config.json")?;
    // let device = candle_examples::device(args.cpu)?;
    // let dtype = if device.is_cuda() {
    //     DType::BF16
    // } else {
    //     DType::F32
    // };
    // let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    // let model = match args.model {
    //     WhichModel::MoeA27b => {
    //         let config: ConfigMoe = serde_json::from_slice(&std::fs::read(config_file)?)?;
    //         Model::Moe(ModelMoe::new(&config, vb)?)
    //     }
    //     _ => {
    //         let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;
    //         Model::Base(ModelBase::new(&config, vb)?)
    //     }
    // };

    // println!("loaded the model in {:?}", start.elapsed());

    // let mut pipeline = TextGeneration::new(
    //     model,
    //     tokenizer,
    //     args.seed,
    //     args.temperature,
    //     args.top_p,
    //     args.repeat_penalty,
    //     args.repeat_last_n,
    //     &device,
    // );
    // pipeline.run(&args.prompt, args.sample_len)?;

    Ok(())
}
