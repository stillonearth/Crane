# Crane 🦩

> Crane focusing on accelerate LLM inference speed with the power of kernels in candle framework, while reducing development overhead, make it portable and fast run model on both CPU and GPU.

**Crane (🦩)** is  **C**andle-based **R**ust **A**ccelerate **N**eural **E**ngine. Inference popular models with Candle to get maximum speed on CPU and GPU.

Crane is a high level Rust framework inference popular models. The current model will support:

- [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
- [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- Basic LLM (Qwen2.5 etc);
- MLLM models such as (Qwen2.5 VL, Namo-R1 etc)

By using Crane, you can have a very ffffffast speed on your mac especially, which means, **you don't need llama.cpp, just Crane, serving your local models in a flash speed**. 🔥 Forgotten cubersome cpp, embrace Rust.

## Why Crane

The reason why Crane is very simple, running LLMs with pure pytorch is relative slow due to limited optimization on inference. Currently llama.cpp is a popular option, but llama.cpp is complicated on support new models and C++ based.

However, candle framework brings the gap between efficiency and simplicity. Without complicated C++ code, we can deploy new models more easily with Rust while still maintaining a fast speed.

## Updates

- **`2025.03.19`**: 🔥project initialized;


## Usage

To use `crane`, here are some notes:

- `crane-core`: All models comes into core, this is a lib;
- `crane`: All Apps (runnable AI pipelines, such as Qwen2-Chat, Spark-TTS, Qwen2.5-VL etc), you can build your apps inside it, each app is a binary for demonstration purpose;
- `crane-oai`: OpenAI API server serving various services in OpenAI format;

1. Make sure latest Rust were installed;
2. Build:

   ```
   cargo run --bin llmbench --release
   cargo run --bin qwenchat --release
   ```

That's it!

Now you can run LLM extremly fast (about 6x faster than vanilla transformers on M1)!


## Contribution

PR are welcomed right now! Since we need support a brand range of new models, but both Crane and HuggingFace's Candle is very limited model scope, so please join and help!

1. How to add a new model?

Generally speaking, you can reference to: `crane-core/src/models/siglip2.rs` for support new model, and all new added models should placed into `crane-core/src/models` and add `pub mod` in `crane-core/src/models/mod.rs` .

For me, the easiest way is to using Claude 3.7 to help write Rust conversion from pytorch code into Rust Candle code, and then manually fixing issues, once the float values of output are matched, the model can be ready to go.

2. How to support a new arch?

As all we know, a TTS model or any model based on LLM, it might consist of different modules, for example, in Spark-TTS, we will have a BiCodec Model before LLM, these module can be made into a separated module, and for Spark-TTS itself, we can gathering all module to inference it correctly.

One can reference to `crane-core/src/models/namo2.rs` for new arch add, which uses `Siglip2`, `mm_projector`, `Qwen2.5` to support a VL model.

## Speed

Here are some speedup compare between **Crane** can other framework.

f32:

| Model/Platform | mac M1 metal | mac M1 cpu | mac M4 metal | v100 GPU | pytorch |
| -------------- | ------------- | ---------- | ------------ | -------- | ------- |
| Qwen2.5-500M   | 17.5 t/s      | 14 t/s     | /            |          | 6.9 t/s |
| Qwen2.5-VL-3B  | /             | /          | /            |          |         |

f16:

| Model/Platform | mac M1 metal | mac M1 metal 16  | mac M4 metal 16 | pytorch |
| -------------- | ------------- | ---------------- | --------------- | ------- |
| Qwen2.5-500M   | 17.5 t/s      | **35 t/s** | /               | 6.9 t/s |
| Qwen2.5-VL-3B  | /             | /                | /               |         |

- *Crane* is blazing fast on macOS with metal, useful for you to run local models;
- int8 quantization still on the way, it's even faster!
