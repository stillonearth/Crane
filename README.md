# Crane 🦩

> Crane focusing on accelerate LLM inference speed with the power of kernels in candle framework, while reducing development overhead, make it portable and fast run model on both CPU and GPU.

**Crane (🦩)** is  **C**andle-based **R**ust **A**ccelerate **N**eural **E**ngine. Inference popular models with Candle to get maximum speed on CPU and GPU.

Crane is a high level Rust framework inference popular models. The current model will support:

- Spark TTS;
- Basic LLM (Qwen2.5 etc);
- MLLM models such as (Qwen2.5 VL, Namo-R1 etc)

By using Crane, you can have a very ffffffast speed on your mac especially, which means, **you don't need llama.cpp, just Crane, serving your local models in a flash speed**. 🔥 Forgotten cubersome cpp, embrace Rust.


## Why Crane

The reason why Crane is very simple, running LLMs with pure pytorch is relative slow due to limited optimization on inference. Currently llama.cpp is a popular option, but llama.cpp is complicated on support new models and C++ based.

However, candle framework brings the gap between efficiency and simplicity. Without complicated C++ code, we can deploy new models more easily with Rust while still maintaining a fast speed.

## Updates

- **`2025.03.19`**: 🔥project initialized;

## Speed Compare

Here are some speedup compare between **Crane** can other framework.

f32:

| Model/Platform | mac M1 metal | mac M1 cpu | mac M4 metal | v100 GPU | pytorch |
| -------------- | ------------- | ---------- | ------------ | -------- | ------- |
| Qwen2.5-500M   | 17.5 t/s      | 14 t/s     | /            |          | 6.9 t/s |
| Qwen2.5-VL-3B  | /             | /          | /            |          |         |

f16:

| Model/Platform | mac M1 metal | mac M1 metal 16 | mac M4 metal 16 | pytorch |
| -------------- | ------------- | --------------- | --------------- | ------- |
| Qwen2.5-500M   | 17.5 t/s      | **35 t/s**     | /               | 6.9 t/s |
| Qwen2.5-VL-3B  | /             | /               | /               |         |


- *Crane* is blazing fast on macOS with metal, useful for you to run local models;
- int8 quantization still on the way, it's even faster!
