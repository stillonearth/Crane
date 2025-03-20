# Crane 𓅥

> Crane focusing on accelerate LLM inference speed with the power of kernels in candle framework, while reducing development overhead, make it portable and fast run model on both CPU and GPU.

**Crane** is  **C**andle-based **R**ust **A**ccelerate **N**eural **E**ngine. Inference popular models with Candle to get maximum speed on CPU and GPU.

Crane is a high level Rust framework inference popular models. The current model will support:

- Spark TTS;
- Basic LLM (Qwen2.5 etc);
- MLLM models such as (Qwen2.5 VL, Namo-R1 etc)


## Why Crane

The reason why Crane is very simple, running LLMs with pure pytorch is relative slow due to limited optimization on inference. Currently llama.cpp is a popular option, but llama.cpp is complicated on support new models and C++ based.

However, candle framework brings the gap between efficiency and simplicity. Without complicated C++ code, we can deploy new models more easily with Rust while still maintaining a fast speed.





## Updates

- **`2025.03.19`**: 🔥project initialized;