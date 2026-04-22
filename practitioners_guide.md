# Hardware Notes

## Development Environment

All training and evaluation was performed on an **NVIDIA DGX Spark** with 128 GB unified CPU/GPU memory and a Grace Blackwell architecture.

### Software Stack

- **Base image:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **Key libraries:** Hugging Face Transformers, TRL (SFTTrainer), PEFT, flash-linear-attention
- **Containerized:** All training runs execute inside Docker containers with GPU passthrough

### Why bf16, Not QLoRA

The Qwen3.5 architecture uses hybrid attention that interleaves Gated DeltaNet layers (linear attention with a decay gate and delta rule) with conventional full softmax attention in a 3:1 ratio. The DeltaNet layers produce **NaN loss values under 4-bit NF4 quantization** — this is a documented limitation of the model family's hybrid linear attention implementation. bf16 LoRA is the working configuration.

This constraint also eliminates QLoRA as a memory-saving strategy, which directly affects the maximum trainable model size on a given hardware configuration.

## Memory Requirements

### Training

| Model | Approx. Memory (bf16 LoRA) | Fits DGX Spark (128 GB) |
|---|---|---|
| 0.8B | ~12 GB | Yes |
| 2B | ~20 GB | Yes |
| 4B | ~35 GB | Yes |
| 9B | ~65 GB | Yes |
| 27B | ~108 GB peak | No (see below) |

Models are processed sequentially — even the 0.8B requires most available memory during training due to optimizer states and activation caching.

### Inference (Quantized GGUF)

| Model | Approx. Memory (4-bit) | Minimum Hardware |
|---|---|---|
| 0.8B | ~2 GB | 4 GB RAM, phone/tablet |
| 2B | ~3 GB | 4 GB RAM |
| 4B | ~4 GB | 8 GB RAM, any modern laptop |
| 9B | ~8 GB | 16 GB RAM or 8 GB VRAM GPU |

## The 27B Training Attempt

A training run on Qwen3.5-27B was attempted to test whether the 9B model's behavioral patterns extend to larger scales.

**What happened:** The 27B model at bf16 precision requires approximately 54 GB for weights alone, which fits the DGX Spark's 128 GB unified memory in steady state. However, the Hugging Face Transformers library's mmap-based loading requires weights in both page cache and CUDA memory simultaneously, producing a **peak demand of approximately 108 GB** that exceeded available memory.

**Why QLoRA isn't a workaround:** As noted above, the Qwen3.5 hybrid attention layers produce NaN loss under 4-bit quantization, eliminating the standard memory-reduction strategy.

**What this means:** 9B is the maximum trainable model size for bf16 LoRA on a single DGX Spark. The 27B would require either dual-Spark NVLink configuration (~256 GB) or a cloud GPU instance with 160 GB+ memory (e.g., A100 80GB × 2 or H100 × 2).

**Note:** The 27B *does* run for inference on the DGX Spark. An adapter trained on more capable hardware could be distributed for use on the same device that cannot train it.

**Prediction:** Based on the observed cross-scale gradient, the 27B is expected to show lower entropy delta (more confident internalization), higher actionability (closing the gap with epistemic calibration), and potentially clean Level 4 capability on Culshaw's maturity model.

## Deployment Hardware

### Local (Ollama / llama.cpp)

The recommended deployment for the target audience uses [Ollama](https://ollama.ai), which manages model downloads and provides a local HTTP API. The fine-tuned model is distributed as a custom Modelfile packaging quantized GGUF weights with the inference-time system prompt. No programming knowledge required.

### Organization-Hosted Server

For teams, a single machine runs the model as a persistent API (vLLM or llama.cpp server mode). Client devices connect through a mesh VPN (e.g., Tailscale) — any device on the mesh accesses the model regardless of its own hardware capability. This reframes the access question from "does every staff member have a capable laptop" to "can the organization acquire or repurpose one capable machine."

## Reproducing This Pipeline on Other Hardware

The pipeline is designed for the DGX Spark but should run on any CUDA-capable GPU with sufficient memory:

- **48 GB+ VRAM** (e.g., A6000, A100 40GB): Can train 9B with gradient checkpointing
- **24 GB VRAM** (e.g., RTX 4090, A5000): Can train up to 4B
- **16 GB VRAM** (e.g., RTX 4080): Can train 2B and 0.8B
- **Cloud options:** Lambda Labs, RunPod, or Vast.ai with A100/H100 instances

Adjust `batch_size` and `grad_accum` in the training configuration if memory is tighter. The effective batch size of 8 should be maintained (e.g., batch_size=1, grad_accum=8 for constrained memory).

## References

The fine-tuning pipeline was developed following the [ARM PyTorch Fine-Tuning on Spark](https://learn.arm.com/learning-paths/laptops-and-desktops/pytorch-finetuning-on-spark/) learning path.

Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated delta networks: Improving Mamba2 with delta rule. *Proceedings of the Thirteenth International Conference on Learning Representations.* https://arxiv.org/abs/2412.06464
