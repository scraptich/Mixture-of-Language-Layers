# 🌐 MoLL: Mixture-of-Languages-Layers

<div align="center">
  <p><b>An Efficient Sparse Expert Architecture for Machine Translation</b></p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/your-hf-profile"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange" alt="Hugging Face"></a>
  <!-- Uncomment when paper is published:
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
  -->
</div>

<br>

**MoLL** is a novel Encoder-Decoder Transformer architecture designed to maximize efficiency in multilingual machine translation. Unlike traditional Mixture-of-Experts (MoE) models, **MoLL binds each expert to a specific language**, enabling sequence-level routing, deterministic decoding, and a massive reduction in computational overhead while maintaining high translation quality.

## ✨ Key Innovations

*   **Language-Specific Experts:** Each expert specializes in a single language (e.g., Expert 0 for English, Expert 1 for Russian).
*   **Sequence-Level Routing $O(1)$:** The encoder routes the entire sequence (with batch averaging) before the layer loop begins. Standard MoEs route per-token, per-layer $O(L \cdot T)$.
*   **Free Decoder Routing:** The decoder selects experts deterministically based on the target language tag (e.g., `<2EN>`), requiring zero routing computations.
*   **Strict Top-1 Sparsity:** No shared experts. Exactly one specialized expert is selected per sequence, keeping active parameters at **~640M**.
*   **Modernized Base Architecture:** Built on an upgraded `moderBART` foundation replacing legacy components with **GQA, RoPE, Pre-RMSNorm, and SwiGLU**.
*   **Asymmetric Design & No Dropout:** A 12-layer encoder and 3-layer decoder for 2.1x faster generation speed. Dropout is completely removed to prevent self-attention gradient bias.

## 🏗️ Architecture Comparison

| Feature | Classical MoE | MoLL (ours) |
| :--- | :--- | :--- |
| **Routing Frequency** | Per-layer, per-token $O(L \cdot T)$ | **Encoder:** $O(1)$; **Decoder:** Free |
| **Decision Level** | Per-token routing | Sequence-level / explicit language tag |
| **Batch Averaging** | No | Encoder: Yes; Decoder: Per-sample |
| **Active Experts** | Top-2 + Shared | **Top-1** (no Shared expert) |
| **Gate Function** | Sigmoid / Linear | Softmax |
| **Number of Experts**| Many small ones | 23 (Separate for encoder & decoder) |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/moll.git
cd moll
pip install -r requirements.txt
