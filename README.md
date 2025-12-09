# GPT (Generative Pre-trained Transformer) Architecture: Built From Scratch

This repository contains a foundational implementation of the **Generative Pre-trained Transformer (GPT)** model, built entirely **from scratch** using core Python libraries. The architecture is derived directly from the seminal paper, **"Attention Is All You Need"** (Vaswani et al., 2017).

The project aims to provide a deep, practical understanding of the underlying mathematical and algorithmic mechanisms of modern Large Language Models (LLMs) by avoiding high-level deep learning frameworks.

---

## 🎯 Architecture and Core Implementation

The model adopts the characteristic **Decoder-only** structure of the GPT framework, which is optimized for **autoregressive generation**—predicting the next token in a sequence based on the preceding context.

### Key Components Coded from the Transformer Paper:

The project meticulously implements the core mechanisms detailed in the paper, focusing on mathematical fidelity:

1.  **Masked Multi-Head Self-Attention:**
    * Crucial for language modeling, this mechanism ensures that during training and generation, a token can only attend to tokens that **precede it** in the sequence (the historical context).
    * **Statistical Expression:** The standard Scaled Dot-Product Attention formula is used, incorporating an **upper-triangular mask ($M$)** to block future information: $Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$. 

2.  **Positional Encoding:**
    * Given the architecture's parallel processing nature, sinusoidal (or learned) vectors are added to the input embeddings to inject information about the **relative position** of each token in the sequence.

3.  **Normalization and Feed-Forward Networks (FFN):**
    * **Layer Normalization** is applied immediately before each sub-layer (Attention and FFN) output, followed by **Residual Connections** for stable training dynamics.
    * The **Position-wise Feed-Forward Networks** (two linear layers) are integrated after each attention block to introduce non-linearity and positional processing.

---

## ✨ Unique Feature: Input-Free Logical Generation

A key feature developed in this repository is an **autonomous generation loop** that bypasses traditional conditional prompting.

* **Mechanism:** Using the model's successfully trained weights, the mechanism initiates sequence generation from a single **seed token** (e.g., `[BOS]` or a random word) and proceeds to generate long-form, **logically coherent sequences without external input**.
* **Analytical Value:** This demonstrates the model's ability to intrinsically capture the **statistical distribution** and syntactic structure of the language embedded within its latent space, unguided by a specific prompt context.

---

## 📊 Training and Licensing

### Statistical Insights:

The model has been successfully developed and trained on a designated language corpus. Key architectural parameters and statistical indicators include:

* **Number of Layers ($N_{layers}$):** $L$
* **Model Dimension ($d_{model}$):** $D$ (The dimensionality of the vector space)
* **Number of Attention Heads ($N_{heads}$):** $H$
* **Convergence:** Training was executed until the **Loss** metric stabilized and converged to a low minimum value (e.g., $L_{final} \approx 0.985$).
* **Model Capacity:** The total number of trainable parameters ($P$) is defined by the chosen layer dimensions.

### License

Due to the fundamental nature of the implementation and the original code contributions based on the paper's principles, a specific license (e.g., the **MIT License**) will be added to the repository.

---

## 📜 References

The foundational academic work for this project:

* **Attention Is All You Need** (Vaswani et al., 2017)
    * [https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
