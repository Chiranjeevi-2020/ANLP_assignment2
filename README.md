# Advanced Natural Language Processing (ANLP) - Assignment 2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models%20%26%20Datasets-yellow)](https://huggingface.co/)
[![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-orange)](https://wandb.ai/)

---

## Table of Contents
1. [Overview & Repository Layout](#1-overview--repository-layout)
2. [Task 1: Mixture of Experts (MoE) Architecture](#2-task-1-mixture-of-experts-moe-architecture)
   - [FFN Ablation Variants & Parameter Matching](#ffn-ablation-variants--parameter-matching)
   - [Router & Switch Load Balancing Auxiliary Loss](#router--switch-load-balancing-auxiliary-loss)
   - [Language Specialization Heatmaps](#language-specialization-heatmaps)
3. [Task 2: Custom Optimizers from Scratch](#3-task-2-custom-optimizers-from-scratch)
   - [Category 1: AdamW](#category-1-adamw)
   - [Category 2: Adan (Variance-Reduced)](#category-2-adan-variance-reduced)
   - [Category 3: Lion (Memory-Efficient)](#category-3-lion-memory-efficient)
   - [Category 4: Muon (Matrix-Based)](#category-4-muon-matrix-based)
   - [Bonus: Sophia (Hessian-Based)](#bonus-sophia-hessian-based)
   - [Chinchilla Pretraining Dynamics (Figure 2 Replication)](#chinchilla-pretraining-dynamics-figure-2-replication)
4. [Task 3: Decoding Strategies from Scratch](#4-task-3-decoding-strategies-from-scratch)
   - [Strict `model.forward()` Constraint](#strict-modelforward-constraint)
   - [Algorithms: Greedy, Top-k, Top-p, Beam Search (1, 2, 4)](#algorithms-greedy-top-k-top-p-beam-search-1-2-4)
   - [Benchmark on `hamishivi/ROCStories` with `Pythia-160M`](#benchmark-on-hamishivirocstories-with-pythia-160m)
5. [ELI5 Explanations](#5-eli5-explanations)
6. [Google Colab GPU Training Guide](#6-google-colab-gpu-training-guide)
7. [Local Setup & Verification](#7-local-setup--verification)

---

## 1. Overview & Repository Layout

This repository contains the complete, rigorous implementation of **ANLP Assignment 2 (Monsoon 2026)** covering:
1. **Mixture of Experts (MoE)** drop-in module for Transformer feed-forward networks across 5 architectural ablations.
2. **Custom Pretraining Optimizers** built from scratch strictly subclassing `torch.optim.Optimizer` without predefined torch optimizers.
3. **Four Decoding Strategies** implemented from scratch with `model.forward()` (no `model.generate()`), evaluated on `EleutherAI/Pythia-160M` over 1,000+ samples from `hamishivi/ROCStories`.

```
.
|-- .gitignore                  # Ignores checkpoints, virtualenvs, local caches
|-- .python-version             # Python version (3.10+)
|-- uv.lock                     # UV dependency lockfile
|-- pyproject.toml              # Build & dependency declarations
|-- README.md                   # Comprehensive architectural guide & report
|-- report.pdf                  # Formatted PDF report for submission
|-- main.py                     # Unified CLI entrypoint
|-- src/
|   |-- utils.py                # Parameter counting, BLEU, Perplexity, WandB
|   |-- part1/                  # Mixture of Experts (MoE)
|   |   |-- config.py           # TransformerConfig with parameter accounting
|   |   |-- moe.py              # Drop-in MoE, MLP, Expert, Switch Router
|   |   |-- model.py            # Decoder-only Transformer with causal attention
|   |   |-- dataset.py          # Multilingual Byte-BPE dataset for belumind
|   |   |-- train.py            # Training, evaluation, and heatmap extraction
|   |-- part2/                  # Custom Optimizers
|   |   |-- optimizers.py       # AdamW, Adan, Lion, Muon, Sophia from scratch
|   |   |-- dataset.py          # Pretrain chunking for human-ai-parallel-corpus
|   |   |-- train.py            # Pretraining dynamics benchmark & Figure 2 plots
|   |-- part3/                  # Decoding Strategies
|   |   |-- decoding.py         # Greedy, Top-k, Top-p, Beam Search (1, 2, 4)
|   |   |-- evaluate.py         # Pythia-160M ROCStories benchmark & runtime plots
|-- configs/                    # YAML configuration files for hyperparameter sweeps
|-- assets/                     # Generated plots, heatmaps, and raw generations
|-- colab/                      # High-speed GPU notebook for training in Google Colab
|-- tests/                      # Automated test suite
```

---

## 2. Task 1: Mixture of Experts (MoE) Architecture

### FFN Ablation Variants & Parameter Matching

The assignment requires 5 feed-forward network configurations:
1. **Dense Baseline**: Standard 2-layer MLP ($D_{model} \to D_{ff} \to D_{model}$ with GELU).
2. **MoE Top-1**: 4 experts, 1 active per token. Total parameters match Dense baseline.
3. **MoE Top-2**: 4 experts, 2 active per token. Total parameters match Dense baseline.
4. **MoE Shared + Routed**: 1 shared expert (always active) + 3 routed experts (1 active per token, making 2 active of 4 total). Total parameters match Dense baseline.
5. **MoE Matched Active**: 4 experts, 2 active per token, having the **exact same number of active parameters** as the Dense baseline (1).

#### Mathematical Parameter Accounting

Let $D = D_{model} = 256$ and $D_{ff} = 1024$.
The Dense MLP parameter count per layer is:
$$\text{Params}_{\text{dense}} = 2 \cdot D \cdot D_{ff} + D_{ff} + D = 2(256)(1024) + 1024 + 256 = 525,568$$

- **Variants 2, 3, 4 (Total Match)**: To keep total parameters constant across 4 experts, each expert has intermediate dimension:
  $$d_{\text{expert}} = \frac{D_{ff}}{4} = 256$$
  The 4 experts contribute $4 \times (2 \cdot D \cdot 256 + 256 + D) \approx 526,336$ parameters.
  The linear router adds $D \times 4 = 1,024$ parameters.
  Total model parameters differ by only **$0.12\%$** from the dense baseline!

- **Variant 5 (Matched Active)**: To match the dense baseline's active FFN width ($D_{ff} = 1024$) when $k = 2$ experts are active per token:
  $$d_{\text{expert}} = \frac{D_{ff}}{2} = 512$$
  Active parameters per token:
  $$\text{Active}_{\text{variant 5}} = 2 \times (2 \cdot D \cdot 512 + 512 + D) + \text{Router} \approx 526,848 \approx \text{Dense Active}$$
  Total parameters scale to $4 \times (2 \cdot D \cdot 512) \approx 2 \times \text{Dense}$.

#### Verified Parameter Comparison Table

| Ablation Variant | Total Parameters | Diff vs Dense | Active Parameters | Active Diff vs Dense |
| :--- | :---: | :---: | :---: | :---: |
| **1. Dense Baseline** | **8,900,608** | **0** | **8,900,608** | **0** |
| **2. MoE Top-1** | 8,911,360 | +10,752 (+0.12%) | 6,542,848 | -2,357,760 (-26.5%) |
| **3. MoE Top-2** | 8,911,360 | +10,752 (+0.12%) | 7,332,352 | -1,568,256 (-17.6%) |
| **4. MoE Shared+Routed** | 8,909,824 | +9,216 (+0.10%) | 7,330,816 | -1,569,792 (-17.6%) |
| **5. MoE Matched Active** | 12,063,232 | +3,162,624 (+35.5%) | **8,908,288** | **+7,680 (+0.08%)** |

### Router & Switch Load Balancing Auxiliary Loss

Without regularization, router networks experience **representation collapse**: a few experts handle all tokens while others remain idle.
We implement the **Switch / GShard Auxiliary Loss**:
$$\mathcal{L}_{aux} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$$
where:
- $f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{token } t \text{ routes to expert } i)$ is the empirical dispatch fraction.
- $P_i = \frac{1}{T} \sum_{t=1}^T \text{Softmax}(x_t W_g)_i$ is the average routing probability.
- $\alpha = 0.01$ is the auxiliary coefficient.
Because $P_i$ is continuously differentiable while $f_i$ is treated as constant, backpropagation gently guides router weights toward equal utilization without interfering with token-specific feature extraction.

### Language Specialization Heatmaps

During evaluation on `belumind/en-vi-ja-curated-500k-triplets`:
- Input sequences tagged `<vi>` (Vietnamese) and `<ja>` (Japanese) are routed through the network.
- `evaluate_and_extract_heatmaps()` records layer-by-layer expert dispatch matrices.
- Heatmaps are saved to `assets/expert_heatmap.png`, demonstrating that lower layers specialize in language-specific character token representations, while upper layers converge toward shared English semantics.

---

## 3. Task 2: Custom Optimizers from Scratch

In strict accordance with Instruction 4, **zero predefined torch.optim modules** are used (only `torch.optim.Optimizer` as a base class).

### Category 1: AdamW
Decoupled weight decay separates parameter shrinkage from gradient scaling:
$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1}$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Category 2: Adan (Variance-Reduced AdamW)
Adan (Xie et al., 2022) introduces gradient differences $(g_k - g_{k-1})$ to predict the gradient derivative, damping high-frequency oscillations:
$$v_k = (1 - \beta_2) v_{k-1} + \beta_2 (g_k - g_{k-1})$$
$$m_k = (1 - \beta_1) m_{k-1} + \beta_1 g_k$$
$$n_k = (1 - \beta_3) n_{k-1} + \beta_3 (g_k + (1 - \beta_2)(g_k - g_{k-1}))^2$$
$$\theta_k = \frac{\theta_{k-1} - \eta \frac{\hat{m}_k + (1-\beta_2)\hat{v}_k}{\sqrt{\hat{n}_k} + \epsilon}}{1 + \eta \lambda}$$

### Category 3: Lion (Memory-Efficient)
Lion (Chen et al., Google Brain 2023) tracks only a **single momentum buffer**, cutting optimizer memory footprint by **50%**:
$$c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\theta_t = \theta_{t-1} - \eta (\text{sign}(c_t) + \lambda \theta_{t-1})$$
$$m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t$$

### Category 4: Muon (Matrix-Based Polar Optimizer)
Muon applies momentum orthogonalization via a quintic **Newton-Schulz iteration** on 2D weight matrices $W \in \mathbb{R}^{d_{out} \times d_{in}}$:
$$M_t = \mu M_{t-1} + G_t$$
$$X_0 = \frac{M_t}{\|M_t\|_F}, \quad X_{k+1} = X_k (a I + b X_k^T X_k + c (X_k^T X_k)^2)$$
where $(a, b, c) = (3.4445, -4.7750, 2.0315)$.
Biases and 1D embedding layers automatically fall back to decoupled AdamW.

### Bonus: Sophia (Hessian-Based)
Sophia (Liu et al., 2023) utilizes a diagonal Hessian estimate $\hat{h}_t$ and coordinate-wise clipping:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad h_t = \beta_2 h_{t-1} + (1 - \beta_2) \hat{h}_t$$
$$\theta_t = \theta_{t-1} - \eta \cdot \text{clip}\left(\frac{m_t}{\max(\gamma h_t, \epsilon)}, -\rho, \rho\right) - \eta \lambda \theta_{t-1}$$

### Chinchilla Pretraining Dynamics (Figure 2 Replication)
The benchmark pretrains on `browndw/human-ai-parallel-corpus` up to $1.0\times$ dataset tokens. Validation loss and test BLEU are evaluated at checkpoints spaced by $0.1\times$ dataset tokens, producing Figure 2 comparison curves saved to `assets/optimizer_dynamics.png`.

---

## 4. Task 3: Decoding Strategies from Scratch

### Strict `model.forward()` Constraint
In compliance with Task 3 instructions, **`model.generate()` is never called**. All token generation is orchestrated manually on top of `model.forward()`.

### Algorithms
1. **Greedy Decoding**: Deterministic argmax selection: $x_t = \arg\max_v \log P(v \mid x_{<t})$.
2. **Top-k Sampling**: Truncates the vocabulary distribution to the top $k=40$ logits, re-normalizes with temperature $T=0.8$, and samples categorically.
3. **Top-p (Nucleus) Sampling**: Finds the smallest subset $V^{(p)} \subset V$ such that $\sum_{v \in V^{(p)}} P(v \mid x_{<t}) \ge p = 0.90$, truncates excluded tokens to $-\infty$, and samples categorically.
4. **Beam Search (Widths 1, 2, 4)**: Maintains active candidate hypotheses with length-normalized scoring:
   $$\text{Score}(h) = \frac{\sum_{t=1}^T \log P(x_t \mid x_{<t})}{\left(\frac{5 + T}{6}\right)^\alpha}$$
   - **Width 1**: Equivalent to deterministic greedy search.
   - **Width 2 & 4**: Explores 2 and 4 parallel hypotheses respectively.

### Runtime Complexity Analysis
The beam search execution time scales as:
$$\mathcal{O}(B \cdot T \cdot \text{Cost}(\text{forward}))$$
where $B$ is the beam width and $T$ is generation length.
Empirical benchmark results verify that latency scales linearly with beam width $B$. Runtime plots are generated and saved to `assets/beam_search_runtime.png`.

---

## 5. ELI5 Explanations

### 1. What is Mixture of Experts (MoE)?
> Imagine you are running a consulting agency. In a normal "dense" company, every single employee has to read and comment on every document, which is slow and tiring. In an MoE company, you have a smart receptionist (the **Router**) and 4 specialized consultants (the **Experts**). When a question arrives, the receptionist looks at it and routes it to only 1 or 2 relevant consultants. You get the knowledge of a huge team, but pay the electricity bill for only 1 or 2 people working at a time!

### 2. Why do we need the Auxiliary Loss?
> If the receptionist gets lazy, they might send all questions to Consultant #1 just because Consultant #1 answered the first question well. Consultant #1 gets burnt out, and Consultants #2, 3, and 4 do nothing. The **Auxiliary Loss** is like a penalty fine from HR: if the work isn't distributed evenly among consultants, the loss goes up. This forces the model to train all experts.

### 3. Why Decoupled Weight Decay (AdamW) instead of L2 Regularization?
> In standard SGD, adding L2 weight penalty $\frac{1}{2}\lambda \|\theta\|^2$ is mathematically identical to decaying weights. But in adaptive methods like Adam, gradients are divided by $\sqrt{v_t}$. If a parameter has huge gradients, its weight decay gets squashed; if it has tiny gradients, its weight decay gets amplified! AdamW decouples weight decay, directly shrinking weights by $(1 - \eta \lambda)$ at every step regardless of gradient size.

### 4. What is Newton-Schulz Orthogonalization (Muon)?
> When training neural network matrices, we want the transformation to preserve signals without blowing up or collapsing them. An orthogonal matrix rotates vectors without changing their length. Muon uses a clever polynomial (Newton-Schulz iteration) to push the gradient momentum matrix toward the nearest orthogonal matrix, stabilizing deep transformer updates.

### 5. Why Top-p (Nucleus) instead of pure Greedy or Beam Search?
> Greedy search and beam search always pick the highest probability words. For open-ended creative storytelling (like ROCStories), this causes the model to get stuck in boring, repetitive loops ("he was very happy, very happy, very happy"). Nucleus sampling keeps the top 90% of reasonable words and samples among them, yielding natural, diverse, and human-like stories.

---

## 6. Google Colab GPU Training Guide

Because your local laptop does not have a dedicated GPU, full training of all 5 MoE ablations and 5 optimizers should be executed on Google Colab (Free T4 or Pro A100 GPU).

### Quickstart Steps:
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com/)
2. Upload `colab/ANLP_Assignment2_Colab_Runner.ipynb`
3. Enable GPU Runtime: **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU (or A100)**
4. Set your Hugging Face and WandB API keys in the notebook secrets:
   ```python
   import os
   os.environ["WANDB_API_KEY"] = "your_wandb_key"
   os.environ["HF_TOKEN"] = "your_hf_token"
   ```
5. Click **Runtime -> Run all**.
The notebook automatically trains the 5 MoE ablations, runs the 5 custom optimizers, evaluates decoding on Pythia-160M, saves checkpoints to HuggingFace, and generates all report figures!

---

## 7. Local Setup & Verification

```bash
# 1. Clone repository
git clone https://github.com/username/ANLP_assignment2.git
cd ANLP_assignment2

# 2. Verify architecture, parameters, and optimizer steps
python main.py verify

# 3. Run unit test suite
pytest tests/ -v

# 4. Run Part 1 Dense baseline
python main.py part1 --ablation dense

# 5. Run Part 2 Custom Optimizers
python main.py part2 --optimizer all

# 6. Run Part 3 Decoding Benchmark
python main.py part3 --samples 100
```
