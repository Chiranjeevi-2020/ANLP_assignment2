# 🎓 ANLP Zero-to-Hero Viva Survival Guide
**Author:** Chiranjeevi Kumaravadivel | **Roll Number:** `2023122010`  
**Course:** Advanced Natural Language Processing (ANLP, Monsoon 2026) | **IIIT Hyderabad**

> [!IMPORTANT]
> **Don't Panic!** You do not need to be a math genius or have a master's in NLP to ace this viva. Language models are actually built on very simple, intuitive concepts. Read this guide carefully. It explains everything from absolute ground zero, gives real-world analogies, explains the code step-by-step, and provides **word-for-word scripts** for the exact questions your TA will ask.

---

## Table of Contents
1. [Level 0: NLP Foundations from Scratch (Zero Background Required)](#level-0-nlp-foundations-from-scratch)
2. [Level 1: Part 1 — Mixture of Experts (MoE) Explained Simply](#level-1-part-1--mixture-of-experts-moe)
3. [Level 2: Part 2 — The 5 Custom Optimizers Explained Simply](#level-2-part-2--the-5-custom-optimizers)
4. [Level 3: Part 3 — The 4 Decoding Strategies Explained Simply](#level-3-part-3--the-4-decoding-strategies)
5. [Level 4: The 4 Core Evaluation Metrics](#level-4-the-4-core-evaluation-metrics)
6. [Level 5: Codebase Architecture & File Execution Flow](#level-5-codebase-architecture--file-execution-flow)
7. [Level 6: Word-for-Word Scripts for Top 10 TA Questions](#level-6-word-for-word-scripts-for-top-10-ta-questions)
8. [Level 7: The "Open Report and Explain" Roadmap](#level-7-the-open-report-and-explain-roadmap)

---

# Level 0: NLP Foundations from Scratch

### 1. What is Natural Language Processing (NLP)?
Computers only understand numbers ($0$s and $1$s). Humans speak in words, sentences, and paragraphs. **NLP** is the science of converting human text into numbers, processing those numbers through neural networks, and converting the numbers back into human words.

### 2. What is an Autoregressive Language Model?
All modern large language models (like GPT-4, LLaMA, Pythia, and our model) are **Autoregressive Language Models**.
- That sounds fancy, but all it means is: **Next-Word Predictor**.
- Given the prompt: *"The student went to the library to study for the"* $\to$ the model predicts the single most probable next word: *"exam"*.
- It then appends *"exam"* to the input and predicts the *next* word, one token at a time. That is called **Autoregressive Generation**.

### 3. What is a Token and a Tokenizer?
- A computer cannot read the letters `'c'`, `'a'`, `'t'`.
- A **Tokenizer** is a dictionary that slices text into pieces ("tokens") and assigns each piece an integer ID.
  - Example: `"cat"` $\to$ ID `4512`. `"eating"` $\to$ ID `892`.
- **Why Byte-Level BPE (Byte-Pair Encoding)?**
  - In our assignment, we translate **Japanese** (which uses Kanji symbols like `東京`) and **Vietnamese** (which uses diacritics like `Tiếng Việt`).
  - If you use a simple word tokenizer, every unfamiliar Japanese symbol becomes an error called `<UNK>` (Unknown / Out-of-Vocabulary).
  - **Byte-Level BPE** breaks any text down to raw computer bytes (0 to 255). Because *every character in every language on Earth* can be represented in UTF-8 bytes, Byte-Level BPE has **0% Out-of-Vocabulary (OOV) errors**! Our vocabulary size is fixed at $16,384$.

### 4. What is an Embedding?
- An integer ID like `4512` has no meaning to a math equation.
- An **Embedding Layer** is a giant lookup matrix. If our model has $D_{\text{model}} = 256$, it converts the single number `4512` into a vector of **256 floating-point numbers** (coordinates in a 256-dimensional semantic space).
- Words with similar meanings (like `"king"` and `"queen"`, or `"<vi>"` and `"<ja>"`) get mapped to nearby points in this space.

### 5. What is Multi-Head Self-Attention?
- Imagine the sentence: *"The animal didn't cross the street because it was too tired."*
- What does the word *"it"* refer to? The animal, or the street?
- **Self-Attention** is the mechanism that allows every word in a sentence to look at ("attend to") every other word to update its meaning based on context.
  - The word *"it"* pays heavy attention to *"animal"*, so the network understands what *"it"* refers to.
  - **Multi-Head Attention** means we split the 256 numbers into 8 separate "heads" ($8 \times 32 = 256$). Each head focuses on a different relationship (one head tracks grammar, another tracks pronoun references, another tracks language identity).
- **The Causal Mask (Lower Triangular Mask):**
  - When training a model to predict the *next* word, we must prevent position 3 from "cheating" by looking at position 4. The **Causal Mask** puts $-\infty$ on all future positions so attention weights to future tokens become exactly $0$.

### 6. What is a Feed-Forward Network (FFN)?
- In every Transformer block, after the Attention layer, there is a **Feed-Forward Network (FFN)**.
- While Attention mixes information *across different tokens* in the sentence, the FFN processes *each token individually*.
- In modern LLMs, **over 66% of all model parameters live in the FFN layers!** The FFN acts like the model's factual memory and cognitive reasoning engine.

### 7. What is Softmax?
- At the very end of the model, we get a list of 16,384 raw, unnormalized numbers (called **logits**), one for each word in our vocabulary.
- **Softmax** is a mathematical formula that turns those raw numbers into probabilities (percentages between $0$ and $1$ that add up to exactly $1.0$ or $100\%$).
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

### 8. What is Cross-Entropy Loss and Backpropagation?
- If the true next word is `"exam"`, but the model gave `"exam"` only a $10\%$ probability, it made a mistake.
- **Cross-Entropy Loss** calculates how wrong the model is: $\mathcal{L} = -\log P(\text{correct token})$. If $P=1.0$, loss is $0$. If $P$ is tiny, loss is huge.
- **Backpropagation**: Calculates the derivative (gradient) of the loss with respect to every single weight in the network. Gradients tell us: *"If I increase this weight by a tiny amount, will the loss go up or down?"*
- **Optimizer**: Takes those gradients and nudges the weights in the direction that lowers the loss.

---

# Level 1: Part 1 — Mixture of Experts (MoE)

### The Core Motivation (Why MoE?)
In a standard "Dense" Transformer, every single token passes through the exact same giant FFN layer. As you make models bigger (e.g., 100 Billion parameters), computing the FFN for every word becomes impossibly slow and expensive.

**The Mixture of Experts (MoE) Solution:**
Instead of one massive FFN, split the FFN into $N=4$ smaller, independent specialist neural networks called **Experts**.
- For each token, a lightweight neural network called the **Router (Gating Network)** looks at the token and decides which 1 or 2 experts should process it.
- **Result:** You get the massive knowledge capacity of a huge model, but only pay the compute cost of running 1 or 2 small experts per token!

```
                    [ Input Token: "Tokyo" ]
                               |
                        [ Router (Wg) ]
                       /       |       \
               Expert 1     Expert 2   Expert 3   Expert 4
               (English)   (Japanese) (General)  (Grammar)
                               |
                      [ Selected: Expert 2 ]
                               |
                         [ Output Token ]
```

### The 5 Architectural Ablations We Implemented:
The assignment required testing 5 distinct FFN designs to see how routing choices affect translation quality:
1. **Dense Baseline (C1):** 1 standard FFN with hidden dimension $D_{\text{ff}} = 1024$. (No routing).
2. **Top-1 MoE (C2):** 4 experts (each $d_e = 256$). The router picks ONLY the single highest-scoring expert ($k=1$).
3. **Top-2 MoE (C3):** 4 experts (each $d_e = 256$). The router picks the top 2 experts ($k=2$) and blends their outputs using normalized softmax gating weights.
4. **Shared + Routed MoE (C4):** 1 expert is designated as "Shared" and processes EVERY single token (learning common English grammar), while 1 additional expert is dynamically chosen from the remaining 3 routed experts.
5. **Matched-Active MoE (C5):** In C2--C4, because only 1 or 2 small experts are activated, the model does fewer FLOPs (calculations) than the Dense baseline. In C5, we double the expert width to $d_e = 512$ with $k=2$, so the **active parameters per token match the Dense baseline exactly** (~525k parameters per layer).

### The "Expert Collapse" Trap & Switch Auxiliary Loss ($\alpha = 0.01$)
> **Crucial Viva Concept — The TA WILL ask this!**

- **The Problem:** When training starts, weights are randomly initialized. By sheer random chance, Expert 1 might give a slightly better answer for the first few words. The router notices this and starts sending *all* words to Expert 1. Expert 1 gets trained, while Experts 2, 3, and 4 get starved and receive zero gradients! This catastrophic failure is called **Expert Collapse** or **Routing Imbalance**.
- **The Solution: Switch Transformers Auxiliary Loss:**
  We add a load-balancing penalty to the loss function:
  $$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$$
  - $\alpha = 0.01$ (loss weight).
  - $N = 4$ (number of experts).
  - $f_i$ is the actual fraction of tokens sent to expert $i$ in this batch (e.g., $0.25$ if perfectly balanced).
  - $P_i$ is the average softmax probability assigned to expert $i$ by the router.
- **Why is it differentiable?**
  $f_i$ is a hard count (non-differentiable), so PyTorch treats it as a constant (`f_i.detach()`). But $P_i$ is calculated from the router's smooth softmax! Therefore, backpropagation can compute $\frac{\partial \mathcal{L}_{\text{aux}}}{\partial W_g}$ and update the router weights to push routing toward a uniform distribution ($f_i = P_i = 1/N$) **without interfering with the language translation gradients**.

### What Our Language Heatmap Proved:
In [`assets/part1/expert_heatmap_matched_active.png`](file:///c:/Users/chira/Downloads/ANLP_assignment2/assets/part1/expert_heatmap_matched_active.png):
- In **Layer 1**, Vietnamese tokens (`<vi>`) were routed almost exclusively to **Expert 0**, while Japanese tokens (`<ja>`) were routed to **Expert 2**!
- This proves that **language specialization naturally emerged**: the model learned on its own to dedicate specific experts to specific linguistic grammars!

---

# Level 2: Part 2 — The 5 Custom Optimizers

The assignment required implementing 5 advanced pretraining optimizers **strictly from scratch** by subclassing `torch.optim.Optimizer`, without calling PyTorch's built-in routines.

### Why Do We Need Different Optimizers?
Standard Stochastic Gradient Descent (SGD) is like a blind hiker taking steps down a foggy mountain. If the valley is narrow and steep (a ravine), SGD bounces wildly between the walls and makes no progress forward. Advanced optimizers use **momentum** (rolling like a heavy ball) and **adaptive scaling** (taking smaller steps on steep slopes and larger steps on flat plains).

| Optimizer | Everyday Analogy | The Special Mathematical Superpower | Memory / Param |
| :--- | :--- | :--- | :--- |
| **CustomAdamW** | A runner with momentum who also wears weighted shoes | **Decoupled Weight Decay:** Separates weight shrinkage $\lambda \theta$ from adaptive variance $\sqrt{v_t}$. Weights decay at a steady rate regardless of gradient scale. | 8 bytes ($m, v$) |
| **CustomAdan** | A car with active suspension and shock absorbers | **Gradient Difference Tracking:** Tracks $v_k \approx (g_k - g_{k-1})$. Anticipates upcoming slope changes and dampens oscillations. | 12 bytes ($m, v, n$) |
| **CustomLion** | A hiker who only looks at the signpost (+1 or -1) | **Sign Momentum:** Replaces floating-point variance with $\text{sign}(c_t)$. Takes uniform steps. Cuts optimizer memory in half! | **4 bytes** ($m$ only) |
| **CustomMuon** | A master pilot who rotates the plane in 3D without stretching | **Matrix Polar Decomposition:** Updates 2D weight matrices via quintic Newton-Schulz polynomial iterations to keep updates orthogonal on the Stiefel manifold. | 8 bytes (2D only) |
| **CustomSophia** | A smart driver who checks road curvature before accelerating | **Lightweight Diagonal Hessian:** Uses Hutchinson curvature estimation to detect sharp bends and clips step sizes to prevent exploding gradients. | 8 bytes ($m, h$) |

### Figure 2 Replication (Our Empirical Findings):
Replicating *Wen et al. (2025)* on `browndw/human-ai-parallel-corpus`:
- **CustomMuon converged the fastest** (achieving validation loss $3.89$ and BLEU $10.85$, $1.35\times$ faster convergence than AdamW).
- **CustomLion achieved 50% memory reduction** with competitive loss ($4.18$), making it the best choice for memory-constrained edge hardware.
- Curves and metrics are saved in [`assets/part2/optimizer_dynamics.png`](file:///c:/Users/chira/Downloads/ANLP_assignment2/assets/part2/optimizer_dynamics.png).

---

# Level 3: Part 3 — The 4 Decoding Strategies

### The Strict Assignment Rule:
You were **strictly forbidden** from calling Hugging Face's high-level `model.generate()`. We had to run `model.forward()` manually token-by-token in a Python loop.

### How Does a Model Choose the Next Word?
At each step, `model.forward()` outputs logits for all $V=50,257$ vocabulary words. The decoding strategy chooses which word ID to pick:

```
Step t: Prompt -> [ Pythia-160M forward() ] -> Logits over Vocab -> [ Decoding Strategy ] -> Pick 1 Token -> Append to Prompt -> Repeat
```

### 1. Greedy Decoding
- **Logic:** Always pick the token with the highest probability: $\hat{w} = \text{argmax}_w P(w \mid x_{<t})$.
- **Pros:** Fast and deterministic.
- **Cons:** Extremely repetitive and dull. Gets trapped in infinite loops (e.g., *"The park was fun and the park was fun and..."*).

### 2. Top-k Sampling ($k=40$, Temperature $T=0.8$)
- **Logic:**
  1. Divide logits by temperature $T$: lower $T$ makes confident words more confident; higher $T$ makes text wild and creative.
  2. Sort vocabulary by probability.
  3. Keep only the top $k=40$ words; set probabilities of all other 50,000+ words to $-\infty$.
  4. Renormalize and roll a weighted dice (`torch.multinomial`).
- **Limitation:** Fixed $k$ is dumb. If the sentence is *"The Eiffel Tower is in [Paris]"*, there is only 1 sensible word. Top-$k$ forces the model to choose from 40 words, risking weird completions.

### 3. Top-p (Nucleus) Sampling ($p=0.9$, $T=0.8$)
- **Logic:**
  - Instead of fixing the *number* of words, fix the **cumulative probability mass** to $p=0.90$ (90%).
  - Sort words descending by probability. Add words to the pool until their sum reaches $90\%$. Discard the tail.
- **Why it's superior:** The pool expands and contracts dynamically!
  - When confident: pool has only 1 or 2 words.
  - When ambiguous: pool expands to 100 words.
  - Generates the most human-like, creative stories.

### 4. Beam Search ($B \in \{1, 2, 4\}$)
- **Logic:** Instead of greedily picking one word at a time, beam search acts like a chess player thinking 3 moves ahead.
- At each step, it keeps the top $B$ most promising **entire partial sentence hypotheses** (called "beams").
- **Length Penalty ($\alpha = 0.6$):**
  - Probabilities are decimals ($0.1 \times 0.1 \times 0.1 = 0.001$). Every time you add a word, the total probability score drops.
  - Without length normalization, beam search will always pick short, 2-word sentences to avoid the score penalty!
  - We divide the score by $\left(\frac{5 + L}{6}\right)^\alpha$, ensuring long, detailed story endings aren't penalized.
- **Latency Scaling:**
  - Running $B=4$ requires tracking 4 simultaneous hypotheses, leading to a strict $\mathcal{O}(B)$ latency penalty:
  - $B=1$: 45 ms/sample $\to$ $B=2$: 89 ms $\to$ $B=4$: 176 ms ($3.85\times$ slower!).

---

# Level 4: The 4 Core Evaluation Metrics

| Metric | What It Stands For | How It Works | Good Score | Where We Used It |
| :--- | :--- | :--- | :--- | :--- |
| **PPL (Perplexity)** | Model "Surprise" | $\text{PPL} = \exp(\text{Cross-Entropy Loss})$. If a model must choose between 10 equally likely words, PPL is 10. | Lower is better (15--25 is great) | Part 3 generation quality |
| **BLEU** | Bilingual Evaluation Understudy | Measures $n$-gram overlap ($1, 2, 3, 4$-word matches) between model output and ground truth references, with a brevity penalty. | Higher is better (12--16 for small models) | Part 1 translation & Part 2 pretraining |
| **Accuracy** | Token Match Percentage | Fraction of generated tokens that match the exact ground-truth token ID at that position. | Higher is better (~45%) | Part 3 narrative evaluation |
| **Latency** | Runtime in milliseconds | Wall-clock execution time per generated sample. | Lower is better (45ms vs 176ms) | Part 3 beam search scaling |

---

# Level 5: Codebase Architecture & File Execution Flow

If your TA asks: *"Show me your code and walk me through which file does what,"* show them this clear roadmap:

```
ANLP_assignment2/
│
├── main.py                     <-- CENTRAL CLI DISPATCHER
│   ├── verify                  -> Runs pytest test-suite & tensor checks
│   ├── part1                   -> Calls src/part1/train.py
│   ├── part2                   -> Calls src/part2/train.py
│   └── part3                   -> Calls src/part3/evaluate.py
│
├── src/
│   ├── utils.py                <-- Shared tools (count_parameters, SacreBLEU, WandbLogger)
│   │
│   ├── part1/                  <-- PART 1: MIXTURE OF EXPERTS
│   │   ├── moe.py              -> Defines Router, Expert, MoELayer, and Switch aux loss
│   │   ├── model.py            -> Causal Transformer with MultiHeadAttention & FFN
│   │   ├── dataset.py          -> Byte-Level BPE tokenizer & en-vi-ja dataloader
│   │   └── train.py            -> Training loop, tracking expert heatmaps & BLEU
│   │
│   ├── part2/                  <-- PART 2: 5 SCRATCH OPTIMIZERS
│   │   ├── optimizers.py       -> CustomAdamW, CustomAdan, CustomLion, CustomMuon, CustomSophia
│   │   ├── dataset.py          -> Pretraining stream for browndw parallel corpus
│   │   └── train.py            -> Evaluates convergence at 0.1x intervals (Figure 2)
│   │
│   └── part3/                  <-- PART 3: DECODING STRATEGIES
│       ├── decoding.py         -> Greedy, Top-k, Top-p, Beam Search (ONLY model.forward())
│       └── evaluate.py         -> Evaluates 1,000 ROCStories on EleutherAI/Pythia-160M
│
└── assets/                     <-- REAL RESULTS & FIGURES (WandB & Colab GPU logs)
    ├── part1/                  -> Language specialization heatmaps & summary JSONs
    ├── part2/                  -> Figure 2 replication dynamics plot & optimizer JSONs
    └── part3/                  -> raw_generations.json (1,000 stories) & latency curves
```

### The Exact Call Chain for Each Part:
1. **Part 1 Call Chain:**  
   `main.py part1` $\to$ `src/part1/train.py` $\to$ loads data via `dataset.py` $\to$ builds `TransformerLanguageModel` in `model.py` $\to$ inserts `MoELayer` from `moe.py` $\to$ computes task loss + Switch auxiliary loss $\to$ logs to `WandbLogger` in `utils.py`.
2. **Part 2 Call Chain:**  
   `main.py part2` $\to$ `src/part2/train.py` $\to$ initializes custom optimizer from `optimizers.py` $\to$ runs token step loop on `browndw` corpus $\to$ evaluates validation loss & test BLEU at $0.1\times$ intervals $\to$ plots Figure 2 dynamics curve.
3. **Part 3 Call Chain:**  
   `main.py part3` $\to$ `src/part3/evaluate.py` $\to$ downloads `hamishivi/ROCStories` and `EleutherAI/Pythia-160M` $\to$ passes 4 prompt sentences to `decoding.py` $\to$ `greedy_decode`, `top_k_decode`, `top_p_decode`, or `beam_search_decode` loops `model(input_ids).logits` $\to$ computes BLEU, perplexity, and latency $\to$ writes `raw_generations.json`.

---

# Level 6: Word-for-Word Scripts for Top 10 TA Questions

### Q1: "Explain how your MoE layer routes a token."
> **Say this:**  
> *"Inside [`src/part1/moe.py`](file:///c:/Users/chira/Downloads/ANLP_assignment2/src/part1/moe.py), each token's hidden state $x$ of dimension $D_{\text{model}}=256$ is multiplied by a linear router matrix $W_g$ ($256 \times 4$) to produce 4 expert logits. We apply a softmax over these logits to compute routing probabilities $P_i$. Depending on the ablation, we select either the top-1 or top-2 highest scoring experts. The token is passed through the chosen expert FFNs, their outputs are multiplied by the router weights, and summed. For load balancing, we compute the Switch Transformers auxiliary loss $\alpha N \sum f_i P_i$ with $\alpha=0.01$ to prevent expert collapse."*

---

### Q2: "Why didn't you just train the router with standard cross-entropy loss?"
> **Say this:**  
> *"Cross-entropy loss only optimizes for predicting the correct target word; it does not care about how many tokens each expert receives. Because of random initialization, one expert initially gets slightly higher gradients, creating a vicious cycle where the router routes 100% of tokens to that one expert while the other 3 experts starve and remain untrained. Switch auxiliary loss acts as an entropy regularizer that penalizes non-uniform dispatch frequencies without disrupting the primary language gradients."*

---

### Q3: "How did you ensure parameter count parity between Dense and MoE?"
> **Say this:**  
> *"In a standard dense Transformer, the FFN intermediate dimension is $D_{\text{ff}} = 1024$, which has $525,568$ parameters per layer. For our C1--C4 MoE ablations, we set each expert's width to $d_{\text{expert}} = D_{\text{ff}} / 4 = 256$. Four experts collectively possess $4 \times 131,584 = 526,336$ parameters. Adding the router's $256 \times 4 = 1,024$ weights, the total parameter count differs by only **+0.12%**, ensuring our comparison isolates routing architecture rather than model capacity. For C5 (Matched-Active), we set $d_{\text{expert}}=512$ with $k=2$ active experts, matching active parameters within **+0.08%**."*

---

### Q4: "Why did you implement optimizers from scratch? Explain how Muon works."
> **Say this:**  
> *"Assignment Part 2 explicitly forbade PyTorch's predefined optimizers to test first-principles understanding. We subclassed `torch.optim.Optimizer` directly. Muon is a second-order matrix optimizer designed for 2D weight tensors. Standard Adam scales each coordinate independently, which can distort the geometry of weight matrices. Muon applies polar decomposition using a quintic Newton-Schulz polynomial iteration to orthogonalize updates on the Stiefel manifold. In our Chinchilla pretraining benchmark, Muon achieved the fastest loss reduction, beating AdamW by $1.35\times$ relative convergence rate."*

---

### Q5: "How does Lion achieve a 50% memory saving compared to AdamW?"
> **Say this:**  
> *"AdamW maintains two state tensors per parameter: the first moment (momentum $m_t$) and the second moment (variance $v_t$), requiring $8$ bytes per parameter. Lion only tracks the momentum buffer $m_t$ and computes updates using $\text{sign}(c_t)$. Because it completely eliminates the second-moment variance buffer $v_t$, it requires only **4 bytes per parameter**—cutting optimizer memory in half with zero loss in training stability."*

---

### Q6: "Why can't you use `model.generate()` for Part 3?"
> **Say this:**  
> *"The assignment instructions explicitly prohibited `model.generate()` to ensure we implemented the core autoregressive decoding algorithms ourselves from tensor fundamentals. In [`src/part3/decoding.py`](file:///c:/Users/chira/Downloads/ANLP_assignment2/src/part3/decoding.py), we pass tokens into `model.forward()`, slice the logits at the last position `logits[:, -1, :]`, apply temperature scaling, perform top-k or top-p filtering, sample the next token ID, concatenate it to the input sequence, and repeat for 32 new tokens."*

---

### Q7: "What is the difference between Top-k and Top-p sampling?"
> **Say this:**  
> *"Top-k keeps a fixed number of candidate tokens (e.g., $k=40$), regardless of the probability distribution. If the model is extremely certain about the next word, top-k still forces 39 low-probability tokens into the sampling pool, risking hallucination. Top-p (nucleus sampling) dynamically sets the threshold based on cumulative probability mass (e.g., $p=0.90$). If the distribution is sharp, the nucleus pool shrinks to 1 or 2 tokens; if the context is open-ended, the pool expands to 50+ tokens. This makes Top-p much more fluent and natural."*

---

### Q8: "Why do we need a length penalty in Beam Search?"
> **Say this:**  
> *"In beam search, sequence score is the cumulative sum of log probabilities. Because log probabilities of tokens are negative numbers, longer sentences naturally accumulate more negative numbers. Without length normalization, beam search always prefers short 2- or 3-word endings. We apply length normalization by dividing the score by $\left(\frac{5 + L}{6}\right)^\alpha$ with $\alpha = 0.6$, penalizing unnaturally brief completions."*

---

### Q9: "Where are your model checkpoints and why aren't multi-gigabyte `.pt` files in Git?"
> **Say this:**  
> *"All training telemetry, step metrics, loss curves, and evaluation checkpoints are permanently logged on my public **Weights & Biases dashboard** and hosted on **Hugging Face Model Hub (`Chiru20/ANLP_assignment2`)**. Because the institute submission repository on `code.iiit.ac.in` enforced a strict **5 MB quota**, multi-hundred-megabyte `.pt` binary checkpoints were excluded from Git tracking via `.gitignore`. All evaluation metrics, generation outputs (`assets/part3/raw_generations.json`), and high-resolution heatmaps are preserved in `assets/`, and the training scripts are fully seeded (`seed=42`) for exact reproducibility."*

---

### Q10: "If you had 10x more GPU compute, what would you improve?"
> **Say this:**  
> *"I would scale the MoE architecture to 16 or 32 experts with hierarchical routing, train on the full 500k triplet dataset for 10 epochs, and test FP8 mixed-precision matrix multiplication with CustomMuon to accelerate pretraining throughput."*

---

# Level 7: The "Open Report and Explain" Roadmap

If your TA says: *"Open your report PDF and walk me through it,"* turn to these pages with confidence:

1. **Page 1 (Problem Formulation & Datasets):**
   - Point to the 3 datasets: `belumind/en-vi-ja-curated-500k-triplets` (Part 1), `browndw/human-ai-parallel-corpus` (Part 2), and `hamishivi/ROCStories` (Part 3). Explain Byte-Level BPE tokenizer.
2. **Page 2 (Common Architecture & MoE Ablations):**
   - Point to the Causal Multi-Head Attention equations and the 5 FFN ablation variants (Dense, Top-1, Top-2, Shared+Routed, Matched-Active).
3. **Page 3 (Switch Auxiliary Loss & First-Order Optimizers):**
   - Point to Eq. (5) ($\mathcal{L}_{\text{aux}}$). Explain why $f_i$ is detached while $P_i$ is differentiable. Point to CustomAdamW decoupled weight decay and CustomLion sign momentum.
4. **Page 4 (Advanced Optimizers & Decoding Algorithms):**
   - Point to CustomMuon Newton-Schulz polynomial iteration Eq. (14) and the 4 decoding algorithms with length penalty Eq. (17).
5. **Page 5 (MoE Results & Bilingual Routing Heatmap):**
   - Show **Table 2** (Matched-Active achieved lowest loss of $2.94$ and highest BLEU of $13.30$).
   - Show **Figure 1 (Language Heatmap)**: Point out how Layer 1 Expert 0 specialized in Vietnamese while Expert 2 specialized in Japanese!
6. **Page 6 (Optimizer Convergence & Decoding Benchmark):**
   - Show **Figure 2**: Muon converging faster than AdamW on next-token pretraining.
   - Show **Table 4**: Beam search achieving lowest perplexity ($15.80$) and highest accuracy ($45.90\%$), but nucleus sampling ($p=0.9$) providing the best balance of speed and story fluency.
   - Show **Figure 3**: Linear latency scaling of beam search ($45\text{ms} \to 176\text{ms}$).
7. **Page 7 (Qualitative Narrative Generations):**
   - Walk through the Marcus basketball story sample, showing how greedy repeats itself while top-p and beam search create a coherent climax.
8. **Page 8 (Deliverables & Links):**
   - Show the public links: Hugging Face Hub (`Chiru20/ANLP_assignment2`), WandB dashboard, and GitHub repository.

---
*Good luck with your viva! You have the complete theoretical knowledge, empirical results, and code mastery to ace this evaluation.*
