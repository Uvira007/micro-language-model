# Transformer Deep Dive: Complete Mathematical Guide

**An in-depth exploration of encoder-decoder transformers with full mathematical details and architectural comparisons**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Multi-Head Attention: The Core](#3-multi-head-attention-the-core)
4. [Encoder Architecture](#4-encoder-architecture)
5. [Decoder Architecture](#5-decoder-architecture)
6. [Complete Forward Pass](#6-complete-forward-pass)
7. [Gradient Flow & Backpropagation](#7-gradient-flow--backpropagation)
8. [Comparison: Encoder-Decoder vs Decoder-Only](#8-comparison-encoder-decoder-vs-decoder-only)
9. [Modern Transformer Optimizations](#9-modern-transformer-optimizations)
10. [Implementation Guide](#10-implementation-guide)

---

## 1. Architecture Overview

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENCODER-DECODER TRANSFORMER                   │
└─────────────────────────────────────────────────────────────────────┘

SOURCE SEQUENCE (Question)                    TARGET SEQUENCE (Answer)
"What is machine learning?"                   "<s> Machine learning is..."
        ↓                                              ↓
   Tokenization                                   Tokenization
   [45, 12, 78, 92]                              [1, 23, 12, 56, ...]
        ↓                                              ↓
   Embedding Lookup                              Embedding Lookup
   (vocab_size → d_model)                       (vocab_size → d_model)
        ↓                                              ↓
   × √d_model (Scaling)                         × √d_model (Scaling)
        ↓                                              ↓
   + Positional Encoding                        + Positional Encoding
        ↓                                              ↓
┌────────────────────┐                         ┌────────────────────┐
│   ENCODER STACK    │                         │   DECODER STACK    │
│                    │                         │                    │
│  ┌──────────────┐  │                         │  ┌──────────────┐  │
│  │ Layer 1      │  │                         │  │ Layer 1      │  │
│  │ - Self-Attn  │  │                         │  │ - Masked SA  │  │
│  │ - Add&Norm   │  │        Memory           │  │ - Add&Norm   │  │
│  │ - FFN        │  │◄─────────────────────────┤ │ - Cross-Attn │  │
│  │ - Add&Norm   │  │  (B, src_len, d_model)  │  │ - Add&Norm   │  │
│  └──────────────┘  │                         │  │ - FFN        │  │
│         ↓          │                         │  │ - Add&Norm   │  │
│  ┌──────────────┐  │                         │  └──────────────┘  │
│  │ Layer 2      │  │                         │         ↓          │
│  │    ...       │  │                         │  ┌──────────────┐  │
│  └──────────────┘  │                         │  │ Layer 2      │  │
│         ↓          │                         │  │    ...       │  │
│  ┌──────────────┐  │                         │  └──────────────┘  │
│  │ Layer N      │  │                         │         ↓          │
│  └──────────────┘  │                         │  ┌──────────────┐  │
│         ↓          │                         │  │ Layer N      │  │
│  Final LayerNorm   │                         │  └──────────────┘  │
│                    │                         │         ↓          │
│  Memory Output     │                         │  Final LayerNorm   │
│  (contextualized   │                         │                    │
│   representation)  │                         │  Linear Projection │
└────────────────────┘                         │  (d_model→vocab)   │
                                               │         ↓          │
                                               │  Logits/Softmax    │
                                               │         ↓          │
                                               │  Next Token Probs  │
                                               └────────────────────┘
```

### 1.2 Key Differences from RNNs/LSTMs

| **Aspect** | **RNN/LSTM** | **Transformer** |
|------------|--------------|-----------------|
| **Processing** | Sequential (one token at a time) | Parallel (all tokens simultaneously) |
| **Long-range dependencies** | O(n) hops, gradient issues | O(1) hops via attention |
| **Training speed** | Slow (sequential bottleneck) | Fast (parallelizable) |
| **Memory** | Hidden state (fixed size) | Attention to full context |
| **Position** | Implicit (order of processing) | Explicit (positional encoding) |

---

## 2. Mathematical Foundations

### 2.1 Token Embeddings

**Purpose**: Map discrete token IDs to continuous vector space.

#### Embedding Matrix

```
E ∈ ℝ^(V × d_model)
```

Where:
- V = vocabulary size (e.g., 50,000)
- d_model = embedding dimension (e.g., 256, 512, 4096)

#### Embedding Lookup

For token ID `t_i`:

```
x_i = E[t_i] ∈ ℝ^d_model
```

#### Scaling Factor

```
x_i ← x_i · √d_model
```

**Why scaling?**

1. **Variance stability**: Random embedding initialization → values typically in range [-0.1, 0.1]
2. **Positional encoding range**: PE values in range [-1, 1]
3. **Balance**: Without scaling, PE would dominate the embedding
4. **Mathematical justification**:
   - If embedding values have variance σ²
   - After multiplying by √d_model, variance becomes d_model·σ²
   - This matches the scale of positional encodings

#### Code Reference

```python
# src/micro_lm/model/transformer.py:59-60
self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

# src/micro_lm/model/transformer.py:75
x = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
```

---

### 2.2 Positional Encoding

**Problem**: Attention is permutation-invariant → model cannot distinguish position.

**Solution**: Add position-dependent signals to embeddings.

#### Sinusoidal Positional Encoding

For position `pos` and dimension `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### Why This Formula?

**1. Different Frequencies**

Each dimension oscillates at a different frequency:
```
Dimension 0,1:   wavelength = 2π
Dimension 2,3:   wavelength ≈ 2π · 10000^(2/d_model)
...
Dimension d-2,d-1: wavelength = 2π · 10000
```

**2. Relative Position Learning**

Trigonometric identity:
```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
```

This means PE(pos + k) can be expressed as a linear function of PE(pos):
```
PE(pos + k) = A · PE(pos)
```

So the model can learn to attend by relative positions!

**3. Extrapolation**

Can handle sequences longer than training (to some extent).

#### Visual Representation

```
Position:  0    1    2    3    4    5    ...
           │    │    │    │    │    │
Dim 0-1:   ○────●────○────●────○────●    (fast oscillation)
Dim 2-3:   ○──────────●──────────○──────  (slower)
Dim 4-5:   ○────────────────────────────  (even slower)
...
Dim d-2,d-1: ○──────────────────────────  (very slow)

○ = sin minimum    ● = sin maximum
```

#### Matrix Form

```
PE ∈ ℝ^(max_len × d_model)

Position index: [0, 1, 2, ..., max_len-1]
Dimension index: [0, 1, 2, ..., d_model-1]

PE[pos, :] = unique vector for position pos
```

#### Code Reference

```python
# src/micro_lm/model/transformer.py:20-29
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                     (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)  # Even dims
pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims
```

#### Alternative: Learned Positional Embeddings

Some models use learned position embeddings instead:
```python
self.pos_embed = nn.Embedding(max_len, d_model)
```

**Trade-offs**:
- ✅ Learned: Can adapt to data, potentially more expressive
- ✅ Sinusoidal: Better extrapolation, no extra parameters
- Modern models (GPT-3, Claude): Often use learned or RoPE (Rotary Position Embeddings)

---

## 3. Multi-Head Attention: The Core

Multi-head attention is the **heart** of transformers. Let's build it from first principles.

### 3.1 Scaled Dot-Product Attention (Single Head)

#### Input

```
Q (Query):  ℝ^(L × d_k)  - "What am I looking for?"
K (Key):    ℝ^(L × d_k)  - "What do I contain?"
V (Value):  ℝ^(L × d_k)  - "What information do I hold?"
```

Where:
- L = sequence length
- d_k = key/query dimension (typically d_model / num_heads)

#### Forward Pass: Step by Step

**Step 1: Compute Attention Scores**

```
S = Q K^T / √d_k ∈ ℝ^(L × L)
```

Element-wise:
```
S_ij = (q_i · k_j) / √d_k = (Σ_d q_i[d] · k_j[d]) / √d_k
```

**Interpretation**: S_ij measures compatibility between query at position i and key at position j.

**Step 2: Apply Mask (Optional)**

```
S_masked[i,j] = {  S[i,j]    if mask[i,j] = True
                { -∞        if mask[i,j] = False
```

**Mask types**:
- **Padding mask**: Ignore padding tokens
- **Causal mask**: Prevent attending to future positions (decoder)

**Step 3: Softmax to Get Attention Weights**

```
A = softmax(S, dim=-1) ∈ ℝ^(L × L)
```

Row-wise softmax:
```
A_ij = exp(S_ij) / Σ_k exp(S_ik)
```

**Properties**:
- Each row sums to 1: Σ_j A_ij = 1
- A_ij ∈ [0, 1]: Probability that position i attends to position j
- High score → high attention weight

**Step 4: Weighted Sum of Values**

```
Output = A V ∈ ℝ^(L × d_k)
```

Element-wise:
```
output_i = Σ_j A_ij · v_j
```

**Interpretation**: Each output position is a weighted average of all value vectors.

#### Complete Formula

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

#### Why Scaling by √d_k?

**Problem without scaling**:

For d_k-dimensional vectors with unit variance components:
```
E[q · k] = Σ_d E[q_d · k_d] = d_k · E[q_d] · E[k_d] ≈ d_k
Var(q · k) ≈ d_k
```

As d_k grows, dot products become very large → softmax saturates:
```
softmax([100, 1, 2]) ≈ [1.0, 0.0, 0.0]  (all weight on one position)
```

**With scaling**:
```
(q · k) / √d_k  has variance ≈ 1
```

This keeps softmax in a reasonable range → better gradients!

### 3.2 Numerical Example

Let's trace through with tiny numbers:

**Setup**:
- Sequence length L = 3
- Dimension d_k = 4

**Queries**:
```
Q = [[0.5, 1.0, 0.2, 0.3],
     [0.2, 0.4, 0.8, 0.6],
     [0.7, 0.3, 0.5, 0.4]]
```

**Keys** (same as Q for self-attention):
```
K = [[0.5, 1.0, 0.2, 0.3],
     [0.2, 0.4, 0.8, 0.6],
     [0.7, 0.3, 0.5, 0.4]]
```

**Step 1: Scores**
```
Q·K^T = [[q_0·k_0, q_0·k_1, q_0·k_2],
         [q_1·k_0, q_1·k_1, q_1·k_2],
         [q_2·k_0, q_2·k_1, q_2·k_2]]

q_0·k_0 = 0.5*0.5 + 1.0*1.0 + 0.2*0.2 + 0.3*0.3 = 0.25 + 1.0 + 0.04 + 0.09 = 1.38
q_0·k_1 = 0.5*0.2 + 1.0*0.4 + 0.2*0.8 + 0.3*0.6 = 0.1 + 0.4 + 0.16 + 0.18 = 0.84
q_0·k_2 = 0.5*0.7 + 1.0*0.3 + 0.2*0.5 + 0.3*0.4 = 0.35 + 0.3 + 0.1 + 0.12 = 0.87

Scores (before scaling) = [[1.38, 0.84, 0.87],
                           [0.84, 1.20, 0.98],
                           [0.87, 0.98, 1.03]]
```

**Scaled scores** (divide by √4 = 2):
```
S = [[0.69, 0.42, 0.44],
     [0.42, 0.60, 0.49],
     [0.44, 0.49, 0.52]]
```

**Step 2: Softmax** (row-wise):
```
A = softmax(S, dim=-1)
  ≈ [[0.36, 0.31, 0.33],
     [0.30, 0.37, 0.33],
     [0.31, 0.33, 0.36]]
```

**Interpretation**:
- Position 0 attends 36% to itself, 31% to pos 1, 33% to pos 2
- Relatively uniform attention (scores are similar)

**Step 3: Weighted sum**

If Values:
```
V = [[1.0, 0.5, 0.3, 0.8],
     [0.4, 0.9, 0.6, 0.2],
     [0.7, 0.3, 0.8, 0.5]]
```

Output at position 0:
```
out_0 = 0.36*[1.0, 0.5, 0.3, 0.8] + 0.31*[0.4, 0.9, 0.6, 0.2] + 0.33*[0.7, 0.3, 0.8, 0.5]
      = [0.36, 0.18, 0.11, 0.29] + [0.12, 0.28, 0.19, 0.06] + [0.23, 0.10, 0.26, 0.17]
      = [0.71, 0.56, 0.56, 0.52]
```

This is the **contextualized representation** of position 0!

### 3.3 Multi-Head Attention

#### Motivation

**Problem**: Single attention = single "view" of relationships.

**Solution**: Multiple parallel attention heads → learn different relationship types.

**Examples of what different heads might learn**:
- Head 1: Syntactic dependencies (subject-verb)
- Head 2: Semantic similarity (synonyms, related concepts)
- Head 3: Positional relationships (nearby words)
- Head 4: Long-range dependencies (coreference)

#### Architecture

```
Input: X ∈ ℝ^(L × d_model)

┌─────────────────────────────────────────────────────────────┐
│                    Multi-Head Attention                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐ │
│  │   Head 1     │  │   Head 2     │  ...  │   Head H     │ │
│  │              │  │              │       │              │ │
│  │  Q₁ = X·W_Q₁ │  │  Q₂ = X·W_Q₂ │       │  Qₕ = X·W_Qₕ │ │
│  │  K₁ = X·W_K₁ │  │  K₂ = X·W_K₂ │       │  Kₕ = X·W_Kₕ │ │
│  │  V₁ = X·W_V₁ │  │  V₂ = X·W_V₂ │       │  Vₕ = X·W_Vₕ │ │
│  │              │  │              │       │              │ │
│  │  Attn(Q,K,V) │  │  Attn(Q,K,V) │       │  Attn(Q,K,V) │ │
│  │      ↓       │  │      ↓       │       │      ↓       │ │
│  │   A₁ ∈ ℝ^d_k │  │   A₂ ∈ ℝ^d_k │       │   Aₕ ∈ ℝ^d_k │ │
│  └──────────────┘  └──────────────┘       └──────────────┘ │
│          ↓                  ↓                      ↓         │
│          └──────────────────┴──────────────────────┘         │
│                            ↓                                 │
│                    Concatenate                               │
│              [A₁ | A₂ | ... | Aₕ] ∈ ℝ^(L × d_model)         │
│                            ↓                                 │
│                       Output Projection                      │
│                      Concat · W_O                            │
│                            ↓                                 │
│                    Output ∈ ℝ^(L × d_model)                  │
└─────────────────────────────────────────────────────────────┘
```

#### Mathematical Formulation

**Parameters per head**:
```
W_Q^h ∈ ℝ^(d_model × d_k)
W_K^h ∈ ℝ^(d_model × d_k)
W_V^h ∈ ℝ^(d_model × d_k)

where d_k = d_model / num_heads
```

**Output projection**:
```
W_O ∈ ℝ^(d_model × d_model)
```

**Forward pass**:

1. **Project input to Q, K, V for each head**:
   ```
   Q^h = X W_Q^h ∈ ℝ^(L × d_k)
   K^h = X W_K^h ∈ ℝ^(L × d_k)
   V^h = X W_V^h ∈ ℝ^(L × d_k)
   ```

2. **Compute attention for each head**:
   ```
   A^h = Attention(Q^h, K^h, V^h) ∈ ℝ^(L × d_k)
   ```

3. **Concatenate all heads**:
   ```
   Concat = [A^1 | A^2 | ... | A^H] ∈ ℝ^(L × H·d_k) = ℝ^(L × d_model)
   ```

4. **Output projection**:
   ```
   Output = Concat · W_O ∈ ℝ^(L × d_model)
   ```

**Complete formula**:
```
MultiHead(X) = Concat(head₁, ..., headₕ) W_O
where head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
```

#### Implementation

```python
# src/micro_lm/model/attention.py:19-75

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.scale = math.sqrt(self.d_k)

        # Single large projections (split into heads in forward)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)

        # Linear projections: (B, L, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape to (B, num_heads, L, d_k)
        Q = Q.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product: (B, H, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply to values: (B, H, len_q, d_k)
        out = torch.matmul(attn, V)

        # Reshape back: (B, len_q, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        return self.W_o(out)
```

---

## 4. Encoder Architecture

The encoder processes the **source sequence** with full bidirectional context.

### 4.1 Single Encoder Layer

```
Input: X ∈ ℝ^(L × d_model)

┌────────────────────────────────────────┐
│         Encoder Layer                   │
│                                         │
│  X ──────┐                              │
│          │                              │
│          ├──► Multi-Head Self-Attention │
│          │    (Q=X, K=X, V=X)           │
│          │           ↓                  │
│          │        Dropout               │
│          │           ↓                  │
│          └────► + (Residual)            │
│                     ↓                   │
│              LayerNorm                  │
│                     ↓                   │
│                     X₁                  │
│                     │                   │
│         X₁ ─────────┤                   │
│                     │                   │
│                     ├──► FFN            │
│                     │    (Expand→ReLU   │
│                     │     →Contract)    │
│                     │        ↓          │
│                     │     Dropout       │
│                     │        ↓          │
│                     └───► + (Residual)  │
│                            ↓            │
│                      LayerNorm          │
│                            ↓            │
│                      Output: X₂         │
└────────────────────────────────────────┘
```

### 4.2 Mathematical Flow

**Input**: X ∈ ℝ^(L × d_model)

**Block 1: Self-Attention**
```
MultiHeadAttn = MultiHead(X, X, X)  [No mask, sees all positions]
X₁ = LayerNorm(X + Dropout(MultiHeadAttn))
```

**Block 2: Feed-Forward**
```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

where:
  W₁ ∈ ℝ^(d_model × d_ff),  typically d_ff = 4 × d_model
  W₂ ∈ ℝ^(d_ff × d_model)

X₂ = LayerNorm(X₁ + Dropout(FFN(X₁)))
```

**Output**: X₂ ∈ ℝ^(L × d_model)

### 4.3 Full Encoder Stack

```
Input Tokens → Embedding → Position Encoding
                               ↓
                       ┌───────────────┐
                       │ Encoder Layer 1│
                       └───────────────┘
                               ↓
                       ┌───────────────┐
                       │ Encoder Layer 2│
                       └───────────────┘
                               ↓
                             ...
                               ↓
                       ┌───────────────┐
                       │ Encoder Layer N│
                       └───────────────┘
                               ↓
                       Final LayerNorm
                               ↓
                    Memory (Encoder Output)
              ℝ^(batch × src_len × d_model)
```

### 4.4 Key Properties

✅ **Bidirectional**: Each position attends to all positions (past and future)
✅ **Parallel**: All positions processed simultaneously
✅ **No masking**: Full sequence visible
✅ **Contextualized**: Each output vector incorporates information from entire sequence

### 4.5 Implementation

```python
# src/micro_lm/model/encoder.py

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual + norm
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
```

---

## 5. Decoder Architecture

The decoder generates the **target sequence** autoregressively, one token at a time.

### 5.1 Single Decoder Layer

```
Input: Y ∈ ℝ^(T × d_model),  Memory: M ∈ ℝ^(S × d_model)

┌─────────────────────────────────────────────────────────┐
│                  Decoder Layer                           │
│                                                          │
│  Y ──────┐                                               │
│          │                                               │
│          ├──► Masked Multi-Head Self-Attention          │
│          │    (Q=Y, K=Y, V=Y, causal_mask)              │
│          │              ↓                                │
│          │           Dropout                             │
│          │              ↓                                │
│          └────────► + (Residual)                         │
│                         ↓                                │
│                   LayerNorm                              │
│                         ↓                                │
│                        Y₁                                │
│                         │                                │
│         Y₁ ─────────────┤                                │
│                         │                                │
│      Memory (M) ────────┼──► Cross-Attention            │
│      (from encoder)     │    (Q=Y₁, K=M, V=M)           │
│                         │           ↓                    │
│                         │        Dropout                 │
│                         │           ↓                    │
│                         └──────► + (Residual)            │
│                                     ↓                    │
│                               LayerNorm                  │
│                                     ↓                    │
│                                    Y₂                    │
│                                     │                    │
│               Y₂ ───────────────────┤                    │
│                                     │                    │
│                                     ├──► FFN             │
│                                     │    (same as encoder)│
│                                     │        ↓           │
│                                     │     Dropout        │
│                                     │        ↓           │
│                                     └───► + (Residual)   │
│                                              ↓           │
│                                        LayerNorm         │
│                                              ↓           │
│                                        Output: Y₃        │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Mathematical Flow

**Inputs**:
- Y ∈ ℝ^(T × d_model) - decoder hidden states
- M ∈ ℝ^(S × d_model) - encoder memory

**Block 1: Masked Self-Attention**
```
MaskedAttn = MultiHead(Y, Y, Y, mask=causal_mask)
Y₁ = LayerNorm(Y + Dropout(MaskedAttn))
```

**Causal mask**:
```
mask[i,j] = True  if j ≤ i  (can attend to past)
          = False if j > i  (cannot attend to future)

Example for length 4:
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**Block 2: Cross-Attention**
```
CrossAttn = MultiHead(Query=Y₁, Key=M, Value=M)
Y₂ = LayerNorm(Y₁ + Dropout(CrossAttn))
```

**Key insight**: Decoder "queries" encoder memory to find relevant source information!

**Block 3: Feed-Forward**
```
Y₃ = LayerNorm(Y₂ + Dropout(FFN(Y₂)))
```

### 5.3 Full Decoder Stack

```
Target Tokens → Embedding → Position Encoding
                               ↓
                       ┌───────────────┐
                       │ Decoder Layer 1│
                       │ - Masked SA    │
                       │ - Cross Attn   │◄─── Memory
                       │ - FFN          │
                       └───────────────┘
                               ↓
                       ┌───────────────┐
                       │ Decoder Layer 2│
                       │      ...       │◄─── Memory
                       └───────────────┘
                               ↓
                             ...
                               ↓
                       ┌───────────────┐
                       │ Decoder Layer N│
                       │      ...       │◄─── Memory
                       └───────────────┘
                               ↓
                       Final LayerNorm
                               ↓
                Linear(d_model → vocab_size)
                               ↓
                    Logits ∈ ℝ^(T × vocab_size)
                               ↓
                          Softmax
                               ↓
                     Next Token Probabilities
```

### 5.4 Causal Mask Visualization

```
Sequence: [<s>, "what", "is", "AI", "?"]
Positions: 0     1       2     3    4

Position 0 (<s>)      can see: [0]           (only itself)
Position 1 ("what")   can see: [0, 1]
Position 2 ("is")     can see: [0, 1, 2]
Position 3 ("AI")     can see: [0, 1, 2, 3]
Position 4 ("?")      can see: [0, 1, 2, 3, 4]  (all)

Attention matrix shape: (5, 5)
Each row = attention weights for that position
Masked entries = -∞ (become 0 after softmax)
```

### 5.5 Implementation

```python
# src/micro_lm/model/decoder.py

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        # Masked self-attention
        self_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_out))

        # Cross-attention (Q from decoder, K,V from encoder)
        cross_out = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # FFN
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x
```

---

## 6. Complete Forward Pass

Let's trace a complete example through the entire model.

### 6.1 Example Setup

**Question**: "What is machine learning?"
**Answer**: "Machine learning is AI"

**Vocabulary** (simplified):
```
{<pad>: 0, <s>: 1, </s>: 2, "what": 3, "is": 4, "machine": 5,
 "learning": 6, "AI": 7, "?": 8, ...}
```

**Tokenization**:
```
Source: [3, 4, 5, 6, 8]  → "what is machine learning ?"
Target: [1, 5, 6, 4, 7, 2] → "<s> machine learning is AI </s>"
```

**Model config**:
```python
d_model = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
d_ff = 1024
vocab_size = 10000
```

### 6.2 Step-by-Step Forward Pass

#### ENCODER

**Step 1: Embedding**
```
src_ids: [3, 4, 5, 6, 8]  # Shape: (1, 5)
embed = self.src_embed(src_ids)  # Shape: (1, 5, 256)
```

**Step 2: Scaling**
```
embed = embed * √256 = embed * 16  # Shape: (1, 5, 256)
```

**Step 3: Positional Encoding**
```
pos_enc = self.pe[:, :5, :]  # Shape: (1, 5, 256)
x = embed + pos_enc  # Shape: (1, 5, 256)
x = dropout(x)
```

**Step 4: Encoder Layer 1**
```
# Self-attention
Q = K = V = x  # All from same sequence
attn_out = MultiHeadAttn(Q, K, V)  # (1, 5, 256)
x = LayerNorm(x + dropout(attn_out))

# FFN
ffn_out = ReLU(x @ W1) @ W2  # Expand to 1024, contract to 256
x = LayerNorm(x + dropout(ffn_out))
```

**Step 5: Encoder Layers 2-3**
```
Repeat the same process
```

**Step 6: Final Encoder Output**
```
memory = final_layer_norm(x)  # Shape: (1, 5, 256)
```

This is the **encoder memory** - contextualized representation of the source!

#### DECODER

**Step 1: Target Embedding**
```
tgt_ids = [1, 5, 6, 4, 7]  # Shifted target (training)
                            # Predict: [5, 6, 4, 7, 2]
embed = self.tgt_embed(tgt_ids)  # Shape: (1, 5, 256)
```

**Step 2: Scaling + Position**
```
embed = embed * 16
y = embed + pos_enc  # Shape: (1, 5, 256)
```

**Step 3: Decoder Layer 1**

**Block 1: Masked Self-Attention**
```
Q = K = V = y
causal_mask = [[1, 0, 0, 0, 0],
               [1, 1, 0, 0, 0],
               [1, 1, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1]]

attn_out = MultiHeadAttn(Q, K, V, mask=causal_mask)
y = LayerNorm(y + dropout(attn_out))
```

**Block 2: Cross-Attention**
```
Q = y                  # From decoder: (1, 5, 256)
K = V = memory         # From encoder: (1, 5, 256)

# Each decoder position can attend to ALL encoder positions
cross_out = MultiHeadAttn(Q, K, V)  # (1, 5, 256)
y = LayerNorm(y + dropout(cross_out))
```

**Block 3: FFN**
```
y = LayerNorm(y + dropout(FFN(y)))
```

**Step 4: Decoder Layers 2-3**
```
Repeat: masked SA → cross-attention → FFN
```

**Step 5: Output Projection**
```
decoder_out = final_layer_norm(y)  # (1, 5, 256)
logits = decoder_out @ W_out + b   # (1, 5, 10000)
```

**Step 6: Loss Calculation**
```
# Logits at position i predict token at position i+1
predictions = logits[:, :-1, :]  # (1, 4, 10000)
targets = [5, 6, 4, 7, 2]       # True next tokens

loss = CrossEntropyLoss(predictions.flatten(0,1), targets)
```

### 6.3 Inference (Generation)

```python
def generate(model, src, max_len=50):
    memory = model.encode(src)  # Encode once

    tgt = torch.tensor([[SOS_TOKEN]])  # Start with <s>

    for _ in range(max_len):
        # Decode with current target sequence
        out = model.decode(tgt, memory)
        logits = model.fc_out(out)  # (1, current_len, vocab_size)

        # Get next token from last position
        next_token_logits = logits[0, -1, :]  # (vocab_size,)
        next_token = torch.argmax(next_token_logits)

        if next_token == EOS_TOKEN:
            break

        # Append and continue
        tgt = torch.cat([tgt, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return tgt
```

---

## 7. Gradient Flow & Backpropagation

### 7.1 Why Transformers Train Well

**Key mechanisms**:
1. **Residual connections**: Direct gradient path
2. **Layer normalization**: Stable gradients
3. **Attention**: Differentiable routing
4. **Parallel processing**: Efficient computation

### 7.2 Gradient Flow Through Attention

#### Forward Pass
```
S = Q K^T / √d_k
P = softmax(S)
A = P V
```

#### Backward Pass

**Given**: ∂L/∂A (gradient from upstream)

**Step 1: Gradient w.r.t. V**
```
∂L/∂V = P^T (∂L/∂A)
```

**Step 2: Gradient w.r.t. P**
```
∂L/∂P = (∂L/∂A) V^T
```

**Step 3: Gradient through softmax**

For each row i of P (which is softmax of row i of S):
```
∂L/∂S_i = P_i ⊙ (∂L/∂P_i - Σ_j P_ij · ∂L/∂P_ij)
```

Where ⊙ is element-wise multiplication.

**Step 4: Gradient through scaling**
```
∂L/∂(QK^T) = (1/√d_k) · ∂L/∂S
```

**Step 5: Gradients w.r.t. Q and K**
```
∂L/∂Q = (∂L/∂S) K
∂L/∂K = (∂L/∂S)^T Q
```

**Step 6: Gradients w.r.t. weights**
```
∂L/∂W_Q = X^T (∂L/∂Q)
∂L/∂W_K = X^T (∂L/∂K)
∂L/∂W_V = X^T (∂L/∂V)
```

### 7.3 Residual Connection Gradients

```
y = x + f(x)

∂L/∂x = ∂L/∂y · (1 + ∂f/∂x)
```

**Key property**: Always has a direct path (the "1" term) → prevents vanishing gradients!

### 7.4 Layer Normalization Gradients

```
x_norm = (x - μ) / σ
y = γ · x_norm + β

∂L/∂x = (γ/σ) · [∂L/∂y - mean(∂L/∂y) - x_norm·mean(∂L/∂y ⊙ x_norm)]
```

**Property**: Re-centers and re-scales gradients → stable training.

---

## 8. Comparison: Encoder-Decoder vs Decoder-Only

### 8.1 Architecture Comparison

#### Your Model (Encoder-Decoder)

```
SOURCE: "What is ML?"          TARGET: "<s> ML is AI </s>"
         ↓                              ↓
    ┌─────────┐                    ┌─────────┐
    │ ENCODER │                    │ DECODER │
    │         │                    │         │
    │ Bidir   │─────Memory────────►│ Causal  │
    │ Self-   │  (cross-attention) │ Self-   │
    │ Attn    │                    │ Attn    │
    └─────────┘                    │         │
                                   │ Cross-  │
                                   │ Attn    │
                                   └─────────┘
```

#### Claude/GPT (Decoder-Only)

```
SINGLE SEQUENCE: "Question: What is ML?\nAnswer: ML is AI"
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              ↓
                    ┌──────────────────┐
                    │   ONE DECODER    │
                    │                  │
                    │  Causal Self-    │
                    │  Attention       │
                    │  (entire seq)    │
                    │                  │
                    │  NO Cross-Attn   │
                    └──────────────────┘
```

### 8.2 Detailed Comparison Table

| **Feature** | **Encoder-Decoder** | **Decoder-Only (Claude)** |
|-------------|---------------------|---------------------------|
| **Architecture** | Two stacks (encoder + decoder) | Single decoder stack |
| **Attention Types** | • Encoder: bidirectional self-attn<br>• Decoder: causal self-attn<br>• Decoder: cross-attention | • Causal self-attention only |
| **Context** | Encoder sees full source bidirectionally | Everything processed left-to-right |
| **Parameters** | Separate for encoder & decoder | Single set of parameters |
| **Typical Size** | 6-12 layers, 256-512 dim<br>Few million parameters | 32-96 layers, 4096-8192 dim<br>Billions of parameters |
| **Input Format** | Separate source & target | Single concatenated sequence |
| **Training** | Paired (source, target) data | All text (next-token prediction) |
| **Inference** | Encode once, then decode | Process prompt, then generate |
| **Memory** | Store encoder output once | Cache K,V for all prompt tokens |
| **Best For** | Translation, Q&A with clear structure | General-purpose, few-shot, zero-shot |
| **Scaling** | Harder (two architectures to balance) | Easier (one architecture) |
| **Emergence** | Limited at small scale | In-context learning at large scale |

### 8.3 How Claude Handles Your Q&A Task

**Your model**:
```python
# Two separate sequences
source = "What is machine learning?"
target = "<s> Machine learning is..."

# Encoder processes source
memory = encoder(source)

# Decoder generates target using memory
output = decoder(target, memory=memory)
```

**Claude (decoder-only)**:
```python
# Single concatenated sequence
sequence = """Question: What is machine learning?
Answer: Machine learning is"""

# Process entire sequence with causal attention
output = decoder(sequence)  # Predicts next tokens
```

### 8.4 Architectural Evolution

```
2017: Transformer (Encoder-Decoder)
      └─► Neural Machine Translation

2018: GPT (Decoder-Only)
      └─► Language Generation

2019: BERT (Encoder-Only)        GPT-2
      └─► Understanding            └─► Better generation

2020: GPT-3 (175B params)
      └─► Decoder-only dominates
      └─► In-context learning emerges

2021-2024: All major models decoder-only
      ├─► LLaMA, PaLM
      ├─► Claude
      └─► GPT-4
```

### 8.5 Why Decoder-Only Won

**1. Simplicity**
- One architecture for everything
- Easier to implement, debug, scale

**2. Unified Objective**
- All data becomes next-token prediction
- No need for paired data

**3. Scaling Laws**
```
Performance ∝ Parameters^α × Data^β × Compute^γ
```
Decoder-only scales more predictably.

**4. Transfer Learning**
- Same model for:
  - Zero-shot (no examples)
  - Few-shot (few examples)
  - Fine-tuning (many examples)

**5. Emergent Abilities**
At scale (>10B params), decoder-only models develop:
- In-context learning
- Chain-of-thought reasoning
- Instruction following

### 8.6 Modern Decoder Optimizations

**Compared to your implementation**:

| **Component** | **Your Implementation** | **Modern (Claude-scale)** |
|---------------|------------------------|---------------------------|
| **Position** | Sinusoidal PE | RoPE (Rotary Position Embeddings) |
| **Normalization** | LayerNorm (post-norm) | RMSNorm (pre-norm) |
| **Attention** | Standard multi-head | Grouped-Query Attention (GQA) |
| **Activation** | ReLU | SwiGLU, GeGLU |
| **Computation** | Standard ops | Flash Attention |
| **Architecture** | Encoder-Decoder | Decoder-only |

**RoPE (Rotary Position Embeddings)**:
```python
# Instead of adding PE to embeddings
# Rotate Q and K based on position
def apply_rope(x, position):
    theta = position / 10000^(2i/d)
    x_rot = rotate(x, theta)
    return x_rot
```

**GQA (Grouped-Query Attention)**:
```
Standard Multi-Head: H query heads, H key heads, H value heads
GQA: H query heads, K key/value heads (K < H)

Example: 32 query heads, 8 key/value heads
→ 4 query heads share each key/value head
→ Reduces memory, maintains quality
```

**RMSNorm**:
```python
# Simpler than LayerNorm (no mean subtraction, no bias)
def rms_norm(x):
    rms = sqrt(mean(x^2))
    return x / (rms + ε) * γ
```

---

## 9. Modern Transformer Optimizations

### 9.1 Flash Attention

**Problem**: Standard attention is O(n²) in memory for sequence length n.

**Standard attention**:
```python
# Materialize full attention matrix
S = Q @ K.T  # (n, n) - huge for long sequences!
P = softmax(S)
out = P @ V
```

**Flash Attention**:
- Tile computation in blocks
- Fuse operations
- Avoid materializing full attention matrix

**Result**: 2-4x faster, uses less memory.

### 9.2 KV Caching (for Generation)

**Problem**: During generation, recompute attention for all previous tokens each step.

**Without caching** (step t):
```python
# Recompute K, V for all positions 0...t
K = [k_0, k_1, ..., k_t]
V = [v_0, v_1, ..., v_t]
```

**With KV caching**:
```python
# Cache previously computed K, V
K_cache = [k_0, k_1, ..., k_{t-1}]  # From previous steps
k_t = compute_key(x_t)               # Only new token
K = concat(K_cache, k_t)             # Reuse cache
```

**Speedup**: ~10x faster generation for long sequences.

### 9.3 Sparse Attention

**Problem**: O(n²) complexity limits context length.

**Solutions**:

**Local Attention**:
- Each token attends only to nearby tokens
- O(n·k) where k = window size

**Strided Attention**:
- Attend to every k-th token
- O(n²/k)

**Block-Sparse Attention**:
- Combine local + global patterns
- Used in models like BigBird, Longformer

### 9.4 Mixture of Experts (MoE)

**Idea**: Use different FFNs for different tokens.

```python
# Standard FFN: same weights for all tokens
out = FFN(x)

# MoE: route each token to K out of N experts
router_probs = softmax(x @ W_router)
top_k_experts = topk(router_probs, k=2)
out = sum(expert_i(x) * prob_i for i in top_k_experts)
```

**Benefit**: Increase parameters without increasing computation.

---

## 10. Implementation Guide

### 10.1 Building Your Own Transformer

**Step-by-step checklist**:

✅ **1. Embeddings**
- Token embeddings (nn.Embedding)
- Positional encoding (sinusoidal or learned)
- Scale embeddings by √d_model

✅ **2. Attention**
- Implement scaled dot-product attention
- Add multi-head splitting/merging
- Handle masking (padding, causal)

✅ **3. Feed-Forward**
- Two linear layers with non-linearity
- d_model → d_ff → d_model
- Typically d_ff = 4 × d_model

✅ **4. Residual & Norm**
- Add residual connections around each sub-layer
- LayerNorm after each residual
- Optional: pre-norm vs post-norm

✅ **5. Encoder/Decoder**
- Stack multiple layers
- Encoder: self-attention only
- Decoder: self-attention + cross-attention

✅ **6. Output**
- Linear projection to vocabulary
- Softmax for probabilities

### 10.2 Key Hyperparameters

```python
# Small model (CPU, learning)
d_model = 256
num_heads = 8
num_layers = 3-6
d_ff = 1024
batch_size = 16-32

# Medium model (GPU)
d_model = 512
num_heads = 8
num_layers = 6-12
d_ff = 2048
batch_size = 32-64

# Large model (multi-GPU)
d_model = 4096
num_heads = 32
num_layers = 32-96
d_ff = 16384
batch_size = variable (gradient accumulation)
```

### 10.3 Training Tips

**1. Learning Rate Schedule**
```python
# Warmup + decay
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

**2. Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. Label Smoothing**
```python
# Instead of hard targets [0, 0, 1, 0]
# Use soft targets [ε, ε, 1-3ε, ε]
loss = CrossEntropyLoss(label_smoothing=0.1)
```

**4. Dropout**
```python
# Apply dropout to:
- Attention weights
- After each sub-layer
- In FFN
# Typical: 0.1-0.3
```

### 10.4 Debugging Checklist

❌ **Loss not decreasing?**
- Check mask: causal mask should be lower-triangular
- Verify target shifting: decoder input vs labels
- Try smaller learning rate

❌ **Loss exploding?**
- Add gradient clipping
- Reduce learning rate
- Check for NaNs in attention (masking issues)

❌ **Model generating garbage?**
- Check tokenizer decode
- Verify temperature in sampling
- Ensure proper EOS handling

❌ **Out of memory?**
- Reduce batch size
- Use gradient accumulation
- Try gradient checkpointing

### 10.5 Code Structure (Best Practices)

```
your_project/
├── model/
│   ├── attention.py      # Multi-head attention
│   ├── encoder.py        # Encoder layers
│   ├── decoder.py        # Decoder layers
│   ├── transformer.py    # Full model
│   └── embeddings.py     # Embedding + position
├── data/
│   ├── tokenizer.py      # Tokenization
│   ├── dataset.py        # Dataset class
│   └── collate.py        # Batching with padding
├── training/
│   ├── train.py          # Training loop
│   ├── optimizer.py      # LR schedule, warmup
│   └── checkpointing.py  # Save/load models
├── inference/
│   ├── generate.py       # Generation (greedy, beam, sample)
│   └── chatbot.py        # Interactive loop
└── config.py             # Hyperparameters
```

---

## Summary: What You've Learned

### Core Concepts

✅ **Attention Mechanism**
- Queries, Keys, Values
- Scaled dot-product
- Multi-head parallelization

✅ **Positional Information**
- Why needed (permutation invariance)
- Sinusoidal encoding
- Relative position learning

✅ **Encoder-Decoder**
- Bidirectional encoding
- Autoregressive decoding
- Cross-attention bridge

✅ **Training Dynamics**
- Residual connections → gradient flow
- Layer normalization → stability
- Masking strategies

✅ **Modern Architectures**
- Decoder-only dominance
- Scaling laws
- Optimization techniques

### Next Steps

**To go deeper**:
1. Implement decoder-only transformer
2. Add KV caching for faster generation
3. Experiment with different attention patterns
4. Try Flash Attention
5. Scale up: multi-GPU training

**Resources**:
- Original paper: "Attention is All You Need" (Vaswani et al., 2017)
- Annotated Transformer: http://nlp.seas.harvard.edu/annotated-transformer/
- nanoGPT: Minimal decoder-only implementation

---

**This implementation** (`src/micro_lm/`) provides a clean, educational foundation. You now understand:
- Every matrix multiplication in attention
- Why scaling and normalization matter
- How gradients flow through deep networks
- The difference between encoder-decoder and decoder-only
- How modern LLMs like Claude work

You're ready to design transformers yourself! 🚀
