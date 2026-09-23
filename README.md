# How Transformer LLMs Work - Complete Theory

## Overview: The Big Picture

Large Language Models (LLMs) like Phi-3 are essentially **next-token prediction machines**. 

Given a sequence of tokens, they predict what token should come next. 
By repeating this process, they can generate entire responses.

```
Input:  "The capital of France is"
Model:  [processes] → predicts "Paris"
Output: "The capital of France is Paris"
```

## 1. Tokenization: Converting Text to Numbers

### What are Tokens?

Tokens are the basic units that models work with. They can be:
- Whole words: `"hello"` → token `15339`
- Subwords: `"running"` → `"run"` + `"ning"`
- Characters: `"a"` → token `97`

**Example from your notebook:**
```
"The capital of France is"
     ↓
[450, 7483, 310, 3444, 338]
```

### Why Tokenization?

- Neural networks work with numbers, not text
- Phi-3 has a vocabulary of **32,064 tokens**
- Each token gets a unique ID (0 to 32,063)

---

## 2. Architecture: The Three Main Components

### Component 1: Token Embeddings

**What it does:** Converts token IDs into dense vectors (embeddings)

```
Token ID: 450 ("The")
     ↓
Embedding Layer (32064 × 3072)
     ↓
Vector: [0.23, -0.15, 0.87, ..., 0.42]  (3072 dimensions)
```

**In Phi-3:**
- Input: Token ID (integer 0-32063)
- Output: 3072-dimensional vector
- This vector captures semantic meaning

**Why embeddings?** 
Similar words get similar vectors. "king" and "queen" have vectors close in the 3072-dimensional space.

---

### Component 2: Transformer Layers (The Brain)

Phi-3 has **32 decoder layers** stacked on top of each other. Each layer has two main parts:

#### Part A: Self-Attention Mechanism

**Purpose:** Let tokens "look at" and learn from other tokens in the sequence

**How it works:**

1. **Query (Q), Key (K), Value (V) Projections**
   ```
   Input vector → Linear projections
   
   Q = "What am I looking for?"
   K = "What do I contain?"
   V = "What information do I have?"
   ```

2. **Attention Scores**
   ```
   For token "is" in "The capital of France is":
   
   Attention to "The":      0.05
   Attention to "capital":  0.10
   Attention to "of":       0.08
   Attention to "France":   0.70  ← High attention!
   Attention to "is":       0.07
   ```

3. **Causal Masking** (Why it's "Causal")
   ```
   Token "France" can see:  [The, capital, of, France]
   Token "France" CANNOT see: [is]
   
   This prevents "cheating" - tokens can only attend to previous tokens!
   ```

4. **Weighted Sum**
   ```
   New representation of "is" = 
       0.05 × V("The") + 
       0.10 × V("capital") +
       0.08 × V("of") + 
       0.70 × V("France") +
       0.07 × V("is")
   ```

**In Phi-3's code:**
```python
Phi3Attention(
    (qkv_proj): Linear(3072 → 9216)  # Creates Q, K, V
    (o_proj): Linear(3072 → 3072)    # Output projection
    (rotary_emb): Phi3RotaryEmbedding()  # Position encoding
)
```

#### Part B: Feed-Forward Network (MLP)

**Purpose:** Process each token's representation independently

```
Input (3072) 
    ↓
gate_up_proj: Expand to 16384 dimensions
    ↓
SiLU activation (smooth non-linearity)
    ↓
down_proj: Compress back to 3072
    ↓
Output (3072)
```

**Why expand then compress?**
- The expansion creates a "high-dimensional thinking space"
- The model can learn more complex patterns
- Similar to how our brain processes information through many neurons

**In Phi-3's code:**
```python
Phi3MLP(
    (gate_up_proj): Linear(3072 → 16384)
    (down_proj): Linear(8192 → 3072)
    (activation_fn): SiLUActivation()
)
```

#### Other Components in Each Layer:

**Layer Normalization (RMSNorm):**
- Stabilizes training
- Applied before attention and MLP
- Normalizes the vectors to have consistent scale

**Residual Connections:**
```
output = input + attention(normalize(input))
output = output + mlp(normalize(output))
```
- Helps gradient flow during training
- Allows the model to learn incremental changes

---

### Component 3: Language Model Head (LM Head)

**Purpose:** Convert the final hidden state into token probabilities

```
Last hidden state (3072 dimensions)
    ↓
Linear projection (3072 → 32064)
    ↓
Logits for each token in vocabulary
    ↓
Softmax (optional)
    ↓
Probability distribution
```

**Example:**
```
Input: "The capital of France is"
Last token ("is") hidden state → LM Head
    ↓
Logits: [-2.3, 0.5, 8.7, -1.2, ..., 3.4]
                    ↑
                Position of "Paris"
    ↓
argmax() → Token ID for "Paris"
```

---

## 3. The Forward Pass: Step-by-Step

Let's trace what happens when you process "The capital of France is":

### Step 1: Tokenization
```
"The capital of France is" → [450, 7483, 310, 3444, 338]
```

### Step 2: Embedding
```
[450, 7483, 310, 3444, 338]
    ↓
Embedding Layer
    ↓
Shape: (1, 5, 3072)
      batch  tokens  embedding_dim
```

### Step 3: Through 32 Decoder Layers

**Layer 1:**
```
Input: (1, 5, 3072)
    ↓
LayerNorm → Self-Attention → Add residual
    ↓
LayerNorm → MLP → Add residual
    ↓
Output: (1, 5, 3072)
```

**Layers 2-32:** Same process, each learning different patterns
- Early layers: Basic patterns (grammar, syntax)
- Middle layers: Semantic relationships
- Late layers: Task-specific knowledge

### Step 4: Final Layer Norm
```
Output from Layer 32: (1, 5, 3072)
    ↓
RMSNorm
    ↓
Normalized output: (1, 5, 3072)
```

### Step 5: LM Head
```
Normalized output: (1, 5, 3072)
    ↓
Linear projection
    ↓
Logits: (1, 5, 32064)
        ↑   ↑    ↑
     batch tokens vocab
```

### Step 6: Next Token Prediction
```
Take last token's logits: (32064,)
    ↓
argmax()
    ↓
Token ID: 3681 (represents "Paris")
    ↓
Decode: "Paris"
```

---

## 4. Key Concepts Explained

### A. Why "Causal" Language Model?

**Causal** means the model can only look at previous tokens, not future ones.

```
When predicting after "France":
✓ Can see: "The capital of France"
✗ Cannot see: "is" (comes after)

This is enforced by attention masking!
```

**Why?** During training, this prevents the model from "cheating" by looking at the answer.

### B. What are "Logits"?

**Logits** are raw, unnormalized scores for each token in the vocabulary.

```
Logits: [-2.3, 0.5, 8.7, -1.2, ...]
         ↓
Softmax (converts to probabilities)
         ↓
Probs:  [0.001, 0.016, 0.954, 0.003, ...]
        
Higher logit = higher probability
```

### C. Greedy Decoding vs Sampling

**Greedy (do_sample=False):**
```
Always pick token with highest probability
Deterministic - same output every time
```

**Sampling (do_sample=True):**
```
Pick tokens based on probability distribution
Random - different outputs each time
More creative but less predictable
```

### D. Temperature

Controls randomness in sampling:
```
Temperature = 0.1:  Very focused, picks top choices
Temperature = 1.0:  Normal distribution
Temperature = 2.0:  More random, considers many options
```

---

## 5. Training: How Models Learn

### Pre-training (What Phi-3 Already Did)

**Objective:** Predict the next token

```
Text: "The cat sat on the mat"

Training examples:
"The" → predict "cat"
"The cat" → predict "sat"
"The cat sat" → predict "on"
...
```

**Loss Function:**
```
Cross-entropy loss between:
- Predicted probabilities
- Actual next token (one-hot encoded)

Model adjusts weights to minimize this loss
```

**Data Scale:**
- Phi-3 was trained on trillions of tokens
- From books, websites, code, conversations
- Takes weeks on hundreds of GPUs

### Fine-tuning (Optional)

Further training on specific tasks:
- Instruction following
- Conversation
- Code generation
- Domain-specific knowledge

---

## 6. Why Does This Work?

### Pattern Recognition
Through training on massive data, the model learns:
- Grammar rules
- World knowledge
- Reasoning patterns
- Common sense
- Cultural references

### Emergent Abilities
Surprisingly, at scale, models develop abilities not explicitly trained:
- Translation
- Summarization
- Question answering
- Basic reasoning
- Code generation

### Limitations
- No true understanding (statistical patterns)
- Can generate false information
- Limited by training data cutoff
- No real-world grounding

---

## 7. Key Architecture Numbers for Phi-3

```
Vocabulary Size:     32,064 tokens
Embedding Dimension: 3,072
Number of Layers:    32
Attention Heads:     32 (96 dim each)
MLP Hidden Size:     8,192
Total Parameters:    ~3.8 billion

Context Window:      4,096 tokens
```

---

## 8. Autoregressive Generation

How multi-token generation works:

```
Step 1: "The capital of France is" → predict "Paris"
Step 2: "The capital of France is Paris" → predict ","
Step 3: "The capital of France is Paris," → predict "a"
Step 4: "The capital of France is Paris, a" → predict "beautiful"
...

Continues until:
- Max tokens reached
- End-of-sequence token generated
- User stops generation
```

Each prediction depends on ALL previous tokens!

---

## Summary

**The transformer LLM workflow:**
1. **Tokenize** text into IDs
2. **Embed** IDs into vectors
3. **Process** through 32 layers of attention + MLP
4. **Project** to vocabulary space
5. **Predict** next token
6. **Repeat** for multiple tokens

**The magic:** Self-attention allows the model to learn complex relationships between tokens, enabling it to capture language patterns, world knowledge, and reasoning abilities from training data.

## ML System Design

<img src="Project Documentation/ML System Design Of Our LLM.png" width="800">
