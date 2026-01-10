# Background Knowledge for Sequential Recommendation

## What is Sequential Recommendation?

Sequential Recommendation is a recommendation system task that core idea is:

**Based on user's historical behavior sequence, predict the next item the user might interact with.**

Example:
- User's browsing sequence on e-commerce: [phone case, charger, screen protector, Bluetooth earphones]
- Model needs to predict: What will the user buy next? (Possibly: phone stand, earphone case, etc.)

## Why Sequential Recommendation?

Traditional recommendation systems (like collaborative filtering) assume user interests are static, ignoring:
1. User interests evolve over time
2. There are temporal dependencies between items (e.g., buying a phone leads to buying accessories)
3. User behavior has contextual relevance

Sequential recommendation can more accurately capture users' current needs by modeling the temporal dynamics of user behavior.

## SASRec Model Overview

**SASRec = Self-Attentive Sequential Recommendation**

This is a model proposed by Kang and McAuley in 2018, introducing the Transformer architecture to sequential recommendation tasks.

### Core Innovations

1. **Self-Attention Mechanism**: Replaces traditional RNN/LSTM, better capturing long-term dependencies in sequences
2. **Positional Embedding**: Adds positional information to each position in the sequence
3. **Multi-layer Stacking**: Learns different levels of sequence patterns by stacking multiple attention blocks

## Model Architecture Details

### 1. Embedding Layer

```
Input: Item IDs (e.g., [5, 3, 7, ...])
Output: Dense vector representations (e.g., [[0.1, 0.3, ...], [0.2, 0.1, ...], ...])
```

**Item Embedding**: Maps each item ID to a fixed-dimensional vector

**Positional Embedding**: Records information about each position in the sequence (e.g., "this is 1st, 2nd...")

### 2. Self-Attention

This is the core component of Transformer. Simply understood:

```
For each position in the sequence, compute its association strength with all other positions
Then weight-sum to get a new representation based on association strength
```

Example:
- Position 4 (most recent behavior) might pay more attention to positions 3, 2
- Position 1 (earliest behavior) might only be associated with few positions

### 3. Feed Forward Network

Each attention layer is followed by a feed-forward network:

```
Input -> Linear Transform -> ReLU -> Dropout -> Linear Transform -> Dropout -> Output
```

Function: Add non-linear expression capability, allowing the model to learn more complex patterns.

### 4. LayerNorm and Residual Connection

Each sublayer (attention, feed-forward network) uses:
- **LayerNorm**: Standardization, stable training
- **Residual Connection**: Alleviate gradient disappearance, help training deep networks

## Training Strategy

### Contrastive Learning

SASRec uses contrastive learning strategy for training:

**Positive samples**: The next item the user actually interacted with
**Negative samples**: Random items the user never interacted with

**Training objective**: Make the positive sample score higher than negative sample scores

### Loss Function

Binary Cross-Entropy Loss (BCE Loss):

```
Loss = -log(sigmoid(positive score)) - log(1 - sigmoid(negative score))
```

Intuitive understanding: Want high scores for positive samples, low scores for negative samples.

## Evaluation Metrics

### NDCG (Normalized Discounted Cumulative Gain)

Measures the quality of recommendation list ranking.

- The higher the true item is ranked, the higher the NDCG
- Range: 0 to 1, higher is better

### HR (Hit Rate)

Hit rate, measures whether the recommendation list contains the target item.

- If the true item is in the top 10 of the recommendation list, HR@10 = 1
- Range: 0 to 1, higher is better

## Code Structure Explanation

### main.py

Program entry point, responsible for:
1. Parsing command-line arguments
2. Loading and partitioning data
3. Initializing models
4. Training loop
5. Model evaluation and saving

### model.py

Model definitions, containing:
- `PointWiseFeedForward`: Feed-forward neural network
- `SASRec`: Complete sequential recommendation model
- `TiSASRec`: Time-aware sequential recommendation model

### model_mhc.py

mHC variant model definitions, containing:
- `mHCResidual`: Manifold-Constrained Hyper-Connections residual module
- `SASRec_mHC`: SASRec with mHC
- `TiSASRec_mHC`: TiSASRec with mHC

### utils.py

Utility functions, containing:
- `build_index`: Build user-item index
- `WarpSampler`: Multi-process sampler
- `data_partition`: Data partitioning
- `evaluate`/`evaluate_valid`: Model evaluation
- `WarpSamplerWithTime`: Time-aware sampler

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 128 | Batch size, larger is more stable but slower |
| `--lr` | 0.001 | Learning rate, too large is unstable, too small converges slowly |
| `--maxlen` | 200 | Maximum sequence length |
| `--hidden_units` | 50 | Embedding dimension |
| `--num_blocks` | 2 | Number of Transformer Blocks |
| `--num_heads` | 1 | Number of attention heads |
| `--dropout_rate` | 0.2 | Dropout ratio |
| `--num_epochs` | 300 | Number of training epochs |

## Running Command Examples

### Training Model
```bash
cd python
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

### Inference Only (Using Pre-trained Model)
```bash
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path=xxx.pth --inference_only=true --maxlen=200
```

### With mHC
```bash
python main.py --dataset=ml-1m --train_dir=mhc_test --mhc_expansion_rate=4
```

### With Time Features (TiSASRec)
```bash
python main.py --dataset=ml-1m --train_dir=tisasrec --use_time
```

## Negative Sampling Detail

Negative sampling is a key technology for training recommendation models.

**Why negative sampling?**
- Number of items is usually large (tens of thousands to millions)
- Impossible to compute loss for all items
- Just need the model to "know" some items are inappropriate

**Negative sampling strategy:**
1. Randomly select from items the user hasn't interacted with
2. Ensure negative samples haven't been interacted by the user
3. Generally pair each positive sample with multiple negative samples (code uses 1:1)

## Causal Attention Mask

In sequence prediction tasks, need to prevent information leakage.

**Problem**: When predicting the item at position 4, shouldn't see information at positions 5, 6...

**Solution**: Use lower triangular mask

```
Position:   1   2   3   4
Can see:    ✓   ✗   ✗   ✗
Can see:    ✓   ✓   ✗   ✗
Can see:    ✓   ✓   ✓   ✗
Can see:    ✓   ✓   ✓   ✓
```

## Data Format

Training data file format (data/xxx.txt):
```
UserID ItemID
1 5
1 7
1 3
2 10
...
```

Each line represents one user-item interaction, ordered by time.

## Core Concepts Explained

### 1. Neuron

**Definition**: Basic computing unit in neural network, simulating working of human brain neurons

**Structure**:
```
Input: [x1, x2, x3, ...]
       ↓
Weights: [w1, w2, w3, ...]
       ↓
Weighted sum: Σ(xi × wi)
       ↓
Activation function: f(Σ(xi × wi))
       ↓
Output
```

### 2. Dropout

**Function**: Prevent model overfitting (overly memorizing training data)

**Operation**: During training, randomly set some neuron outputs to 0

**Purpose**:
- Prevent model from over-relying on specific neurons
- Enhance model generalization capability
- Reduce overfitting risk

### 3. LayerNorm

**Operation**: Standardize each layer of each sample

**Formula**:
```
output = (x - mean(x)) / std(x) × γ + β
```

**Purpose**:
- Stabilize training process
- Speed up convergence
- Reduce sensitivity to initialization

### 4. Residual Connection

**Operation**: Directly add input to output

**Structure**:
```
Input → [Network Layer] → Output
  ↓                     ↓
   →——Add——→
```

**Purpose**:
- Alleviate gradient disappearance
- Make it easier for model to learn "identity mapping"
- Allow deeper networks

**Intuitive Understanding**: If a layer learns nothing (output=input), the network can still work normally, information can be directly transmitted.

### 5. Transformer Block

**Definition**: Basic unit of Transformer architecture, containing self-attention layer and feed-forward network

**Structure**:
```
Input
  ↓
LayerNorm
  ↓
Multi-Head Attention
  ↓
Residual Connection (+)
  ↓
LayerNorm
  ↓
Feed Forward
  ↓
Residual Connection (+)
  ↓
Output
```

### 6. Why Stack Multiple Blocks?

**Reason**: Each layer learns different levels of information

```
Input Sequence: [A, B, C, D]

Block 1 (Learn local patterns):
  - Relationship between A and B
  - Relationship between B and C
  - Relationship between C and D

Block 2 (Learn complex patterns):
  - Relationship between A and D (long-range dependency)
  - Relationship between A and C
  - Relationship between B and D

Block 3+ (Learn deep abstraction):
  More complex combined patterns
```

### 7. Feed Forward Network

**Structure**:
```
Input → Linear Transform → ReLU → Dropout → Linear Transform → Dropout → Output
```

**Function**:
- Add non-linear expression capability
- Transform each position independently
- Enhance model fitting capability

**Intuitive Understanding**:
- Self-attention is "information exchange": Let different positions exchange information with each other
- Feed-forward network is "independent thinking": Let each position process information independently

### 8. Self-Attention

**Core Idea**: Each position in the sequence can "attend to" all other positions

**Calculation Steps**:

**Step 1: Generate Q/K/V vectors**
```
Input: [h1, h2, h3] (each is 50-dimensional vector)
       ↓
   Q/K/V Matrix Transform
       ↓
   Q = [q1, q2, q3]
   K = [k1, k2, k3]
   V = [v1, v2, v3]
```

**Step 2: Compute attention scores**
```
score_ij = q_i · k_j / √d (d is vector dimension, e.g., 50)

For example, compute position 1's attention to other positions:
- q1·k1/√50 → Position 1's attention to itself
- q1·k2/√50 → Position 1's attention to position 2
- q1·k3/√50 → Position 1's attention to position 3
```

**Step 3: Softmax normalization**
```
attention_ij = exp(score_ij) / Σexp(score_i)
```

**Step 4: Weighted sum**
```
new_h1 = attention_11×v1 + attention_12×v2 + attention_13×v3
```

### 9. Loss Function

**Function**: Measure the gap between model prediction and true target

**Code Example**:
```python
bce_criterion = torch.nn.BCEWithLogitsLoss()
loss = bce_criterion(pos_logits, pos_labels)      # Positive sample loss
loss += bce_criterion(neg_logits, neg_labels)     # Negative sample loss
```

**BCE Loss Formula**:
```
Loss = -log(sigmoid(positive score)) - log(1 - sigmoid(negative score))
```

**Intuitive Understanding**:
- Positive samples (items user actually interacted with): Want higher scores → probability close to 1 → log(1)=0 → small loss
- Negative samples (items user didn't interact with): Want lower scores → probability close to 0 → log(1)=0 → small loss

**Training Objective**: Continuously reduce Loss through backpropagation, making the model learn to distinguish between "user will like" and "user doesn't like" items.

### 10. Why This Design?

| Design | Problem Solved | Effect |
|--------|---------------|--------|
| Self-Attention | RNN long sequence information loss | Directly model arbitrary distance dependencies |
| Multi-Head Attention | Single attention pattern limited | Capture multiple association types |
| Positional Embedding | Sequence order information loss | Inject positional information |
| LayerNorm | Training instability | Stabilize training process |
| Residual Connection | Deep networks hard to train | Support deeper networks |
| Dropout | Overfitting | Improve generalization |
| Feed Forward Network | Attention layer expression limited | Enhance non-linear ability |
| Multi-layer Stacking | Shallow model capability limited | Learn multi-level features |

**Overall Approach**:
1. Use positional embedding to preserve sequence order
2. Use self-attention to model dependencies
3. Use multi-layer stacking to learn multi-level patterns
4. Use various techniques (Norm, residual, Dropout) to ensure stable training

This is why the Transformer architecture succeeds!

## FAQ

### Q: Why can the model predict what the user wants to buy next?

A: The model learns:
1. Co-occurrence patterns between items (e.g., buying phone leads to buying accessories)
2. Evolution of user interests over time
3. Temporal dependency relationships

### Q: Why is positional embedding important?

A: Without positional information, the model cannot distinguish:
- [A, B] and [B, A] sequences
- What the user bought first and last

### Q: How to choose hyperparameters?

A: Usually through experimental tuning:
- Start with default values
- Adjust based on validation set performance
- Pay attention to overfitting issues

## Extended Learning

1. **Transformer Original Paper**: Attention Is All You Need
2. **SASRec Original Paper**: Self-attentive sequential recommendation
3. **Other Sequential Recommendation Models**: GRU4Rec, BERT4Rec, TiSASRec
4. **mHC Paper**: Manifold-Constrained Hyper-Connections (DeepSeek-AI, 2025)

## References

```bibtex
@article{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}

@article{li2020time,
  title={Time Interval Aware Self-Attention for Sequential Recommendation},
  author={Li, Zeynep and McAuley, Julian},
  year={2020}
}

@article{deepseek2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={DeepSeek-AI},
  year={2025}
}
```
