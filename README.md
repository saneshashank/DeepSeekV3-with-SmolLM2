# DeepSeek-MoE Model Implementation and Training

This notebook provides a custom implementation of the DeepSeek-MoE (Mixture of Experts) model, inspired by the architecture used in models like DeepSeek-V2. It includes custom layers for Multi-Head Latent Attention (MHLA), a DeepSeekDecoderLayer, DeepSeekExpert, and DeepSeekMoE, along with utility components like LlamaRotaryEmbedding and DeepSeekRMSNorm.

The goal of this notebook is to:
1. Implement the core components of the DeepSeek-MoE architecture from scratch.
2. Initialize a custom `DeepSeekModel` with parameters largely derived from `HuggingFaceTB/SmolLM2-135M`.
3. Train the custom model on a subset of the `HuggingFaceFW/fineweb-edu` dataset using PyTorch Lightning.
4. Evaluate the trained model's text generation capabilities and perplexity.

## Key Components Implemented:

### 1. `LlamaRotaryEmbedding`
- **Purpose**: Implements Rotary Position Embeddings (RoPE) to inject positional information into attention keys and queries without relying on absolute positional encodings.
- **Mechanism**: Rotates vectors based on their position, preserving relative positional information.

### 2. `DeepSeekRMSNorm`
- **Purpose**: A simplified normalization layer, similar to LayerNorm but using Root Mean Square (RMS) for normalization, improving training stability.
- **Mechanism**: Normalizes inputs by their RMS, scaled by a learnable weight.

### 3. `DeepSeekExpert`
- **Purpose**: A gated feedforward network (SwiGLU) that serves as the basic expert unit within the Mixture-of-Experts layer.
- **Architecture**: Consists of `gate_proj`, `up_proj`, and `down_proj` linear layers with a SiLU activation function.

### 4. `MultiHeadLatentAttention` (MHLA)
- **Core Innovation**: Compresses the attention mechanism by projecting Q, K, V into a lower-dimensional latent space before attention computation, reducing computational cost.
- **Key Features**: Two-stage projection (down and up), dual-stream keys & queries (context + RoPE), and standard multi-head attention structure with causal masking.

### 5. `DeepSeekMoE`
- **Purpose**: Replaces the standard feedforward network in a transformer layer with a Mixture of Experts.
- **Architecture**: Combines `shared_experts` (always active) and `routed_experts` (selected via a router network with top-k routing).
- **Load Balancing**: Includes mechanisms (`update_bias_terms`, `compute_balance_loss`) to encourage balanced utilization of experts.

### 6. `DeepSeekDecoderLayer`
- **Purpose**: Represents a single decoder block in the DeepSeek-MoE model.
- **Architecture**: Follows a pre-norm structure with two main sublayers: `MultiHeadLatentAttention` and `DeepSeekMoE`, each with residual connections and RMS normalization.

### 7. `DeepSeekModel`
- **Purpose**: The full DeepSeek-MoE language model architecture.
- **Architecture**: Comprises an embedding layer, a stack of `DeepSeekDecoderLayer` modules, a final RMS normalization, and a language modeling head. It also implements weight tying between the embedding layer and the LM head.

## Training Process:

1.  **Model Initialization**: A `DeepSeekModel` is instantiated with parameters like `vocab_size`, `hidden_size`, `num_layers`, `num_heads`, `intermediate_size`, `compression_ratio`, `num_experts`, `num_shared_experts`, and `top_k`.
    -   Parameters for `vocab_size`, `hidden_size`, `num_heads`, and `intermediate_size` are extracted from a pre-trained `SmolLM2-135M` model loaded from Hugging Face Transformers.
    -   Embedding weights are copied from `SmolLM2-135M` to the custom model for faster convergence.

2.  **Data Loading**: The `HuggingFaceFW/fineweb-edu` dataset (sample-10BT split) is loaded in streaming mode.

3.  **Tokenizer**: The `AutoTokenizer` from `SmolLM2-135M` is used to tokenize the dataset.

4.  **PyTorch Lightning Setup**:
    -   A `DeepSeekLightningModule` wraps the `DeepSeekModel` for streamlined training.
    -   The `training_step` calculates cross-entropy loss and incorporates a `balance_loss` from the MoE layers to prevent expert underutilization. Expert biases are also updated.
    -   An `AdamW` optimizer and a `CosineAnnealingLR` scheduler are configured.
    -   A `FineWebEduDataModule` handles data loading and batching using `DataLoader`.
    -   A `Trainer` is set up with `max_steps=10000`, `precision="16-mixed"`, and includes `ModelCheckpoint` and `TensorBoardLogger` callbacks.

5.  **Training Execution**: The model is trained for 10,000 steps using `trainer.fit()`.
   <img width="1116" height="357" alt="image" src="https://github.com/user-attachments/assets/f0cd4206-10bb-48e8-a7cf-49744ae32045" />


## Evaluation:

After training, the model's performance is evaluated:
-   **Text Generation**: A `generate_text` function is used to produce continuations based on a given prompt (e.g., "Once upon a time"), demonstrating the model's ability to generate coherent text.
-   
<img width="1685" height="203" alt="image" src="https://github.com/user-attachments/assets/ce02a2f8-74fd-49a4-885c-c03b6d0fe3b8" />


-   **Perplexity**: The cross-entropy loss from the last training step is used to calculate perplexity, which serves as a metric for the model's uncertainty in predicting the next token. After 10,000 steps, the perplexity was reduced significantly from a hypothetical initial value.
  

<img width="453" height="213" alt="image" src="https://github.com/user-attachments/assets/9ee250a6-2abc-415d-a00e-d764b4a61c66" />


## Dependencies:

-   `torch`
-   `torch.nn`
-   `torch.nn.functional`
-   `pytorch_lightning`
-   `transformers`
-   `datasets`
-   `math`
-   `tensorboard`
