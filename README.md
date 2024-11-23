# **Recursive GPT Implementation in C++**

This project implements a GPT-like architecture in C++ with a focus on recursive reasoning capabilities. The model integrates tokenization, embedding layers, and transformer blocks into a unified architecture, complete with training functionality, logging, and debugging support.

---

## **Project Structure**

```
project/
├── src/        # Source files
├── include/    # Header files
├── lib/        # External libraries (optional)
├── build/      # Compiled output files
├── tests/      # Unit tests
├── data/       # Data files (corpus, training data, etc.)
├── logs/       # Log files for debugging and tracking execution
├── Makefile    # Build and run instructions
```

---

## **Implemented Features**

### **1. Tokenizer**
**Purpose**: Converts raw text into token IDs.
- **Features**:
  - Build a vocabulary from a text corpus.
  - Tokenize input text into a sequence of integers (token IDs).

### **2. Embedding Layer**
**Purpose**: Maps token IDs into dense vector representations.
- **Features**:
  - Learnable embedding matrix initialized with random values.
  - Provides embeddings for input token IDs.

### **3. Transformer Block**
**Purpose**: Implements multi-head attention and feed-forward layers.
- **Features**:
  - **Multi-Head Attention**: Captures relationships between tokens.
  - **Feed-Forward Network**: Applies non-linear transformations.
  - **Residual Connections**: Stabilizes gradients for deeper models.

### **4. Output Layer**
**Purpose**: Maps final transformer outputs to logits and probabilities.
- **Features**:
  - Softmax activation to convert logits into probabilities.

### **5. Loss Function**
**Purpose**: Measures the difference between predictions and targets.
- **Features**:
  - Cross-entropy loss for classification.
  - Gradient computation for weight updates.

### **6. Training Functionality**
**Purpose**: Trains the GPTModel to minimize loss on a given dataset.
- **Features**:
  - Forward pass to compute predictions.
  - Backward pass to compute gradients.
  - Weight and bias updates using gradient descent.

---

## **How to Build and Run**

### **1. Build the Project**
Use the Makefile to compile the project. Run:
```bash
make
```

### **2. Run Training Tests**
Use the Makefile to execute the training test:
```bash
make test
```

### **3. Clean Build Files**
To remove all compiled files and logs, use:
```bash
make clean
```

---

## **Example Logs**
Training logs provide detailed insights into the forward and backward passes:
```
[2024-11-23T15:30:00.123] [INFO] Starting GPTModel training pass
[2024-11-23T15:30:00.456] [DEBUG] Computed loss: 1.2034
[2024-11-23T15:30:00.789] [DEBUG] gradients: 2x100
[2024-11-23T15:30:01.123] [DEBUG] output_weights: 100x16
[2024-11-23T15:30:01.456] [DEBUG] Updated model weights and biases
[2024-11-23T15:30:02.123] [INFO] Training pass completed successfully.
```

---

## **Next Steps**
1. **Evaluation Metrics**:
   - Implement metrics like accuracy or perplexity.
2. **Real Dataset**:
   - Train the model on a realistic text dataset.
3. **Optimization**:
   - Speed up training with threading or GPU acceleration.

