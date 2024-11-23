# **Recursive GPT Implementation in C++**

This project implements a GPT-like architecture in C++ with a focus on recursive reasoning capabilities. The model integrates tokenization, embedding layers, and transformer blocks into a unified architecture, complete with logging and debugging support.

---

## **Project Structure**
The project follows a modular directory structure:
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

## **Environment Setup**

### **1. System Setup**
Run the following commands to set up the development environment on Ubuntu:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential cmake g++ libboost-all-dev libeigen3-dev \
                 libopenblas-dev libdlib-dev libarmadillo-dev libopencv-dev \
                 git gdb valgrind nvidia-cuda-toolkit -y
```

### **2. Verify Setup**
Test the C++ environment with a simple program:
```cpp
#include <iostream>
int main() {
    std::cout << "C++ Environment Setup Successful!" << std::endl;
    return 0;
}
```
Compile and run:
```bash
g++ test.cpp -o test
./test
```

---

## **Implemented Components**

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
  - Fully integrated logging for debugging.

### **4. Logging System**
**Purpose**: Provides detailed execution tracking with timestamps in ISO 8601 format.
- Logs messages at levels: `INFO`, `DEBUG`, `WARNING`, `ERROR`.
- Logs are sequentially written to `logs/project.log`.

---

## **How to Build and Run**

### **1. Build the Project**
Use the Makefile to compile the project. Run:
```bash
make
```
This will build the main executable and any necessary tests.

### **2. Run Tests**
Use the Makefile to execute tests:
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
The following log entries are written to `logs/project.log` during execution:
```
[2024-11-22T15:50:00.123] [INFO] Initializing TransformerBlock
[2024-11-22T15:50:00.456] [DEBUG] Multi-head attention parameters initialized
[2024-11-22T15:50:00.789] [INFO] Starting forward pass of TransformerBlock
[2024-11-22T15:50:01.123] [DEBUG] Computed Q, K, V matrices for multi-head attention
[2024-11-22T15:50:01.456] [INFO] End-to-end execution completed successfully.
```

---

## **Next Steps**

### **1. Integrate Components into GPTModel**
- Combine Tokenizer, EmbeddingLayer, and TransformerBlock.
- Implement an end-to-end forward pass.

### **2. Add Training and Inference**
- Define a loss function for text generation.
- Implement a training loop to optimize the model.

### **3. Test with Real Data**
- Use a small dataset for evaluation.
- Validate the model’s accuracy and output quality.
