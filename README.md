# **Recursive GPT Implementation in C++**

This project aims to implement a GPT-like architecture in C++ with a focus on recursive reasoning capabilities. The project is structured to ensure scalability and modularity for future expansion.

---

## **Project Structure**
The project follows a modular directory structure for clean organization:
```
project/
├── src/        # Source files
├── include/    # Header files
├── lib/        # External libraries (optional)
├── build/      # Compiled output files
├── tests/      # Unit tests
├── data/       # Data files (corpus, training data, etc.)
├── logs/       # Log files for debugging and tracking execution
```

---

## **Environment Setup**

### **1. System Setup**
Run the following commands to prepare the development environment on Ubuntu:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install build-essential cmake g++ -y

# Install Boost (general-purpose C++ libraries)
sudo apt install libboost-all-dev -y

# Install Eigen (linear algebra library)
sudo apt install libeigen3-dev -y

# Install OpenBLAS (optimized basic linear algebra operations)
sudo apt install libopenblas-dev -y

# Install DLib (machine learning and image processing)
sudo apt install libdlib-dev -y

# Install Armadillo (linear algebra and ML library)
sudo apt install libarmadillo-dev -y

# Install OpenCV (optional: for computer vision tasks)
sudo apt install libopencv-dev -y

# Install Git (version control system)
sudo apt install git -y

# Install GDB (debugging tool)
sudo apt install gdb -y

# Install Valgrind (memory and performance profiler)
sudo apt install valgrind -y

# Install CUDA (GPU acceleration for machine learning)
sudo apt install nvidia-cuda-toolkit -y
```

### **2. Verify Setup**
Test the C++ environment with a simple program:
```bash
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
The `Tokenizer` class preprocesses raw text by tokenizing it and converting it into token IDs.  
**Features**:
- Build a vocabulary from a text corpus.
- Convert input text into a sequence of token IDs.

### **2. Embedding Layer**
The `EmbeddingLayer` maps token IDs to dense vector representations.  
**Features**:
- Learnable embedding matrix initialized with random values.
- Provides embeddings for input token IDs.

### **3. Transformer Block**
The `TransformerBlock` is the core of the GPT architecture.  
**Features**:
- **Multi-Head Attention**: Computes attention weights and combines token relationships.
- **Feed-Forward Network**: Applies non-linear transformations.
- **Residual Connections and Layer Normalization**: Stabilizes and enhances training.
- Integrated with logging for debugging and execution tracking.

#### **Logging**
The project now includes a logging system that tracks the execution of major steps and outputs them to:
- The console.
- Log files in the `logs/` directory (e.g., `logs/transformer_block.log`).

---

## **How to Run**

1. **Compile Tests**:
   Example for Transformer Block test:
   ```bash
   mkdir -p logs
   g++ -Iinclude -I/usr/include/eigen3 -o test_transformer tests/test_transformer_block.cpp src/TransformerBlock.cpp src/Logger.cpp
   ./test_transformer
   ```

2. **Expected Logs**:
   Logs provide step-by-step insights into the computation process, such as:
   ```
   [INFO] Initializing TransformerBlock
   [DEBUG] Multi-head attention parameters initialized
   [INFO] Starting forward pass of TransformerBlock
   [DEBUG] Computed Q, K, V matrices
   [DEBUG] Performing scaled dot-product attention
   [DEBUG] Applied first feed-forward layer and ReLU activation
   ```

---

## **Next Steps**

1. Integrate the `Tokenizer`, `EmbeddingLayer`, and `TransformerBlock` into a unified `GPTModel` class.
2. Add forward and training logic for the GPT model.
3. Design and implement tests with sample text input.

