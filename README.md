Here’s a detailed **README.md** file for your project documentation, covering everything we've done so far:

---

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
The `Tokenizer` class is responsible for breaking raw text into smaller parts (tokens) and converting them into integers for model processing.

#### **Features**
- **Vocabulary Building**: Create a vocabulary from a corpus of text.
- **Tokenization**: Convert input text into a sequence of token IDs.

#### **File Locations**
- Header: `include/Tokenizer.h`
- Implementation: `src/Tokenizer.cpp`

#### **Code Example**
```cpp
#include "Tokenizer.h"
#include <iostream>

int main() {
    Tokenizer tokenizer;
    tokenizer.build_vocab({"hello world", "this is a test", "hello again"});

    auto tokens = tokenizer.tokenize("hello world");
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### **Compile and Test**
```bash
g++ -Iinclude -o test_tokenizer tests/test_tokenizer.cpp src/Tokenizer.cpp
./test_tokenizer
```

#### **Expected Output**
```
0 1
```
Where `0` corresponds to `hello` and `1` corresponds to `world` (vocabulary IDs).

---

## **Next Steps**

### **Planned Components**
1. **EmbeddingLayer**:
   - Converts token IDs into dense vector representations.
2. **TransformerBlock**:
   - Implements multi-head attention and feed-forward networks.
3. **GPTModel**:
   - Combines embeddings and multiple transformer blocks to form the GPT architecture.


