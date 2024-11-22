#include "../include/TransformerBlock.h"
#include <iostream>

int main() {
    // Example: Embedding dimension = 8, Heads = 2, Feedforward dimension = 16
    TransformerBlock transformer(8, 2, 16);

    // Example input: 4 tokens, embedding dimension = 8
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 8);

    // Perform forward pass
    Eigen::MatrixXd output = transformer.forward(input);

    // Print output
    std::cout << "Transformer Block Output:\n" << output << std::endl;

    return 0;
}
