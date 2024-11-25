#ifndef GPT_MODEL_H
#define GPT_MODEL_H

#include "Tokenizer.h"
#include "EmbeddingLayer.h"
#include "TransformerBlock.h"
#include "Loss.h"
#include <Eigen/Dense>
#include <vector>

class GPTModel {
private:
    Tokenizer tokenizer;                  // Tokenizer for text preprocessing
    EmbeddingLayer embedding_layer;      // Embedding layer
    std::vector<TransformerBlock> layers; // Transformer blocks
    Eigen::MatrixXd output_weights;      // Output layer weights
    Eigen::VectorXd output_bias;         // Output layer bias
    double learning_rate;                // Learning rate for optimization

public:
    GPTModel(int vocab_size, int embedding_dim, int num_layers, int num_heads, int feedforward_dim, double learning_rate = 0.001);

    Eigen::MatrixXd forward(const std::string& input_text);
    double train(const std::string& input_text, const Eigen::MatrixXd& targets);
    Tokenizer& get_tokenizer() { return tokenizer; }
};

#endif
