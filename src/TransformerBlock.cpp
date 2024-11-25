#include "TransformerBlock.h"
#include "Logger.h"
#include <cmath>

TransformerBlock::TransformerBlock(int embedding_dim, int num_heads, int feedforward_dim)
    : embedding_dim(embedding_dim), num_heads(num_heads), feedforward_dim(feedforward_dim) {
    Logger::get_instance().log("Initializing TransformerBlock", LogLevel::INFO);

    // Initialize parameters for multi-head attention
    W_q = Eigen::MatrixXd::Random(embedding_dim, embedding_dim) * 0.01;
    W_k = Eigen::MatrixXd::Random(embedding_dim, embedding_dim) * 0.01;
    W_v = Eigen::MatrixXd::Random(embedding_dim, embedding_dim) * 0.01;
    W_o = Eigen::MatrixXd::Random(embedding_dim, embedding_dim) * 0.01;
    Logger::get_instance().log("Initialized multi-head attention parameters", LogLevel::DEBUG);

    // Initialize parameters for feed-forward network
    W1 = Eigen::MatrixXd::Random(feedforward_dim, embedding_dim) * 0.01;
    W2 = Eigen::MatrixXd::Random(embedding_dim, feedforward_dim) * 0.01;
    b1 = Eigen::VectorXd::Random(feedforward_dim);
    b2 = Eigen::VectorXd::Random(embedding_dim);
    Logger::get_instance().log("Initialized feed-forward network parameters", LogLevel::DEBUG);
}

Eigen::MatrixXd TransformerBlock::scaled_dot_product_attention(
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V) {
    Logger::get_instance().log("Performing scaled dot-product attention", LogLevel::DEBUG);

    Eigen::MatrixXd scores = Q * K.transpose() / std::sqrt(static_cast<double>(embedding_dim));
    Eigen::MatrixXd exp_scores = scores.array().exp();
    Eigen::VectorXd row_sums = exp_scores.rowwise().sum();
    Eigen::MatrixXd attention_weights = exp_scores.array().colwise() / row_sums.array();
    Logger::get_instance().log("Computed attention weights", LogLevel::DEBUG);

    return attention_weights * V;
}

Eigen::MatrixXd TransformerBlock::forward(const Eigen::MatrixXd& input) {
    Logger::get_instance().log("Starting forward pass of TransformerBlock", LogLevel::INFO);

    // Multi-head attention
    Eigen::MatrixXd Q = input * W_q;
    Eigen::MatrixXd K = input * W_k;
    Eigen::MatrixXd V = input * W_v;
    Logger::get_instance().log("Computed Q, K, V matrices", LogLevel::DEBUG);

    Eigen::MatrixXd attention_output = scaled_dot_product_attention(Q, K, V);
    Eigen::MatrixXd multi_head_output = attention_output * W_o;
    Logger::get_instance().log("Computed multi-head attention output", LogLevel::DEBUG);

    // Residual connection and feed-forward network
    Eigen::MatrixXd residual_output = multi_head_output + input;
    Logger::get_instance().log("Added residual connection to multi-head attention output", LogLevel::DEBUG);

    Eigen::MatrixXd hidden = (W1 * residual_output.transpose()).colwise() + b1;
    hidden = hidden.array().max(0.0);  // ReLU activation
    Logger::get_instance().log("Applied ReLU activation in feed-forward network", LogLevel::DEBUG);

    Eigen::MatrixXd output = (W2 * hidden).colwise() + b2;
    Logger::get_instance().log("Computed output of feed-forward network", LogLevel::DEBUG);

    return output.transpose() + residual_output;  // Residual connection
}
