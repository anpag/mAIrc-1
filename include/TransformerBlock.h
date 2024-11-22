#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <Eigen/Dense>
#include <vector>
#include "Logger.h"

class TransformerBlock {
private:
    int embedding_dim;
    int num_heads;
    int feedforward_dim;

    // Parameters for multi-head attention
    Eigen::MatrixXd W_q, W_k, W_v, W_o;

    // Parameters for feed-forward network
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2;

    Logger logger;  // Add logger instance

    // Helper methods
    Eigen::MatrixXd scaled_dot_product_attention(
        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V);

public:
    TransformerBlock(int embedding_dim, int num_heads, int feedforward_dim);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
};

#endif
