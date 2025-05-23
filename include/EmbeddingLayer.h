#ifndef EMBEDDING_LAYER_H
#define EMBEDDING_LAYER_H

#include <vector>
#include <Eigen/Dense>

class EmbeddingLayer {
private:
    Eigen::MatrixXd embedding_matrix; // The embedding matrix
    int vocab_size;                   // Number of tokens in vocabulary
    int embedding_dim;                // Dimension of each embedding vector

public:
    // Constructor
    EmbeddingLayer(int vocab_size, int embedding_dim);

    // Retrieve embeddings for a sequence of token IDs
    Eigen::MatrixXd get_embeddings(const std::vector<int>& token_ids);
};

#endif
