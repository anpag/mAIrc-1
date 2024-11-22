#include "EmbeddingLayer.h"
#include "Logger.h"

static Logger logger("logs/embedding_layer.log", LogLevel::DEBUG);

EmbeddingLayer::EmbeddingLayer(int vocab_size, int embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    logger.log("Initializing EmbeddingLayer", LogLevel::INFO);
    embedding_matrix = Eigen::MatrixXd::Random(vocab_size, embedding_dim);
    logger.log("Embedding matrix initialized with random values", LogLevel::DEBUG);
}

Eigen::MatrixXd EmbeddingLayer::get_embeddings(const std::vector<int>& token_ids) {
    logger.log("Fetching embeddings for token IDs", LogLevel::DEBUG);
    Eigen::MatrixXd result(token_ids.size(), embedding_dim);

    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            result.row(i) = embedding_matrix.row(token_id);
            logger.log("Token ID " + std::to_string(token_id) + " mapped to embedding", LogLevel::DEBUG);
        } else {
            result.row(i).setZero();
            logger.log("Unknown token ID " + std::to_string(token_id) + " mapped to zero vector", LogLevel::WARNING);
        }
    }

    return result;
}
