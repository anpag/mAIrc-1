#include "EmbeddingLayer.h"
#include "Logger.h"


EmbeddingLayer::EmbeddingLayer(int vocab_size, int embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    Logger::get_instance().log("Initializing EmbeddingLayer", LogLevel::INFO);
    embedding_matrix = Eigen::MatrixXd::Random(vocab_size, embedding_dim);
    Logger::get_instance().log("Initializing EmbeddingLayer with vocab size: " +
                               std::to_string(vocab_size) + " and embedding dimension: " +
                               std::to_string(embedding_dim), LogLevel::INFO);
}

Eigen::MatrixXd EmbeddingLayer::get_embeddings(const std::vector<int>& token_ids) {
    Logger::get_instance().log("Fetching embeddings for token IDs", LogLevel::INFO);
    Eigen::MatrixXd result(token_ids.size(), embedding_dim);

    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            result.row(i) = embedding_matrix.row(token_id);
            Logger::get_instance().log("Token ID " + std::to_string(token_id) + " mapped to embedding", LogLevel::INFO);
        } else {
            result.row(i).setZero();
            Logger::get_instance().log("Unknown token ID " + std::to_string(token_id) + " mapped to zero vector", LogLevel::WARNING);
        }
    }

    return result;
}
