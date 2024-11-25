#include "EmbeddingLayer.h"
#include "Logger.h"

EmbeddingLayer::EmbeddingLayer(int vocab_size, int embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    Logger::get_instance().log("Initializing EmbeddingLayer", LogLevel::INFO);
    embedding_matrix = Eigen::MatrixXd::Random(vocab_size, embedding_dim);
    Logger::get_instance().log("Embedding matrix initialized with dimensions: " +
                               std::to_string(vocab_size) + "x" +
                               std::to_string(embedding_dim), LogLevel::DEBUG);
}

Eigen::MatrixXd EmbeddingLayer::get_embeddings(const std::vector<int>& token_ids) {
    Logger::get_instance().log("Fetching embeddings for token IDs", LogLevel::DEBUG);
    Eigen::MatrixXd result(token_ids.size(), embedding_dim);

    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            result.row(i) = embedding_matrix.row(token_id);
        } else {
            result.row(i).setZero();
            Logger::get_instance().log("Invalid token ID " + std::to_string(token_id) +
                                       ". Using zero vector.", LogLevel::WARNING);
        }
    }

    Logger::get_instance().log("Embedding lookup completed for sequence of length: " +
                               std::to_string(token_ids.size()), LogLevel::DEBUG);

    return result;
}
