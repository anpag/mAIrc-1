#include "GPTModel.h"
#include "Logger.h"
#include <cmath>

Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits) {
    const double epsilon = 1e-12; // Avoid division by zero
    Eigen::VectorXd row_max = logits.rowwise().maxCoeff();
    Eigen::MatrixXd stable_logits = logits.array().colwise() - row_max.array();

    Eigen::MatrixXd exp_logits = stable_logits.array().exp();
    Eigen::VectorXd row_sums = exp_logits.rowwise().sum().array().max(epsilon); // Clamp row sums

    Eigen::MatrixXd probabilities = exp_logits.array().colwise() / row_sums.array();

    Logger::get_instance().log("Softmax output min: " + std::to_string(probabilities.minCoeff()) +
                               ", max: " + std::to_string(probabilities.maxCoeff()), LogLevel::DEBUG);

    return probabilities;
}



GPTModel::GPTModel(int vocab_size, int embedding_dim, int num_layers, int num_heads, int feedforward_dim, double learning_rate)
    : embedding_layer(vocab_size, embedding_dim), learning_rate(learning_rate) {
    Logger::get_instance().log("Initializing GPTModel", LogLevel::INFO);
    for (int i = 0; i < num_layers; ++i) {
        layers.emplace_back(TransformerBlock(embedding_dim, num_heads, feedforward_dim));
        Logger::get_instance().log("Added TransformerBlock " + std::to_string(i + 1), LogLevel::DEBUG);
    }
    output_weights = Eigen::MatrixXd::Random(vocab_size, embedding_dim) * 0.01; // Small values
    output_bias = Eigen::VectorXd::Zero(vocab_size);
    Logger::get_instance().log("Output layer initialized", LogLevel::DEBUG);
}

Eigen::MatrixXd GPTModel::forward(const std::string& input_text) {
    Logger::get_instance().log("Starting forward pass", LogLevel::INFO);
    auto tokens = tokenizer.tokenize(input_text);
    Eigen::MatrixXd embeddings = embedding_layer.get_embeddings(tokens);
    for (size_t i = 0; i < layers.size(); ++i) {
        embeddings = layers[i].forward(embeddings);
        Logger::get_instance().log("Passed through TransformerBlock " + std::to_string(i + 1), LogLevel::DEBUG);
    }
    Eigen::MatrixXd logits = (embeddings * output_weights.transpose()).rowwise() + output_bias.transpose();
    Logger::get_instance().log("Computed logits", LogLevel::DEBUG);
    return softmax(logits);
}

double GPTModel::train(const std::string& input_text, const Eigen::MatrixXd& targets) {
    Logger::get_instance().log("Starting training pass", LogLevel::INFO);
    Eigen::MatrixXd predictions = forward(input_text);
    double loss = Loss::cross_entropy(predictions, targets);
    Logger::get_instance().log("Loss: " + std::to_string(loss), LogLevel::INFO);

    Eigen::MatrixXd gradients = Loss::cross_entropy_gradient(predictions, targets);
    Eigen::MatrixXd embeddings = embedding_layer.get_embeddings(tokenizer.tokenize(input_text));

    output_weights -= learning_rate * gradients.transpose() * embeddings;
    output_bias -= learning_rate * gradients.colwise().sum().transpose();
    Logger::get_instance().log("Updated weights and biases", LogLevel::DEBUG);

    return loss;
}
