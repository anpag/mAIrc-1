#include "GPTModel.h"
#include "Logger.h"
#include <cmath>

// Helper function for softmax
Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits) {
    Eigen::MatrixXd exp_logits = logits.array().exp();
    Eigen::VectorXd row_sums = exp_logits.rowwise().sum();
    return exp_logits.array().colwise() / row_sums.array();
}

std::string matrix_dims(const Eigen::MatrixXd& matrix, const std::string& name) {
    std::ostringstream oss;
    oss << name << ": " << matrix.rows() << "x" << matrix.cols();
    return oss.str();
}

// Constructor
GPTModel::GPTModel(int vocab_size, int embedding_dim, int num_layers, int num_heads, int feedforward_dim, double learning_rate)
    : tokenizer(), embedding_layer(vocab_size, embedding_dim), learning_rate(learning_rate) {
    Logger::get_instance().log("Initializing GPTModel", LogLevel::INFO);

    // Initialize transformer layers
    for (int i = 0; i < num_layers; ++i) {
        layers.emplace_back(TransformerBlock(embedding_dim, num_heads, feedforward_dim));
        Logger::get_instance().log("Added TransformerBlock " + std::to_string(i + 1), LogLevel::DEBUG);
    }

    // Initialize output layer weights and bias
    output_weights = Eigen::MatrixXd::Random(vocab_size, embedding_dim);
    output_bias = Eigen::VectorXd::Random(vocab_size);

    Logger::get_instance().log("Output layer initialized", LogLevel::DEBUG);
}

// Forward pass with embeddings
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GPTModel::forward_with_embeddings(const std::string& input_text) {
    Logger::get_instance().log("Starting GPTModel forward pass", LogLevel::INFO);

    // Step 1: Tokenize the input text
    auto tokens = tokenizer.tokenize(input_text);
    Logger::get_instance().log("Tokenized input text: " + input_text, LogLevel::DEBUG);

    // Step 2: Get embeddings for the tokens
    auto embeddings = embedding_layer.get_embeddings(tokens);
    Logger::get_instance().log("Computed embeddings for tokens", LogLevel::DEBUG);

    // Step 3: Pass embeddings through transformer blocks
    Eigen::MatrixXd output = embeddings;
    for (size_t i = 0; i < layers.size(); ++i) {
        Logger::get_instance().log("Passing through TransformerBlock " + std::to_string(i + 1), LogLevel::DEBUG);
        output = layers[i].forward(output);
    }

    // Step 4: Apply output layer
    Eigen::MatrixXd logits = (output * output_weights.transpose()).rowwise() + output_bias.transpose();
    Logger::get_instance().log("Computed logits using output layer", LogLevel::DEBUG);

    // Step 5: Apply softmax to logits
    Eigen::MatrixXd probabilities = softmax(logits);
    Logger::get_instance().log("Applied softmax to logits to compute probabilities", LogLevel::DEBUG);

    Logger::get_instance().log("Completed GPTModel forward pass", LogLevel::INFO);
    return {embeddings, probabilities};
}

// Train method
double GPTModel::train(const std::string& input_text, const Eigen::MatrixXd& targets) {
    Logger::get_instance().log("Starting GPTModel training pass", LogLevel::INFO);

    // Forward pass
    auto [embeddings, predictions] = forward_with_embeddings(input_text);

    // Compute loss
    double loss = Loss::cross_entropy(predictions, targets);
    Logger::get_instance().log("Computed loss: " + std::to_string(loss), LogLevel::DEBUG);

    // Backward pass (compute gradients)
    Eigen::MatrixXd gradients = Loss::cross_entropy_gradient(predictions, targets);
    Logger::get_instance().log(matrix_dims(gradients, "gradients"), LogLevel::DEBUG);

    // Step 1: Backpropagate through the output layer
    Eigen::MatrixXd embedding_gradients = gradients * output_weights; // Gradients in embedding space

    // Step 2: Update output layer weights and biases
    output_weights -= learning_rate * gradients.transpose() * embeddings;  // Match dimensions
    output_bias -= learning_rate * gradients.colwise().sum();

    Logger::get_instance().log(matrix_dims(output_weights, "output_weights"), LogLevel::DEBUG);
    Logger::get_instance().log(matrix_dims(output_bias, "output_bias"), LogLevel::DEBUG);

    // Optional: Log embedding_gradients dimensions for future backpropagation
    Logger::get_instance().log(matrix_dims(embedding_gradients, "embedding_gradients"), LogLevel::DEBUG);

    Logger::get_instance().log("Updated model weights and biases", LogLevel::DEBUG);
    return loss;
}
