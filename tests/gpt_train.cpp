#include "../include/GPTModel.h"
#include "../include/Logger.h"
#include "../include/Metrics.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring> // For strcmp
#include <json.hpp> // Include JSON library (e.g., nlohmann/json.hpp)


// Function to clean text
std::string clean_text(const std::string& text) {
    std::string cleaned;
    
    // Remove unwanted characters and normalize case
    for (char ch : text) {
        if (std::isalnum(ch) || std::isspace(ch)) {
            cleaned += std::tolower(ch);
        }
    }

    // Normalize whitespace
    std::string normalized;
    bool in_whitespace = false;
    for (char ch : cleaned) {
        if (std::isspace(ch)) {
            if (!in_whitespace) {
                normalized += ' ';
                in_whitespace = true;
            }
        } else {
            normalized += ch;
            in_whitespace = false;
        }
    }
    
    return normalized;
}

// JSON Parsing Library Alias
using json = nlohmann::json;

std::vector<std::string> load_text_from_json(const std::string& file_path, int max_entries) {
    std::vector<std::string> corpus;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        exit(1);
    }

    std::string line;
    int count = 0;

    while (std::getline(file, line) && count < max_entries) {
        try {
            json parsed_line = json::parse(line);
            if (parsed_line.contains("text")) {
                std::string cleaned_text = clean_text(parsed_line["text"].get<std::string>());
                std::cout << "Cleaned text: " << cleaned_text << std::endl;
                corpus.push_back(cleaned_text);
                ++count;
            }
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing JSON line: " << e.what() << std::endl;
        }
    }
    
    file.close();
    return corpus;
}

int main(int argc, char* argv[]) {
    LogLevel log_level = LogLevel::INFO;

    // Default model parameters
    int vocab_size = 141006;
    int embedding_dim = 16;
    int num_heads = 2;
    int feedforward_dim = 32;
    double learning_rate = 0.001;
    int num_epochs = 10;
    int num_layers = 2;
    int max_entries = 1000; // Number of entries from JSON file
    std::string json_file = "data.json";

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--log_level") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "DEBUG") == 0) log_level = LogLevel::DEBUG;
            else if (strcmp(argv[i + 1], "WARNING") == 0) log_level = LogLevel::WARNING;
            else if (strcmp(argv[i + 1], "ERROR") == 0) log_level = LogLevel::ERROR;
            ++i;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = std::atoi(argv[i + 1]);
            if (num_epochs <= 0) {
                std::cerr << "Invalid value for --epochs. Must be a positive integer.\n";
                return 1;
            }
            ++i;
         } else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
            vocab_size = std::atoi(argv[i + 1]);
            if (vocab_size <= 0) {
                std::cerr << "Invalid value for --vocab-size. Must be a positive integer.\n";
                return 1;
            }
            ++i;
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            num_layers = std::atoi(argv[i + 1]);
            if (num_layers <= 0) {
                std::cerr << "Invalid value for --layers. Must be a positive integer.\n";
                return 1;
            }
            ++i;
        } else if (strcmp(argv[i], "--json_file") == 0 && i + 1 < argc) {
            json_file = argv[i + 1];
            ++i;
        } else if (strcmp(argv[i], "--max_entries") == 0 && i + 1 < argc) {
            max_entries = std::atoi(argv[i + 1]);
            if (max_entries <= 0) {
                std::cerr << "Invalid value for --max_entries. Must be a positive integer.\n";
                return 1;
            }
            ++i;
        }
    }

    Logger& logger = Logger::get_instance("logs/gpt_training_with_metrics.log", log_level);
    logger.log("Log level set to " + std::to_string(static_cast<int>(log_level)), LogLevel::INFO);

    // Load text data from JSON file
    logger.log("Loading data from JSON file: " + json_file, LogLevel::INFO);
    std::vector<std::string> corpus = load_text_from_json(json_file, max_entries);
    logger.log("Loaded " + std::to_string(corpus.size()) + " entries from JSON.", LogLevel::INFO);

    // Initialize GPTModel
    logger.log("Initializing GPTModel", LogLevel::INFO);
    GPTModel model(vocab_size, embedding_dim, num_layers, num_heads, feedforward_dim, learning_rate);

    // Build vocabulary for the tokenizer
    model.get_tokenizer().build_vocab(corpus);
    logger.log("Vocabulary built with " + std::to_string(vocab_size) + " unique tokens.", LogLevel::INFO);

    // Prepare training dataset
    std::vector<std::pair<std::string, Eigen::MatrixXd>> dataset;
    for (const auto& text : corpus) {
        auto tokens = model.get_tokenizer().tokenize(text);
        Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(tokens.size(), vocab_size);

        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] >= 0 && tokens[i] < vocab_size) {
                targets(i, tokens[i]) = 1; // Create one-hot target
            }
        }
        dataset.emplace_back(text, targets);
    }

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        logger.log("Starting epoch " + std::to_string(epoch + 1), LogLevel::INFO);

        double total_loss = 0.0;
        double total_accuracy = 0.0;
        double total_perplexity = 0.0;

        for (const auto& [input_text, targets] : dataset) {
            double loss = model.train(input_text, targets);
            total_loss += loss;

            // Evaluate
            auto predictions = model.forward(input_text);
            total_accuracy += Metrics::accuracy(predictions, targets);
            total_perplexity += Metrics::perplexity(predictions, targets);
        }

        logger.log("Epoch " + std::to_string(epoch + 1) +
                   " - Loss: " + std::to_string(total_loss / dataset.size()) +
                   ", Accuracy: " + std::to_string(total_accuracy / dataset.size()) +
                   ", Perplexity: " + std::to_string(total_perplexity / dataset.size()), LogLevel::INFO);
    }

    logger.log("Training completed successfully.", LogLevel::INFO);
    return 0;
}
