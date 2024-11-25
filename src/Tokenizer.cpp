#include "Tokenizer.h"
#include "Logger.h"

// Constructor
Tokenizer::Tokenizer(const std::string& delim) : delimiter(delim) {
    Logger::get_instance().log("Tokenizer initialized with delimiter: '" + delimiter + "'", LogLevel::INFO);
}

// Build vocabulary from a corpus of sentences
void Tokenizer::build_vocab(const std::vector<std::string>& corpus) {
    Logger::get_instance().log("Building vocabulary from corpus", LogLevel::INFO);

    int id = 0;
    for (const auto& sentence : corpus) {
        std::istringstream stream(sentence);
        std::string word;
        while (std::getline(stream, word, delimiter[0])) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = id++;
                Logger::get_instance().log("Added word to vocab: '" + word + "' with ID: " + std::to_string(id - 1), LogLevel::DEBUG);
            }
        }
    }

    Logger::get_instance().log("Vocabulary built with " + std::to_string(vocab.size()) + " unique tokens", LogLevel::INFO);
}

// Tokenize a given input string into token IDs
std::vector<int> Tokenizer::tokenize(const std::string& text) const {
    Logger::get_instance().log("Tokenizing input text: '" + text + "'", LogLevel::DEBUG);

    std::vector<int> token_ids;
    std::istringstream stream(text);
    std::string word;
    while (std::getline(stream, word, delimiter[0])) {
        auto it = vocab.find(word);
        int token_id = (it != vocab.end()) ? it->second : -1;
        token_ids.push_back(token_id);

        if (token_id == -1) {
            Logger::get_instance().log("Unknown token: '" + word + "', mapped to -1", LogLevel::WARNING);
        } else {
            Logger::get_instance().log("Token: '" + word + "' mapped to ID: " + std::to_string(token_id), LogLevel::DEBUG);
        }
    }

    Logger::get_instance().log("Tokenization completed. Total tokens: " + std::to_string(token_ids.size()), LogLevel::INFO);
    return token_ids;
}
