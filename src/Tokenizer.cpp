#include "Tokenizer.h"
#include "Logger.h"

Tokenizer::Tokenizer() : delimiter(" ") {
    Logger::get_instance().log("Tokenizer initialzied with default delimiter: ' '", LogLevel::INFO);
}

Tokenizer::Tokenizer(const std::string& delimiter) : delimiter(delimiter) {
        Logger::get_instance().log("Tokenizer initialzied with custom delimiter: " + delimiter, LogLevel::INFO);
}

// Tokenize text into a sequence of integers (IDs)
std::vector<int> Tokenizer::tokenize(const std::string& text) {
    Logger::get_instance().log("Tokenizing text: " + text, LogLevel::INFO);
    std::vector<int> tokens;
    std::stringstream ss(text);
    std::string word;

    while (std::getline(ss, word, delimiter[0])) {
        if (vocab.find(word) != vocab.end()) {
            tokens.push_back(vocab[word]); // Convert word to token ID
        } else {
            tokens.push_back(-1); // Unknown token
        }
    }

    return tokens;
}

//Build vocabualry from a list of senteces
void Tokenizer::build_vocab(const std::vector<std::string>& corpus) {
    Logger::get_instance().log("Building vocabulary from corpus: ", LogLevel::INFO);
    int id = 0;
    for (const auto& sentence : corpus) {
        std::stringstream ss(sentence);
        std::string word;
        while (std::getline(ss, word, delimiter[0])) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = id++; // Assign unique ID to each word
            }
        }
    }
}

