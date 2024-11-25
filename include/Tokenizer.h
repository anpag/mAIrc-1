#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

class Tokenizer {
private:
    std::unordered_map<std::string, int> vocab; // Token-to-ID mapping
    std::string delimiter;                      // Delimiter for tokenization

public:
    // Constructor
    explicit Tokenizer(const std::string& delim = " ");

    // Build vocabulary from a corpus of sentences
    void build_vocab(const std::vector<std::string>& corpus);

    // Tokenize a given input string into token IDs
    std::vector<int> tokenize(const std::string& text) const;
};

#endif
