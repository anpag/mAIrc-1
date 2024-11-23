#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>


class Tokenizer {
private:
    std::unordered_map<std::string, int> vocab; // Maps words to token IDs
    std::string delimiter; // Delimiter for splitting text



public:
    Tokenizer();
    explicit Tokenizer(const std::string& delimiter);

    std::vector<int> tokenize (const std::string& text); // Converts text to tokens
    void build_vocab(const std::vector<std::string>& corpus); // Builds vocabulary from corpus
};

#endif