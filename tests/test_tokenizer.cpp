#include "../include/Tokenizer.h"
#include <iostream>

int main() {
    Tokenizer tokenizer;
    tokenizer.build_vocab({"hello world", "this is a test", "hello again"});

    auto tokens = tokenizer.tokenize("hello world");
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout <<std::endl;

    return 0;
}