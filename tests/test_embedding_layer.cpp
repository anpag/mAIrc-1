#include "../include/EmbeddingLayer.h"
#include <iostream>

int main() {
    // Example: Vocabulary size = 10, Embedding dimension = 5
    EmbeddingLayer embedding_layer(10, 5);

    // Example token IDs
    std::vector<int> token_ids = {0, 1, 2, 9, 11};  // Includes an unknown token (11)

    // Get embeddings
    auto embeddings = embedding_layer.get_embeddings(token_ids);

    // Print embeddings
    std::cout << "Embeddings:\n" << embeddings << std::endl;

    return 0;
}
