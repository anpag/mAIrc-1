#include "Loss.h"
#include <cmath>

double Loss::cross_entropy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    double loss = 0.0;
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            if (targets(i, j) > 0) {
                loss -= targets(i, j) * std::log(predictions(i, j) + 1e-9);  // Add epsilon for numerical stability
            }
        }
    }
    return loss / predictions.rows();
}

Eigen::MatrixXd Loss::cross_entropy_gradient(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    // Return gradients normalized by batch size
    return (predictions - targets);  // Gradients remain the same
}