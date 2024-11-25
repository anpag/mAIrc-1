#include "Loss.h"
#include "Logger.h"
#include <cmath>

double Loss::cross_entropy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    Logger::get_instance().log("Calculating cross-entropy loss", LogLevel::DEBUG);

    const double epsilon = 1e-12; // Avoid log(0)
    Eigen::MatrixXd clipped_predictions = predictions.array().max(epsilon).min(1.0 - epsilon);
    Eigen::MatrixXd log_predictions = clipped_predictions.array().log();
    
    double loss = -(targets.array() * log_predictions.array()).sum() / targets.rows();

    Logger::get_instance().log("Clipped predictions min: " + std::to_string(clipped_predictions.minCoeff()) +
                               ", max: " + std::to_string(clipped_predictions.maxCoeff()), LogLevel::DEBUG);
    Logger::get_instance().log("Cross-entropy loss: " + std::to_string(loss), LogLevel::INFO);

    return loss;
}

Eigen::MatrixXd Loss::cross_entropy_gradient(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    Logger::get_instance().log("Calculating cross-entropy gradient", LogLevel::DEBUG);

    Eigen::MatrixXd gradients = (predictions.array() - targets.array()) / targets.rows();
    Logger::get_instance().log("Cross-entropy gradient calculated", LogLevel::DEBUG);

    return gradients;
}
