#include "Metrics.h"
#include "Logger.h"
#include <cmath>

double Metrics::accuracy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    Logger::get_instance().log("Calculating accuracy", LogLevel::INFO);

    int correct = 0;
    int total = predictions.rows();

    for (int i = 0; i < total; ++i) {
        int predicted_index;
        int target_index;

        // Find the index of the maximum probability in the prediction
        predictions.row(i).maxCoeff(&predicted_index);

        // Find the index of the "1" in the one-hot encoded target
        targets.row(i).maxCoeff(&target_index);

        if (predicted_index == target_index) {
            ++correct;
        }
    }

    double accuracy = static_cast<double>(correct) / total;
    Logger::get_instance().log("Accuracy: " + std::to_string(accuracy), LogLevel::INFO);

    return accuracy;
}

double Metrics::perplexity(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    const double epsilon = 1e-12;
    double total_log_prob = 0.0;
    int total = predictions.rows();

    for (int i = 0; i < total; ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            if (targets(i, j) > 0) {
                total_log_prob += std::log(predictions(i, j) + epsilon);
            }
        }
    }

    double avg_log_prob = total_log_prob / total;
    return std::exp(-avg_log_prob);
}

