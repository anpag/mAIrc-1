#ifndef METRICS_H
#define METRICS_H

#include <Eigen/Dense>

class Metrics {
public:
    // Compute accuracy
    static double accuracy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);

    // Compute perplexity
    static double perplexity(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);
};

#endif
