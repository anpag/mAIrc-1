#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

class Loss {
public:
    // Compute cross-entropy loss
    static double cross_entropy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);

    // Compute gradient of cross-entropy loss
    static Eigen::MatrixXd cross_entropy_gradient(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);
};

#endif
