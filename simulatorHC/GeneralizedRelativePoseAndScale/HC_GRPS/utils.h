#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#define EIGEN_USE_MKL_ALL
#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <torch/torch.h>

Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd & M);
Eigen::MatrixXd pseudoInverseViaLU(const Eigen::MatrixXd& matrix);
// Eigen::MatrixXd pinv(const Eigen::MatrixXd& M);

Eigen::Matrix<double, 8, 9> pinv(const Eigen::Matrix<double, 9, 8>& M);
Eigen::MatrixXd skew_sym(Eigen::VectorXd & a);
Eigen::MatrixXd getRfromq(Eigen::VectorXd& q);
Eigen::MatrixXd dRdq1(Eigen::VectorXd& q);
Eigen::MatrixXd dRdq2(Eigen::VectorXd& q);
Eigen::MatrixXd dRdq3(Eigen::VectorXd& q);
Eigen::MatrixXd dRdq4(Eigen::VectorXd& q);


Eigen::Matrix3d skew_sym(Eigen::Vector3d & a);
Eigen::Matrix3d getRfromq(Eigen::Vector4d& q);
Eigen::Matrix3d dRdq1(Eigen::Vector4d& q);
Eigen::Matrix3d dRdq2(Eigen::Vector4d& q);
Eigen::Matrix3d dRdq3(Eigen::Vector4d& q);
Eigen::Matrix3d dRdq4(Eigen::Vector4d& q);


Eigen::Matrix3d skew_sym(const Eigen::Vector3d & a);
Eigen::Matrix3d getRfromq(const Eigen::Vector4d& q);

Eigen::Matrix3d dRdq1(const Eigen::Vector4d& q);
Eigen::Matrix3d dRdq2(const Eigen::Vector4d& q);
Eigen::Matrix3d dRdq3(const Eigen::Vector4d& q);
Eigen::Matrix3d dRdq4(const Eigen::Vector4d& q);


Eigen::MatrixXd getMatfromTensor(const torch::Tensor& x_tensor);
torch::Tensor eigen_matrix_to_torch_tensor(const Eigen::MatrixXd& matrix);


template<typename MatrixType>
MatrixType getMatFromTensor(const torch::Tensor& x_tensor) {
    // Ensure the tensor is on CPU and is of the correct scalar type
    auto tensor_cpu = x_tensor.to(torch::kCPU, torch::CppTypeToScalarType<double>()).contiguous();

    // Get the dimensions of the tensor
    int rows = tensor_cpu.size(0);
    int cols = tensor_cpu.size(1);

    // Create an Eigen matrix with dynamic size and row major format
    MatrixType matrix(rows, cols);

    // Copy the data from the PyTorch tensor to the Eigen matrix
    // std::memcpy(matrix.data(), tensor_cpu.data_ptr<double>(), sizeof(double) * rows * cols);

    // Copy the data from the PyTorch tensor to the Eigen matrix in a column-wise manner
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            matrix(row, col) = tensor_cpu.index({row, col}).item<double>();
        }
    }

    return matrix;
};
