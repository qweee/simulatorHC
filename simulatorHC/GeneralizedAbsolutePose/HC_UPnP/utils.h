#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#define EIGEN_USE_MKL_ALL
#pragma once
#include <Eigen/Dense>
#include<Eigen/Core>
#include <vector>

typedef Eigen::Matrix<double, 11, 11> M11;
typedef Eigen::Matrix<double, 5, 4> M54;
typedef Eigen::Matrix<double, 4, 5> M45;
typedef Eigen::Matrix<double, 5, 1> M51;
typedef Eigen::Matrix<double, 4, 1> M41;

M51 UPnP_polys(const Eigen::Vector4d & q, const M11 & M);
M54 Jacobian(const Eigen::Vector4d & q, const M11 & M);
M45 PseudoInverse(const M54 && M);
Eigen::MatrixXd pinv(const Eigen::MatrixXd& M);
