#include "utils.h"
#include <iostream>


Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd & M)
{
	// compute Pseudo Inverse

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
	// Eigen::VectorXd singularValues = svd.singularValues();

	double tolerance = 1e-10;

	// Eigen::MatrixXd pseudoInverse = svd.matrixV() * singularValues.asDiagonal() * svd.matrixU().transpose();
    Eigen::MatrixXd pseudoInverse = svd.matrixV() * Eigen::MatrixXd((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0)).asDiagonal() * svd.matrixU().adjoint();


	return pseudoInverse;
}


Eigen::MatrixXd skew_sym(Eigen::VectorXd & a) {
    assert(a.size() == 3);  // Ensure the input vector has size 3

    Eigen::MatrixXd a_skew(3, 3);
    a_skew << 0, -a(2), a(1),
              a(2), 0, -a(0),
              -a(1), a(0), 0;

    return a_skew;
};

Eigen::Matrix3d skew_sym(Eigen::Vector3d & a) {
    assert(a.size() == 3);  // Ensure the input vector has size 3

    Eigen::Matrix3d a_skew(3, 3);
    a_skew << 0, -a(2), a(1),
              a(2), 0, -a(0),
              -a(1), a(0), 0;

    return a_skew;
};


Eigen::MatrixXd getRfromq(Eigen::VectorXd& q) {


	double q1 = q[0];
    double q2 = q[1];
    double q3 = q[2];
    double q4 = q[3];

    // Eigen::Matrix3d R;
    Eigen::MatrixXd R(3,3);
    R << q1*q1 + q2*q2 - q3*q3 - q4*q4, 2*q2*q3 - 2*q1*q4, 2*q2*q4 + 2*q1*q3,
         2*q2*q3 + 2*q1*q4, q1*q1 - q2*q2 + q3*q3 - q4*q4, 2*q3*q4 - 2*q1*q2,
         2*q2*q4 - 2*q1*q3, 2*q3*q4 + 2*q1*q2, q1*q1 - q2*q2 - q3*q3 + q4*q4;
    
    return R;

};

Eigen::MatrixXd dRdq1(Eigen::VectorXd& q){

    Eigen::Matrix3d grad;

    grad << 2 * q(0), -2 * q(3),  2 * q(2),
             2 * q(3),  2 * q(0), -2 * q(1),
            -2 * q(2),  2 * q(1),  2 * q(0);

    return grad;

};

Eigen::MatrixXd dRdq2(Eigen::VectorXd& q){
    
    Eigen::Matrix3d grad;

    grad << 2 * q(1),  2 * q(2),  2 * q(3),
             2 * q(2), -2 * q(1), -2 * q(0),
             2 * q(3),  2 * q(0), -2 * q(1);

    return grad;
};

Eigen::MatrixXd dRdq3(Eigen::VectorXd& q){
    
    Eigen::Matrix3d grad;

    grad << -2 * q(2),  2 * q(1),  2 * q(0),
             2 * q(1),  2 * q(2),  2 * q(3),
            -2 * q(0),  2 * q(3), -2 * q(2);

    return grad;
};

Eigen::MatrixXd dRdq4(Eigen::VectorXd& q){
    
    Eigen::Matrix3d grad;

    grad << -2 * q(3), -2 * q(0),  2 * q(1),
             2 * q(0), -2 * q(3),  2 * q(2),
             2 * q(1),  2 * q(2),  2 * q(3);

    return grad;
    
};


Eigen::Matrix3d getRfromq(Eigen::Vector4d& q) {


	double q1 = q[0];
    double q2 = q[1];
    double q3 = q[2];
    double q4 = q[3];

    // Eigen::Matrix3d R;
    Eigen::Matrix3d R(3,3);
    R << q1*q1 + q2*q2 - q3*q3 - q4*q4, 2*q2*q3 - 2*q1*q4, 2*q2*q4 + 2*q1*q3,
         2*q2*q3 + 2*q1*q4, q1*q1 - q2*q2 + q3*q3 - q4*q4, 2*q3*q4 - 2*q1*q2,
         2*q2*q4 - 2*q1*q3, 2*q3*q4 + 2*q1*q2, q1*q1 - q2*q2 - q3*q3 + q4*q4;
    
    return R;

};

Eigen::Matrix3d dRdq1(Eigen::Vector4d& q){

    Eigen::Matrix3d grad;

    grad << 2 * q(0), -2 * q(3),  2 * q(2),
             2 * q(3),  2 * q(0), -2 * q(1),
            -2 * q(2),  2 * q(1),  2 * q(0);

    return grad;

};

Eigen::Matrix3d dRdq2(Eigen::Vector4d& q){
    
    Eigen::Matrix3d grad;

    grad << 2 * q(1),  2 * q(2),  2 * q(3),
             2 * q(2), -2 * q(1), -2 * q(0),
             2 * q(3),  2 * q(0), -2 * q(1);

    return grad;
};

Eigen::Matrix3d dRdq3(Eigen::Vector4d& q){
    
    Eigen::Matrix3d grad;

    grad << -2 * q(2),  2 * q(1),  2 * q(0),
             2 * q(1),  2 * q(2),  2 * q(3),
            -2 * q(0),  2 * q(3), -2 * q(2);

    return grad;
};

Eigen::Matrix3d dRdq4(Eigen::Vector4d& q){
    
    Eigen::Matrix3d grad;

    grad << -2 * q(3), -2 * q(0),  2 * q(1),
             2 * q(0), -2 * q(3),  2 * q(2),
             2 * q(1),  2 * q(2),  2 * q(3);

    return grad;
    
};



Eigen::Matrix3d skew_sym(const Eigen::Vector3d & a) {
    assert(a.size() == 3);  // Ensure the input vector has size 3

    Eigen::Matrix3d a_skew(3, 3);
    a_skew << 0, -a(2), a(1),
              a(2), 0, -a(0),
              -a(1), a(0), 0;

    return a_skew;
};


Eigen::Matrix3d getRfromq(const Eigen::Vector4d& q) {


	double q1 = q[0];
    double q2 = q[1];
    double q3 = q[2];
    double q4 = q[3];

    // Eigen::Matrix3d R;
    Eigen::Matrix3d R(3,3);
    R << q1*q1 + q2*q2 - q3*q3 - q4*q4, 2*q2*q3 - 2*q1*q4, 2*q2*q4 + 2*q1*q3,
         2*q2*q3 + 2*q1*q4, q1*q1 - q2*q2 + q3*q3 - q4*q4, 2*q3*q4 - 2*q1*q2,
         2*q2*q4 - 2*q1*q3, 2*q3*q4 + 2*q1*q2, q1*q1 - q2*q2 - q3*q3 + q4*q4;
    
    return R;

};


Eigen::Matrix3d dRdq1(const Eigen::Vector4d& q){

    Eigen::Matrix3d grad;

    grad << 2 * q(0), -2 * q(3),  2 * q(2),
             2 * q(3),  2 * q(0), -2 * q(1),
            -2 * q(2),  2 * q(1),  2 * q(0);

    return grad;

};

Eigen::Matrix3d dRdq2(const Eigen::Vector4d& q){
    
    Eigen::Matrix3d grad;

    grad << 2 * q(1),  2 * q(2),  2 * q(3),
             2 * q(2), -2 * q(1), -2 * q(0),
             2 * q(3),  2 * q(0), -2 * q(1);

    return grad;
};

Eigen::Matrix3d dRdq3(const Eigen::Vector4d& q){
    
    Eigen::Matrix3d grad;

    grad << -2 * q(2),  2 * q(1),  2 * q(0),
             2 * q(1),  2 * q(2),  2 * q(3),
            -2 * q(0),  2 * q(3), -2 * q(2);

    return grad;
};

Eigen::Matrix3d dRdq4(const Eigen::Vector4d& q){
    
    Eigen::Matrix3d grad;

    grad << -2 * q(3), -2 * q(0),  2 * q(1),
             2 * q(0), -2 * q(3),  2 * q(2),
             2 * q(1),  2 * q(2),  2 * q(3);

    return grad;
    
};


Eigen::MatrixXd getMatfromTensor(const torch::Tensor& x_tensor){
    
    torch::Tensor tensor_cpu = x_tensor.to(torch::kCPU).to(torch::kDouble);
    tensor_cpu = tensor_cpu.contiguous();

    // std::cout << "input before mcp: " << tensor_cpu << std::endl;
    // std::cout.flush();
    
    int rows = tensor_cpu.size(0);
    int cols = tensor_cpu.size(1);
	
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(rows, cols);
    std::memcpy(matrix.data(), tensor_cpu.data_ptr<double>(), sizeof(double) * rows * cols);
    
    // std::cout << "input after mcp: " << matrix << std::endl;
    // std::cout.flush();

	return matrix;
};


torch::Tensor eigen_matrix_to_torch_tensor(const Eigen::MatrixXd& matrix) {
    // Get the dimensions of the matrix

    // std::cout<< "out tensor:" << matrix << std::endl;
    // std::cout.flush();
    int rows = matrix.rows();
    int cols = matrix.cols();

    // Create a tensor of the appropriate size
    torch::Tensor tensor = torch::empty({rows, cols}, torch::kDouble);

    // Copy the data from the Eigen matrix to the tensor
    std::memcpy(tensor.data_ptr<double>(), matrix.data(), sizeof(double) * rows * cols);

    return tensor;
};

