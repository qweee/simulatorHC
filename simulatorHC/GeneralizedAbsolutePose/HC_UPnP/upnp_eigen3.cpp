#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#define EIGEN_USE_MKL_ALL
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include "utils.h"
#include <iostream>
#include <chrono>

namespace py = pybind11;

int newton_iter = 5;
double step_size = 0.02;
bool adaptive_flag = true;
bool predictor_flag = true;
double epsilon = 1e-10;


Eigen::Vector4d g(const M11& M_F, const M11& M_G, double t, Eigen::Vector4d q) {
	M54 Jq = Jacobian(q, M_F)*t + Jacobian(q, M_G)*(1 - t);
	M51 t_grad = UPnP_polys(q, M_F) - UPnP_polys(q, M_G);
	return -pinv(Jq.array() + epsilon)*t_grad;
};

Eigen::Vector4d HC_base(const Eigen::Vector4d& _q, const M11& M_F, const M11& M_G)
{
	const int hc_iter_times = int(1 / step_size + 0.5);
	double t = 0;
	
	M54 Jq, Jq2;
	M51 t_grad, Ht2;
	Eigen::Vector4d q{ _q }; // save result 

	for (int i = 0; i < hc_iter_times; i++)
	{
		// Prediction(Euler's method)
		if (predictor_flag) {
			Eigen::Vector4d k1 = g(M_F, M_G, t, q);
			Eigen::Vector4d k2 = g(M_F, M_G, t + step_size/2, q + step_size/2 * k1);
			Eigen::Vector4d k3 = g(M_F, M_G, t + step_size/2, q + step_size/2 * k2);
			Eigen::Vector4d k4 = g(M_F, M_G, t + step_size, q + step_size * k3);
			q += (k1 + 2*k2 + 2*k3 + k4)* step_size/6;
		} else {
			q += g(M_F, M_G, t, q) * step_size;
		}

        t += step_size;
		// Correction(Newton's method)
		for (int iter = 0; iter < newton_iter; iter++)
		{
			Ht2 = t * UPnP_polys(q, M_F) + UPnP_polys(q, M_G)*(1 - t);
			Jq2 = Jacobian(q, M_F)*t + Jacobian(q, M_G)*(1 - t);
			if (adaptive_flag) {
				Eigen::Vector4d q0 = q;
				q = q - pinv(Jq2.array() + epsilon)*Ht2;
				if ( (q - q0).norm() < 1e-8) {
					break;
				}
			} else {
				q = q - pinv(Jq2.array() + epsilon)*Ht2;
			}
		}
	}
	return q;
}


std::pair<py::array_t<double>, long long> HomotopyContinuation(py::array_t<double> q_array, py::array_t<double> M_F_array, py::array_t<double> M_G_array)
{
	// Access the underlying C++ arrays
	py::buffer_info q_buf_info = q_array.request();
	py::buffer_info M_F_buf_info = M_F_array.request();
	py::buffer_info M_G_buf_info = M_G_array.request();

	Eigen::Map<Eigen::Vector4d> q(static_cast<double *>(q_buf_info.ptr), q_buf_info.size);
	Eigen::Map<M11> M_F(static_cast<double *>(M_F_buf_info.ptr), M_F_buf_info.shape[0], M_F_buf_info.shape[1]);
	Eigen::Map<M11> M_G(static_cast<double *>(M_G_buf_info.ptr), M_G_buf_info.shape[0], M_G_buf_info.shape[1]);

	// Start timing
    auto start = std::chrono::high_resolution_clock::now();

	// Call the HC function
	Eigen::Vector4d result = HC_base(q, M_F, M_G);

	// Stop timing
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	auto time = duration.count();
				
	// Convert the result to a NumPy array and return
	py::array_t<double> result_array(result.size());
	py::buffer_info result_buf_info = result_array.request();
	double *result_ptr = static_cast<double *>(result_buf_info.ptr);
	std::copy(result.data(), result.data() + result.size(), result_ptr);

	return std::make_pair(result_array, time);
}

Eigen::MatrixXd getMatfromTensor(const torch::Tensor& x_tensor)
{
    torch::Tensor tensor_cpu = x_tensor.to(torch::kCPU).to(torch::kDouble);

    int rows = tensor_cpu.size(0);
    int cols = tensor_cpu.size(1);
	
    Eigen::MatrixXd matrix(rows, cols);
    std::memcpy(matrix.data(), tensor_cpu.data_ptr<double>(), sizeof(double) * rows * cols);

	return matrix;
}

torch::Tensor eigen_matrix_to_torch_tensor(const Eigen::MatrixXd& matrix) 
{
    // Get the dimensions of the matrix
    int rows = matrix.rows();
    int cols = matrix.cols();

    // Create a tensor of the appropriate size
    torch::Tensor tensor = torch::empty({rows, cols}, torch::kDouble);

    // Copy the data from the Eigen matrix to the tensor
    std::memcpy(tensor.data_ptr<double>(), matrix.data(), sizeof(double) * rows * cols);

    return tensor;
}

void set_newton_iter(int value)
{
	newton_iter = value;
}

void set_step_size(double value)
{
	step_size = value;
}

void set_value(double newton_iter_, double step_size_, bool adaptive_flag_, bool predictor_flag_) 
{
	newton_iter = newton_iter_;
	step_size = step_size_;
	adaptive_flag = adaptive_flag_;
	predictor_flag = predictor_flag_;
}

PYBIND11_MODULE(HC_UPnP, m) 
{
	m.def("HomotopyContinuation", &HomotopyContinuation, "Do the Homotopy Continuation result with adaptive Newton's method stepsize and 4th predictor");
	m.def("set_value", &set_value, "Set the newton iter, step size of HC, adaptive Newton's method flag and predictor flag");
	m.def("set_newton_iter", &set_newton_iter, "Set the newton iter variable");
	m.def("set_step_size", &set_step_size, "Set the Homotopy Continuation step size");
}

