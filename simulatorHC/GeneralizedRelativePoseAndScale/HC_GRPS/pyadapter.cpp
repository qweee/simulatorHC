#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include "utils.h"
#include "HCBase.h"
#include <iostream>
#include <chrono>
#include <utility>


namespace py = pybind11;


std::pair<at::Tensor, long long> HomotopyContinuation_GRPS(const torch::Tensor& x_tensor, 
	const torch::Tensor& fs_tensor, const torch::Tensor& fs_prime_tensor, 
	const torch::Tensor& vs_tensor, const torch::Tensor& vs_prime_tensor, 
	const torch::Tensor& fs_hat_tensor, const double& step_size, const double& newton_iter, int& nPts)
{
	// Create an Eigen matrix and copy the data
	Eigen::Matrix<double, 8, 1> x = getMatFromTensor<Eigen::Matrix<double, 8, 1>>(x_tensor);
	Eigen::Matrix<double, 8, 1> result;
	result.setZero();
	long long time;
	
	if (nPts == 7) {
		Eigen::Matrix<double, 3, 7> fs = getMatFromTensor<Eigen::Matrix<double, 3, 7>>(fs_tensor);
		Eigen::Matrix<double, 3, 7> fs_prime = getMatFromTensor<Eigen::Matrix<double, 3, 7>>(fs_prime_tensor);
		Eigen::Matrix<double, 3, 7> vs = getMatFromTensor<Eigen::Matrix<double, 3, 7>>(vs_tensor);
		Eigen::Matrix<double, 3, 7> vs_prime = getMatFromTensor<Eigen::Matrix<double, 3, 7>>(vs_prime_tensor);

		Eigen::Matrix<double, 3, 7> fs_hat = getMatFromTensor<Eigen::Matrix<double, 3, 7>>(fs_hat_tensor);

		// Start timing
		auto start = std::chrono::high_resolution_clock::now();

		GRPS<7> GRPS_HC(fs, fs_prime, vs, vs_prime, fs_hat);
		result = GRPS_HC.HC_base(x, step_size, newton_iter);

		// Stop timing
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
		time = duration.count();

	} else if (nPts == 8) {
		Eigen::Matrix<double, 3, 8> fs = getMatFromTensor<Eigen::Matrix<double, 3, 8>>(fs_tensor);
		Eigen::Matrix<double, 3, 8> fs_prime = getMatFromTensor<Eigen::Matrix<double, 3, 8>>(fs_prime_tensor);
		Eigen::Matrix<double, 3, 8> vs = getMatFromTensor<Eigen::Matrix<double, 3, 8>>(vs_tensor);
		Eigen::Matrix<double, 3, 8> vs_prime = getMatFromTensor<Eigen::Matrix<double, 3, 8>>(vs_prime_tensor);

		Eigen::Matrix<double, 3, 8> fs_hat = getMatFromTensor<Eigen::Matrix<double, 3, 8>>(fs_hat_tensor);

		// Start timing
		auto start = std::chrono::high_resolution_clock::now();

		GRPS<8> GRPS_HC(fs, fs_prime, vs, vs_prime, fs_hat);
		result = GRPS_HC.HC_base(x, step_size, newton_iter);

		// Stop timing
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
		time = duration.count();
	} else {
		std::cout << "not 7 or 8 points then use dynamic eigen" << std::endl;
		// TODO: use dynamic eigen
	}
	

	return std::make_pair(eigen_matrix_to_torch_tensor(result),time);
};


PYBIND11_MODULE(HC_GRPS, m) 
{
	m.def("HomotopyContinuation_GRPS", &HomotopyContinuation_GRPS, "C++ HC for Generalized Relative Pose and Scale problem output with time");
};
