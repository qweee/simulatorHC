#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "utils.h"

template<typename T>
bool checkValid(T& A){

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd singular_values = svd.singularValues();

    double tolerance = 1e-10;
    if (singular_values.minCoeff() < tolerance) {

        return false;

    } else {

        return true; 
    }

};

template<int N>
Eigen::Matrix<double, 8, N+1> pinv(const Eigen::Matrix<double, N+1, 8>& M) {

    return (M.transpose() * M).inverse() * M.transpose();

};

template<int N>
class GRPS {

    Eigen::Matrix<double, 3, N> fs;
    Eigen::Matrix<double, 3, N> fs_prime;
	Eigen::Matrix<double, 3, N> vs;
    Eigen::Matrix<double, 3, N> vs_prime;

    Eigen::Matrix<double, 3, N> fs_hat;

    std::vector<Eigen::Matrix3d> skewF;
    std::vector<Eigen::Matrix3d> FprimeF;
    std::vector<Eigen::Matrix3d> skewVprimeFprimeF;
    std::vector<Eigen::Vector3d> skewVprimeFprime;
    std::vector<Eigen::Matrix3d> skewFprimeFV;
    std::vector<Eigen::Matrix<double, 1, 3>> skewFV;

    std::vector<Eigen::Matrix3d> skewF_hat;
    std::vector<Eigen::Matrix3d> FprimeF_hat;
    std::vector<Eigen::Matrix3d> skewVprimeFprimeF_hat;
    std::vector<Eigen::Matrix3d> skewFprimeFV_hat;
    std::vector<Eigen::Matrix<double, 1, 3>> skewFV_hat;
    
    int newton_iter = 10;
    double step_size = 0.02;
    bool adaptive_flag = true;
    bool predictor_flag = true;
    double epsilon = 1e-10;

public:

    GRPS(const Eigen::Matrix<double, 3, N>& fs_, const Eigen::Matrix<double, 3, N>& fs_prime_,
	        const Eigen::Matrix<double, 3, N>& vs_, const Eigen::Matrix<double, 3, N>& vs_prime_, 
            const Eigen::Matrix<double, 3, N>& fs_hat_);

    void Hpolyandjacobian(double t, Eigen::Matrix<double, 8, 1>& x, 
        Eigen::Matrix<double, N+1, 1>& polys, Eigen::Matrix<double, N+1, 8>& J);

    // Eigen::Matrix<double, 8, 1> Hg4(double t, Eigen::Matrix<double, 8, 1>& x);
    bool Hg4(double t, Eigen::Matrix<double, 8, 1>& x, Eigen::Matrix<double, 8, 1>& deltax);

    void polyandjacobian(Eigen::Matrix<double, 8, 1>& x, Eigen::Matrix<double, 3, N>& fs1, 
        std::vector<Eigen::Matrix3d>& skewF_, std::vector<Eigen::Matrix<double, 1, 3>>& skewFV_,
        std::vector<Eigen::Matrix3d>& FprimeF_, std::vector<Eigen::Matrix3d>& skewVprimeFprimeF_,
        std::vector<Eigen::Matrix3d>& skewFprimeFV_, 
        Eigen::Matrix<double, N+1, 1>& polys, Eigen::Matrix<double, N+1, 8>& J);

    void precomputeSkewSym(const Eigen::Matrix<double, 3, N>& fs, const Eigen::Matrix<double, 3, N>& fs_prime, 
        const Eigen::Matrix<double, 3, N>& vs, const Eigen::Matrix<double, 3, N>& vs_prime,
        std::vector<Eigen::Matrix3d>& skewF, std::vector<Eigen::Matrix3d>& FprimeF, 
        std::vector<Eigen::Matrix3d>& skewVprimeFprimeF, std::vector<Eigen::Vector3d>& skewVprimeFprime, 
        std::vector<Eigen::Matrix3d>& skewFprimeFV, std::vector<Eigen::Matrix<double, 1, 3>>& skewFV);

    void precomputeSkewSym_fhat(const Eigen::Matrix<double, 3, N>& fs, const Eigen::Matrix<double, 3, N>& fs_prime, 
        const Eigen::Matrix<double, 3, N>& vs, const Eigen::Matrix<double, 3, N>& vs_prime,
        std::vector<Eigen::Matrix3d>& skewF, std::vector<Eigen::Matrix3d>& FprimeF, 
        std::vector<Eigen::Matrix3d>& skewVprimeFprimeF,
        std::vector<Eigen::Matrix3d>& skewFprimeFV, std::vector<Eigen::Matrix<double, 1, 3>>& skewFV);

    Eigen::Matrix<double, 8, 1> HC_base(const Eigen::Matrix<double, 8, 1>& _x, 
        const double& step_size_new, const double& newton_iter_new);

    bool HC_base(const Eigen::Matrix<double, 8, 1>& _x, 
        const double& step_size_new, const double& newton_iter_new, Eigen::Matrix<double, 8, 1>& result);
    
};

// GRPS<N> where N is the number of points
template<int N>
void GRPS<N>::precomputeSkewSym(const Eigen::Matrix<double, 3, N>& fs, const Eigen::Matrix<double, 3, N>& fs_prime, 
    const Eigen::Matrix<double, 3, N>& vs, const Eigen::Matrix<double, 3, N>& vs_prime,
    std::vector<Eigen::Matrix3d>& skewF, std::vector<Eigen::Matrix3d>& FprimeF, 
    std::vector<Eigen::Matrix3d>& skewVprimeFprimeF, std::vector<Eigen::Vector3d>& skewVprimeFprime, 
    std::vector<Eigen::Matrix3d>& skewFprimeFV, std::vector<Eigen::Matrix<double, 1, 3>>& skewFV) {

    for (int i = 0; i < fs.cols(); ++i) {
        const Eigen::Vector3d& f = fs.col(i);
        const Eigen::Vector3d& f_prime = fs_prime.col(i);
        const Eigen::Vector3d& v = vs.col(i);
        const Eigen::Vector3d& v_prime = vs_prime.col(i);

        skewF.push_back(skew_sym(f));
        Eigen::Matrix3d FprimeF_ = f_prime * f.transpose();
        FprimeF.push_back(FprimeF_);
        Eigen::Vector3d temp0 = skew_sym(v_prime) * f_prime;
        skewVprimeFprime.push_back(temp0);
        Eigen::Matrix3d temp1 = skew_sym(v_prime) * f_prime * f.transpose();
        skewVprimeFprimeF.push_back(temp1);
        Eigen::Matrix<double, 1, 3> temp2 = f.transpose() * skew_sym(v);
        skewFV.push_back(temp2);
        Eigen::Matrix3d temp3 = f_prime * f.transpose() * skew_sym(v);
        skewFprimeFV.push_back(temp3);

    }
    
};

template<int N>
void GRPS<N>::precomputeSkewSym_fhat(const Eigen::Matrix<double, 3, N>& fs, const Eigen::Matrix<double, 3, N>& fs_prime, 
    const Eigen::Matrix<double, 3, N>& vs, const Eigen::Matrix<double, 3, N>& vs_prime,
    std::vector<Eigen::Matrix3d>& skewF, std::vector<Eigen::Matrix3d>& FprimeF, 
    std::vector<Eigen::Matrix3d>& skewVprimeFprimeF,
    std::vector<Eigen::Matrix3d>& skewFprimeFV, std::vector<Eigen::Matrix<double, 1, 3>>& skewFV) {

    for (int i = 0; i < fs.cols(); ++i) {
        const Eigen::Vector3d& f = fs.col(i);
        const Eigen::Vector3d& f_prime = fs_prime.col(i);
        const Eigen::Vector3d& v = vs.col(i);
        const Eigen::Vector3d& v_prime = vs_prime.col(i);

        skewF.push_back(skew_sym(f));
        FprimeF.push_back(f_prime * f.transpose());
        skewVprimeFprimeF.push_back(skew_sym(v_prime) * f_prime * f.transpose());
        skewFV.push_back(f.transpose() * skew_sym(v));
        skewFprimeFV.push_back(f_prime * f.transpose() * skew_sym(v));

    }
    
};

template<int N>
GRPS<N>::GRPS(const Eigen::Matrix<double, 3, N>& fs_, const Eigen::Matrix<double, 3, N>& fs_prime_,
	        const Eigen::Matrix<double, 3, N>& vs_, const Eigen::Matrix<double, 3, N>& vs_prime_, 
            const Eigen::Matrix<double, 3, N>& fs_hat_){
            
    fs = fs_;
    fs_prime = fs_prime_;
    vs = vs_;
    vs_prime = vs_prime_;
    fs_hat = fs_hat_;

    precomputeSkewSym(fs, fs_prime, vs, vs_prime, skewF, FprimeF, 
        skewVprimeFprimeF, skewVprimeFprime, skewFprimeFV, skewFV);

    precomputeSkewSym_fhat(fs_hat, fs_prime, vs, vs_prime, skewF_hat, 
        FprimeF_hat, skewVprimeFprimeF_hat, skewFprimeFV_hat, skewFV_hat);


};

template<int N>
Eigen::Matrix<double, 8, 1> GRPS<N>::HC_base(const Eigen::Matrix<double, 8, 1>& _x, 
    const double& step_size_new, const double& newton_iter_new)
{

    step_size = step_size_new;
    newton_iter = newton_iter_new;

    const int hc_iter_times = int(1 / step_size + 0.5);
	
	double t = 0;
	
	Eigen::Matrix<double, N+1, 8> Jx2;
	Eigen::Matrix<double, N+1, 1> t_grad, Ht2;
	Eigen::Matrix<double, 8, 1> x{ _x };

	for (int i = 0; i < hc_iter_times; i++)
	{
		// Prediction(Kutta Runge or Euler's method)
		// t = i * step_size;
        
		if (predictor_flag) {
            
			Eigen::Matrix<double, 8, 1> k1;
            bool k1_flag = Hg4(t, x, k1);
            if (!k1_flag) {
                break;
            }
            Eigen::Matrix<double, 8, 1> x_temp1 = x + step_size/2 * k1;
			Eigen::Matrix<double, 8, 1> k2;
            bool k2_flag = Hg4(t + step_size/2, x_temp1, k2);
            if (!k2_flag) {
                break;
            }
            Eigen::Matrix<double, 8, 1> x_temp2 = x + step_size/2 * k2;
			Eigen::Matrix<double, 8, 1> k3;
            bool k3_flag = Hg4(t + step_size/2, x_temp2, k3);
            if (!k3_flag) {
                break;
            }
            Eigen::Matrix<double, 8, 1> x_temp3 = x + step_size * k3;
			Eigen::Matrix<double, 8, 1> k4;
            bool k4_flag = Hg4(t + step_size, x_temp3, k4);
            if (!k4_flag) {
                break;
            }
			x += (k1 + 2*k2 + 2*k3 + k4)* step_size/6;

		} else {
            Eigen::Matrix<double, 8, 1> k1;
            bool k1_flag = Hg4(t, x, k1);
            if (!k1_flag) {
                break;
            }
			x += k1 * step_size;
		}

        t += step_size;

		// Correction(Newton method)
		for (int iter = 0; iter < newton_iter; iter++)
		{
            Eigen::Matrix<double, N+1, 1> Ht2 = Eigen::Matrix<double, N+1, 1>::Zero();
            Eigen::Matrix<double, N+1, 8> Jx2 = Eigen::Matrix<double, N+1, 8>::Zero();
            Hpolyandjacobian(t, x, Ht2, Jx2);

            if (!checkValid(Jx2)) {
                break;
            }

			if (adaptive_flag) {
				Eigen::Matrix<double, 8, 1> x0 = x;
                Eigen::Matrix<double, 8, N+1> Jinv = pinv<N>(Jx2.array() + epsilon);
				x = x0 - Jinv*Ht2;
                
				if ( (x - x0).norm() < 1e-8) {
					break;
				}
			} else {
                Eigen::Matrix<double, 8, N+1> Jinv = pinv<N>(Jx2.array() + epsilon);
				x = x - Jinv*Ht2;
			}
            
		}

        bool containsNaN = !(x.array() == x.array()).all(); // NaN check
        bool containsInf = !x.array().isFinite().all(); // Inf check
        bool containsLarge = !(x.array().abs() < 1e3).all(); // large check

        if (containsNaN || containsInf || containsLarge) {
            break;
        }

    }

	return x;
};

template<int N>
void GRPS<N>::polyandjacobian(Eigen::Matrix<double, 8, 1>& x, Eigen::Matrix<double, 3, N>& fs1, 
    std::vector<Eigen::Matrix3d>& skewF_, std::vector<Eigen::Matrix<double, 1, 3>>& skewFV_,
    std::vector<Eigen::Matrix3d>& FprimeF_, std::vector<Eigen::Matrix3d>& skewVprimeFprimeF_,
    std::vector<Eigen::Matrix3d>& skewFprimeFV_, 
    Eigen::Matrix<double, N+1, 1>& polynomials, Eigen::Matrix<double, N+1, 8>& J){

    const int nPts = fs1.cols();

    const Eigen::Vector4d q = x.block<4, 1>(0, 0);
    const Eigen::Vector3d t = x.block<3, 1>(4, 0);
    const double s = x(7, 0);
    const Eigen::Matrix3d R = getRfromq(q);

    const Eigen::Matrix3d skewtT = skew_sym(t).transpose();
    const Eigen::Matrix<double, 3, N> Rfprime = R * fs_prime;

    // #pragma omp parallel for
    for (int i = 0; i < nPts; i++) {
        const Eigen::Vector3d& f = fs1.col(i);

        // for polys
        double part1 = t.transpose() * (skewF_[i] * Rfprime.col(i));
        double part2 = -s * f.transpose() * R * skewVprimeFprime[i];
        polynomials(i, 0) = part1 + part2 + skewFV_[i] * Rfprime.col(i);

        // for jacobians
        Eigen::Matrix3d a = FprimeF_[i] * skewtT - s * skewVprimeFprimeF_[i] + skewFprimeFV_[i];

        J(i, 0) = 2 * q(0) * a(0, 0) + 2 * q(0) * a(1, 1) + 2 * q(0) * a(2, 2) +
          2 * q(1) * a(1, 2) - 2 * q(1) * a(2, 1) -
          2 * q(2) * a(0, 2) + 2 * q(2) * a(2, 0) +
          2 * q(3) * a(0, 1) - 2 * q(3) * a(1, 0);

        J(i, 1) = 2 * q(0) * a(1, 2) - 2 * q(0) * a(2, 1) + 
                2 * q(1) * a(0, 0) - 2 * q(1) * a(1, 1) - 2 * q(1) * a(2, 2) +
                2 * q(2) * a(0, 1) + 2 * q(2) * a(1, 0) + 
                2 * q(3) * a(0, 2) + 2 * q(3) * a(2, 0);

        J(i, 2) = -2 * q(0) * a(0, 2) + 2 * q(0) * a(2, 0) + 
                2 * q(1) * a(0, 1) + 2 * q(1) * a(1, 0) - 
                2 * q(2) * a(0, 0) + 2 * q(2) * a(1, 1) - 2 * q(2) * a(2, 2) + 
                2 * q(3) * a(1, 2) + 2 * q(3) * a(2, 1);

        J(i, 3) = 2 * q(0) * a(0, 1) - 2 * q(0) * a(1, 0) + 
                2 * q(1) * a(0, 2) + 2 * q(1) * a(2, 0) + 
                2 * q(2) * a(1, 2) + 2 * q(2) * a(2, 1) - 
                2 * q(3) * a(0, 0) - 2 * q(3) * a(1, 1) + 2 * q(3) * a(2, 2);

        J.block(i, 4, 1, 3) = (skewF_[i] * Rfprime.col(i)).transpose();
        J(i, 7) = -f.transpose() * R * skewVprimeFprime[i];
    }

    J.template block<1, 4>(N, 0) = 2 * q.transpose();

    polynomials(N, 0) = q.squaredNorm() - 1;

};


template<int N>
void GRPS<N>::Hpolyandjacobian(double t, Eigen::Matrix<double, 8, 1>& x, 
    Eigen::Matrix<double, N+1, 1>& polys, Eigen::Matrix<double, N+1, 8>& J){
    
    Eigen::Matrix<double, N+1, 1> F = Eigen::Matrix<double, N+1, 1>::Zero();
    Eigen::Matrix<double, N+1, 8> JF = Eigen::Matrix<double, N+1, 8>::Zero();

    Eigen::Matrix<double, N+1, 1> G = Eigen::Matrix<double, N+1, 1>::Zero();
    Eigen::Matrix<double, N+1, 8> JG = Eigen::Matrix<double, N+1, 8>::Zero();

    polyandjacobian(x, fs, skewF, skewFV, FprimeF, skewVprimeFprimeF, skewFprimeFV, F, JF);
    polyandjacobian(x, fs_hat, skewF_hat, skewFV_hat, FprimeF_hat, skewVprimeFprimeF_hat, skewFprimeFV_hat, G, JG);

    polys = t * F + (1-t) * G;
    J = t * JF + (1-t) * JG;

};


template<int N>
bool GRPS<N>::Hg4(double t, Eigen::Matrix<double, 8, 1>& x, Eigen::Matrix<double, 8, 1>& deltax){

    Eigen::Matrix<double, N+1, 1> F = Eigen::Matrix<double, N+1, 1>::Zero();
    Eigen::Matrix<double, N+1, 8> JF = Eigen::Matrix<double, N+1, 8>::Zero();

    Eigen::Matrix<double, N+1, 1> G = Eigen::Matrix<double, N+1, 1>::Zero();
    Eigen::Matrix<double, N+1, 8> JG = Eigen::Matrix<double, N+1, 8>::Zero();

    polyandjacobian(x, fs, skewF, skewFV, FprimeF, skewVprimeFprimeF, skewFprimeFV, F, JF);
    polyandjacobian(x, fs_hat, skewF_hat, skewFV_hat, FprimeF_hat, skewVprimeFprimeF_hat, skewFprimeFV_hat, G, JG);

    Eigen::Matrix<double, N+1, 8> Jx = t * JF + (1-t) * JG;
    Eigen::Matrix<double, N+1, 1> t_grad = F - G;

    // Compute the Singular Value Decomposition of A
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jx, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd singular_values = svd.singularValues();

    double tolerance = 1e-10;
    if (singular_values.minCoeff() < tolerance) {

        return false;

    } else {
        deltax.setZero();
        deltax = -pinv<N>(Jx.array())*t_grad;

        return true; 
    }
 
         
};
