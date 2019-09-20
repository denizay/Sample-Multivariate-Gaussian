#include <iostream>
#include <vector>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;
using namespace std;

std::vector<double> eig2vec(Eigen::MatrixXd mat) {
    std::vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    return vec;
}

int main() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0,1};

    Eigen::MatrixXd mat(300, 2);
    for(int j=0; j<300;++j){
        for(int ii=0; ii<2; ++ii){
            mat(j,ii) = d(gen);
        }
    }

    Eigen::Vector2i mean;
    mean << 0, 0;
    Matrix2d cov;
    cov << 1, 0.8,
           0.8, 1;
    JacobiSVD<MatrixXd> svd( cov, ComputeThinU | ComputeThinV);
    MatrixXd tmp, u, v, a, fin;
    u = svd.singularValues().asDiagonal();
    u = u.array().sqrt();
    v = svd.matrixV();
    a = u*v;
    std::cout << mat * a << std::endl;
    fin = mat * a;
    plt::scatter(eig2vec(fin.col(0)), eig2vec(fin.col(1)));
    plt::show();
    return 0;
}
