#pragma once
#include <string>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>
#include "gurobi_c++.h"

using namespace std;
using namespace Eigen;

namespace RPE
{
    class RSolver
    {
    public:
        RSolver();
        void solve();
    };

    class ASolver
    {
    public:
        ASolver(string settings_path);
        void solve(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12, const MatrixXd &A_tilde, MatrixXd &A);

        void computeDotProducts(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12,
                                vector<double> &p21_dot_p21, vector<double> &p2_dot_p2, vector<double> &p21_dot_p2);
        void computeLinearLeastSquaresObjective(const vector<double> &p21_dot_p21, const vector<double> &p2_dot_p2, const vector<double> &p21_dot_p2, const double mu,
                                                const MatrixXd &A_tilde, const vector<GRBVar> &A,
                                                GRBQuadExpr &obj);

        void computeDistanceMatrix(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12, MatrixXd &D);
        void computeLinearObjective(const MatrixXd &D, const double mu, const MatrixXd &A_tilde, const vector<GRBVar> &A, GRBLinExpr &obj);

        int GRB_log_to_console;
        double GRB_time_limit;
        double mu, nu;
    };
}