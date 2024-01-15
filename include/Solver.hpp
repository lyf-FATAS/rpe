#pragma once
#include <string>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>
#include "gurobi_c++.h"
#include "ceres/ceres.h"
#include "ceres/autodiff_cost_function.h"

using namespace std;
using namespace Eigen;

namespace RPE
{
    class ASolver
    {
    public:
        ASolver(string settings_path);
        void solve(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12, const MatrixXd &A_tilde, MatrixXd &A);

        // QP: deprecated
        void computeDotProducts(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12,
                                vector<double> &p21_dot_p21, vector<double> &p2_dot_p2, vector<double> &p21_dot_p2);
        void computeLinearLeastSquaresObjective(const vector<double> &p21_dot_p21, const vector<double> &p2_dot_p2, const vector<double> &p21_dot_p2,
                                                const MatrixXd &A_tilde, const vector<GRBVar> &A,
                                                GRBQuadExpr &obj);

        // LP: actually used
        void computeDistanceMatrix(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12, MatrixXd &D);
        void computeLinearObjective(const MatrixXd &D, const MatrixXd &A, const MatrixXd &A_tilde, const vector<GRBVar> &A_, GRBLinExpr &obj);

        bool enable_outdoor_mode;
        int GRB_log_to_console;
        double GRB_time_limit;
        double mu, nu;
    };

    class TSolver
    {
    public:
        TSolver(string settings_path);
        void solve(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const MatrixXd &A, Matrix3d &R12, Vector3d &t12);

        bool Ceres_log_to_console;
    };

    class PointCloudRegistrationErrorTerm
    {
    public:
        PointCloudRegistrationErrorTerm(const Vector3d &p1_, const Vector3d &p2_)
            : p1(move(p1_)), p2(move(p2_)) {}

        template <typename D>
        bool operator()(const D *const q12_ptr, const D *const t_ptr, D *residuals_ptr) const;

        static ceres::CostFunction *Create(const Vector3d &p1, const Vector3d &p2);

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        const Vector3d p1;
        const Vector3d p2;
    };
}