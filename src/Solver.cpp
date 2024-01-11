#include "Solver.hpp"
#include <iostream>

using namespace std;

namespace RPE
{
    RSolver::RSolver() {}

    void RSolver::solve() {}

    ASolver::ASolver(string settings_path)
    {
        cv::FileStorage settings(settings_path, cv::FileStorage::READ);
        settings["GRB_log_to_console"] >> GRB_log_to_console;
        settings["GRB_time_limit"] >> GRB_time_limit;
        settings["mu"] >> mu;
        settings["nu"] >> nu;
    }

    void ASolver::solve(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12, const MatrixXd &A_tilde, MatrixXd &A)
    {
        CHECK_EQ(P1.size(), A_tilde.cols());
        CHECK_EQ(P2.size(), A_tilde.rows());
        CHECK_EQ(A.rows(), A_tilde.rows());
        CHECK_EQ(A.cols(), A_tilde.cols());

        const size_t N1 = P1.size();
        const size_t N2 = P2.size();

        // These dot products are required in quadratic obj construction
        // vector<double> p21_dot_p21, p2_dot_p2, p21_dot_p2;
        // computeDotProducts(P1, P2, R12, t12, p21_dot_p21, p2_dot_p2, p21_dot_p2);

        MatrixXd D;
        computeDistanceMatrix(P1, P2, R12, t12, D);

        /******************************/
        /****** Solve A by QP/LP ******/
        /******************************/
        try
        {
            GRBEnv env;
            GRBModel model(env);
            model.set(GRB_StringAttr_ModelName, "A_subproblem");
            model.set(GRB_IntParam_LogToConsole, GRB_log_to_console);

            vector<GRBVar> A_(N2 * N1);
            for (size_t i = 0; i < N2; i++)
                for (size_t j = 0; j < N1; j++)
                    A_[i * N1 + j] = model.addVar(
                        0.0, 1.0, 0.0, GRB_BINARY, "match_" + to_string(j) + to_string(i));

            // GRBQuadExpr obj;
            // computeLinearLeastSquaresObjective(p21_dot_p21, p2_dot_p2, p21_dot_p2, mu, A_tilde, A_, obj);

            GRBLinExpr obj;
            computeLinearObjective(D, mu, A_tilde, A_, obj);

            model.setObjective(obj, GRB_MINIMIZE);

            // Doubly stochastic constraints for the matching matrix A
            for (size_t i = 0; i < N2; i++)
            {
                GRBLinExpr row_sum = 0;
                for (size_t j = 0; j < N1; j++)
                    row_sum += A_[i * N1 + j];
                model.addConstr(row_sum <= 1, "doubly_stochastic_constraint_row_" + to_string(i));
            }

            for (size_t j = 0; j < N1; j++)
            {
                GRBLinExpr col_sum = 0;
                for (size_t i = 0; i < N2; i++)
                    col_sum += A_[i * N1 + j];
                model.addConstr(col_sum <= 1, "doubly_stochastic_constraint_col_" + to_string(j));
            }

            // Set the initial value of A
            for (size_t i = 0; i < N2; i++)
                for (size_t j = 0; j < N1; j++)
                    A_[i * N1 + j].set(GRB_DoubleAttr_Start, (bool)A(i, j));

            model.set(GRB_DoubleParam_TimeLimit, GRB_time_limit);
            model.optimize();

            for (size_t i = 0; i < N2; i++)
                for (size_t j = 0; j < N1; j++)
                    A(i, j) = A_[i * N1 + j].get(GRB_DoubleAttr_X);
        }
        catch (const GRBException &e)
        {
            LOG(FATAL) << "GRB exception: error code = " << e.getErrorCode() << " " << e.getMessage();
        }
        catch (...)
        {
            LOG(FATAL) << "Exception during optimization #^#";
        }
    }

    void ASolver::computeDotProducts(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12,
                                     vector<double> &p21_dot_p21, vector<double> &p2_dot_p2, vector<double> &p21_dot_p2)
    {
        const size_t N1 = P1.size();
        const size_t N2 = P2.size();

        p21_dot_p21.resize(N1);
        p2_dot_p2.resize(N2 * N2);
        p21_dot_p2.resize(N1 * N2);

        vector<Vector3d> P21;
        for (size_t i = 0; i < N1; i++)
            P21.emplace_back(R12.transpose() * (P1[i] - t12));

        for (size_t i = 0; i < N1; i++)
            p21_dot_p21[i] = P21[i].transpose() * P21[i];

        for (size_t i = 0; i < N2; i++)
            for (size_t j = 0; j < N2; j++)
                p2_dot_p2[i * N2 + j] = P2[i].transpose() * P2[j];

        for (size_t i = 0; i < N1; i++)
            for (size_t j = 0; j < N2; j++)
                p21_dot_p2[i * N2 + j] = P21[i].transpose() * P2[j];
    }

    void ASolver::computeLinearLeastSquaresObjective(const vector<double> &p21_dot_p21, const vector<double> &p2_dot_p2, const vector<double> &p21_dot_p2, const double mu,
                                                     const MatrixXd &A_tilde, const vector<GRBVar> &A,
                                                     GRBQuadExpr &obj)
    {
        const size_t N1 = A_tilde.cols();
        const size_t N2 = A_tilde.rows();

        obj = 0;

        // Quadratic terms
        for (size_t i = 0; i < N2; i++)
        {
            for (size_t j = 0; j < N2; j++)
            {
                for (size_t k = 0; k < N1; k++)
                {
                    double coefficient;
                    if (i == j)
                    {
                        coefficient = p21_dot_p21[k] + p2_dot_p2[i * N2 + i] - 2 * p21_dot_p2[k * N2 + i] + mu;
                    }
                    else
                    {
                        coefficient = p21_dot_p21[k] + p2_dot_p2[i * N2 + j] - p21_dot_p2[k * N2 + i] - p21_dot_p2[k * N2 + j];
                    }
                    if (coefficient != 0)
                        obj += coefficient * A[i * N1 + k] * A[j * N1 + k];
                }
            }
        }

        // Linear terms
        for (size_t i = 0; i < N2; i++)
        {
            for (size_t j = 0; j < N1; j++)
            {
                double coefficient = -2 * mu * A_tilde(i, j);
                if (coefficient != 0)
                    obj += coefficient * A[i * N1 + j];
            }
        }
    }

    void ASolver::computeDistanceMatrix(const vector<Vector3d> &P1, const vector<Vector3d> &P2, const Matrix3d &R12, const Vector3d &t12, MatrixXd &D)
    {
        const size_t N1 = P1.size();
        const size_t N2 = P2.size();

        vector<Vector3d> P21;
        for (size_t i = 0; i < N1; i++)
            P21.emplace_back(R12.transpose() * (P1[i] - t12));

        D.resize(N2, N1);
        for (size_t i = 0; i < N2; i++)
            for (size_t j = 0; j < N1; j++)
                D(i, j) = (P2[i] - P21[j]).transpose() * (P2[i] - P21[j]);
    }

    void ASolver::computeLinearObjective(const MatrixXd &D, const double mu, const MatrixXd &A_tilde, const vector<GRBVar> &A, GRBLinExpr &obj)
    {
        CHECK_EQ(D.rows(), A_tilde.rows());
        CHECK_EQ(D.cols(), A_tilde.cols());

        const size_t N1 = A_tilde.cols();
        const size_t N2 = A_tilde.rows();

        obj = 0;

        for (size_t i = 0; i < N2; i++)
        {
            for (size_t j = 0; j < N1; j++)
            {
                double coefficient = D(i, j) - mu * A_tilde(i, j) - nu;
                obj += coefficient * A[i * N1 + j];
            }
        }
    }
}