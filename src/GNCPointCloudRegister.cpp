#include "GNCPointCloudRegister.hpp"
#include <gtsam/nonlinear/GncOptimizer.h>

namespace RPE
{
    GNCPointCloudRegister::GNCPointCloudRegister(string settings_path)
    {
        cv::FileStorage settings(settings_path, cv::FileStorage::READ);
        settings["gnc_verbosity_level"] >> gnc_verbosity_level;
    }

    using symbol_shorthand::X;
    void GNCPointCloudRegister::registerPointCloudGNC(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2,
                                                      Matrix3d &R12, Vector3d &t12, vector<int> &inliers)
    {
        NonlinearFactorGraph graph;
        noiseModel::Diagonal::shared_ptr noise = noiseModel::Unit::Create(3);

        CHECK_EQ(kps3d1.size(), kps3d2.size());
        for (size_t i = 0; i < kps3d1.size(); i++)
            graph.add(Point2pFactor(X(0), kps3d1[i], kps3d2[i], noise));
        LOG(INFO) << "Number of matches = " << kps3d1.size() << " in GNC point cloud registration";

        Values initial;
        initial.insert(X(0), Pose3(Matrix4d::Identity()));

        GncParams<GaussNewtonParams> gncParams;
        gncParams.setLossType(GncLossType::TLS);
        gncParams.setMuStep(5.0);
        switch (gnc_verbosity_level)
        {
        case 0:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::SILENT);
            break;
        case 1:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::SUMMARY);
            break;
        case 2:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::MU);
            break;
        case 3:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::WEIGHTS);
            break;
        case 4:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::VALUES);
            break;
        default:
            LOG(FATAL) << "Invalid GNC verbosity level #^#";
        }

        auto optimizer = GncOptimizer<GncParams<GaussNewtonParams>>(graph, initial, gncParams);
        Values estimate = optimizer.optimize();
        VectorXd weights = optimizer.getWeights();
        CHECK_EQ(kps3d1.size(), weights.rows());

        Matrix4d T = estimate.at<Pose3>(X(0)).matrix();
        R12 = T.topLeftCorner<3, 3>();
        t12 = T.topRightCorner<3, 1>();

        inliers.resize(weights.rows());
        for (size_t i = 0; i < weights.rows(); i++)
            if (weights(i))
                inliers[i] = 1;
            else
                inliers[i] = 0;
    }
}