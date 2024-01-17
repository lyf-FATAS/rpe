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
    void GNCPointCloudRegister::registerPointCloudGNC(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2, Matrix3d &R12_ransac, Vector3d &t12_ransac)
    {
        NonlinearFactorGraph graph;
        noiseModel::Diagonal::shared_ptr noise = noiseModel::Unit::Create(3);

        CHECK_EQ(kps3d1.size(), kps3d2.size());
        for (size_t i = 0; i < kps3d1.size(); i++)
            graph.add(Point2pFactor(X(0), kps3d1[i], kps3d2[i], noise));

        Values initial;
        initial.insert(X(0), Pose3(Matrix4d::Identity()));

        GncParams<GaussNewtonParams> gncParams;
        gncParams.setLossType(GncLossType::TLS);
        switch (gnc_verbosity_level)
        {
        case 0:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::SILENT);
            break;
        case 1:
            gncParams.setVerbosityGNC(GncParams<GaussNewtonParams>::Verbosity::SUMMARY);
            break;
        default:
            LOG(FATAL) << "Invalid GNC verbosity level #^#";
        }

        auto optimizer = GncOptimizer<GncParams<GaussNewtonParams>>(graph, initial, gncParams);
        Values estimate = optimizer.optimize();

        Matrix4d T = estimate.at<Pose3>(X(0)).matrix();
        R12_ransac = T.topLeftCorner<3, 3>();
        t12_ransac = T.topRightCorner<3, 1>();
    }
}