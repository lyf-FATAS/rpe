#include "GNCPointCloudRegister.hpp"
#include <gtsam/nonlinear/GncOptimizer.h>

namespace RPE
{
    using symbol_shorthand::X;
    void registerPointCloudGNC(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2, Matrix3d &R12_gv, Vector3d &t12_gv)
    {
        NonlinearFactorGraph graph;
        noiseModel::Diagonal::shared_ptr noise = noiseModel::Unit::Create(3);

        CHECK_EQ(kps3d1.size(), kps3d2.size());
        for (size_t i = 0; i < kps3d1.size(); i++)
            graph.add(Point2pFactor(X(0), kps3d1[i], kps3d2[i], noise));

        Values initial;
        initial.insert(X(0), Pose3(Matrix4d::Identity()));

        // Set options for the non-minimal solver
        LevenbergMarquardtParams lmParams;
        lmParams.setMaxIterations(1000);
        lmParams.setRelativeErrorTol(1e-5);

        GncParams<LevenbergMarquardtParams> gncParams(lmParams);
        gncParams.setLossType(GncLossType::TLS);
        gncParams.setVerbosityGNC(GncParams<LevenbergMarquardtParams>::Verbosity::SUMMARY);

        GncOptimizer<GncParams<LevenbergMarquardtParams>> optimizer(graph, initial, gncParams);
        LOG(INFO) << "Entering GNC optimization...";
        Values estimate = optimizer.optimize();
        LOG(INFO) << "GNC optimization complete :)";

        Matrix4d T = estimate.at<Pose3>(X(0)).matrix();
        R12_gv = T.topLeftCorner<3, 3>();
        t12_gv = T.topRightCorner<3, 1>();
    }
}