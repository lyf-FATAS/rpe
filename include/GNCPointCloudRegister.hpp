#pragma once

#include <boost/pointer_cast.hpp>
#include <opencv2/core/core.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <glog/logging.h>

using namespace std;
using namespace Eigen;
using namespace gtsam;

namespace RPE
{

    // The following codes are copied from git@github.com:HKUST-Aerial-Robotics/Pagor.git
    class Point2pFactor : public NoiseModelFactor1<Pose3>
    {
    public:
        Point2pFactor() {} ///< Default constructor for serialization
        Point2pFactor(Key X, Vector3d src, Vector3d tgt,
                      const SharedNoiseModel &model = nullptr) : NoiseModelFactor1<Pose3>(model, X), msrc_(src), mtgt_(tgt) {}
        ~Point2pFactor() override {}

        /// @return a deep copy of this factor
        virtual NonlinearFactor::shared_ptr clone() const
        {
            return boost::static_pointer_cast<NonlinearFactor>(
                NonlinearFactor::shared_ptr(new Point2pFactor(*this)));
        }

        inline Matrix3d skew(const Vector3d &axis) const
        {
            Matrix3d skew_matrix = Matrix3d::Identity();

            skew_matrix << 0, -axis(2, 0), axis(1, 0),
                axis(2, 0), 0, -axis(0, 0),
                -axis(1, 0), axis(0, 0), 0;

            return skew_matrix;
        }

        virtual Vector evaluateError(const Pose3 &X, OptionalMatrixType H = OptionalNone) const override
        {
            const Rot3 &R = X.rotation();
            Vector3 mx = R * mtgt_ + X.translation();
            Vector3 error = mx - msrc_;
            if (H)
            {
                *H = gtsam::Matrix(3, 6);
                (*H).block(0, 0, 3, 3) = -X.rotation().matrix() * skew(mtgt_);
                (*H).block(0, 3, 3, 3) = X.rotation().matrix();
            }
            return error;
        }

        Vector3d msrc_, mtgt_; ///< src and tgt measurements
    };

    class GNCPointCloudRegister
    {
    public:
        GNCPointCloudRegister(string settings_path);

        void registerPointCloudGNC(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2, Matrix3d &R12_ransac, Vector3d &t12_ransac);

        int gnc_verbosity_level;
    };

}
