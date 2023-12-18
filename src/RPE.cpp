#include "RPE.hpp"

namespace RPE
{
    RPE::RPE(string settings_path, const ros::NodeHandle &nh_) : nh(nh_)
    {
        cv::FileStorage settings(settings_path, cv::FileStorage::READ);

        settings["baseline"] >> baseline;
        settings["fx"] >> fx;
        settings["fy"] >> fy;
        settings["cx"] >> cx;
        settings["cy"] >> cy;
        settings["enable_subpix_corner_refinement"] >> enable_subpix_corner_refinement;

        settings["feature_detector_type"] >> feature_detector_type;
        switch (feature_detector_type)
        {
        case FeatureDetectorType::GFTT:
        {
            int max_corner = settings["GFTT.max_corner"];
            double quality_level = settings["GFTT.quality_level"];
            double min_distance = settings["GFTT.min_distance"];
            int block_size = settings["GFTT.block_size"];
            bool use_harris = (int)settings["GFTT.use_harris"];
            double k = settings["GFTT.k"];

            feature_detector = cv::GFTTDetector::create(max_corner, quality_level, min_distance, block_size, use_harris, k);
            break;
        }
        case FeatureDetectorType::SIFT:
        {
            int n_features = settings["SIFT.n_features"];
            int n_octave_layers = settings["SIFT.n_octave_layers"];
            double contrast_threshold = settings["SIFT.contrast_threshold"];
            double edge_threshold = settings["SIFT.edge_threshold"];
            double sigma = settings["SIFT.sigma"];
            bool enable_precise_upscale = (int)settings["SIFT.enable_precise_upscale"];

            feature_detector = cv::SIFT::create(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma, enable_precise_upscale);
            break;
        }
        case FeatureDetectorType::HLOC:
        {
            hloc_feature_detector = nh.serviceClient<rpe::FeatureDetection>("/feature_detection");
            break;
        }
        default:
            LOG(FATAL) << "Unknown feature detector type #^#";
        }

        if (feature_detector_type != FeatureDetectorType::HLOC)
        {
            settings["descriptor_extractor_type"] >> descriptor_extractor_type;
            switch (descriptor_extractor_type)
            {
            case DescriptorExtractorType::SIFT:
            {
                int n_features = settings["SIFT.n_features"];
                int n_octave_layers = settings["SIFT.n_octave_layers"];
                double contrast_threshold = settings["SIFT.contrast_threshold"];
                double edge_threshold = settings["SIFT.edge_threshold"];
                double sigma = settings["SIFT.sigma"];
                int descriptor_type = settings["SIFT.descriptor_type"];
                bool enable_precise_upscale = (int)settings["SIFT.enable_precise_upscale"];

                descriptor_extractor = cv::SIFT::create(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma, descriptor_type, enable_precise_upscale);
                break;
            }
            case DescriptorExtractorType::ORB:
            {
                int n_features = settings["ORB.n_features"];
                double scale_factor = settings["ORB.scale_factor"];
                int n_levels = settings["ORB.n_levels"];
                int edge_threshold = settings["ORB.edge_threshold"];
                int first_level = settings["ORB.first_level"];
                int WTA_K = settings["ORB.WTA_K"];
                int score_type_ = settings["ORB.score_type"];
                cv::ORB::ScoreType score_type;
                switch (score_type_)
                {
                case 0:
                    score_type = cv::ORB::ScoreType::HARRIS_SCORE;
                    break;
                case 1:
                    score_type = cv::ORB::ScoreType::FAST_SCORE;
                    break;
                default:
                    LOG(FATAL) << "Unknown score type for ORB extractor #^#";
                }
                int patch_size = settings["ORB.patch_size"];
                int fast_threshold = settings["ORB.fast_threshold"];

                descriptor_extractor = cv::ORB::create(n_features, scale_factor, n_levels, edge_threshold, first_level, WTA_K, score_type, patch_size, fast_threshold);
                break;
            }
            default:
                LOG(FATAL) << "Unknown descriptor extractor type #^#";
            }
        }

        matcher = make_unique<Matcher>(settings_path);

        ransacer = make_unique<opengv::sac::Ransac<SacProblem>>();
        settings["ransac_max_iterations"] >> ransacer->max_iterations_;
        settings["ransac_probability"] >> ransacer->probability_;
        settings["ransac_threshold"] >> ransacer->threshold_;
        settings["ransac_min_inliers"] >> ransac_min_inliers;
        settings["ransac_verbosity_level"] >> ransac_verbosity_level;
    }

    bool RPE::estimate(const cv::Mat &img1_l_, const cv::Mat &img1_r_,
                       const cv::Mat &img2_l_, const cv::Mat &img2_r_,
                       Matrix3d &R12_, Vector3d &t12_)
    {
        LOG(INFO) << "============= RPE Estimation =============";
        auto t0 = chrono::high_resolution_clock::now();

        img1_l = img1_l_;
        img1_r = img1_r_;
        img2_l = img2_l_;
        img2_r = img2_r_;

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img1_l, img1_l);
        clahe->apply(img1_r, img1_r);
        clahe->apply(img2_l, img2_l);
        clahe->apply(img2_r, img2_r);

        auto t1 = chrono::high_resolution_clock::now();

        kps1_l.clear();
        kps1_r.clear();
        kps2_l.clear();
        kps2_r.clear();

        kps1_l_ori.clear();
        kps1_r_ori.clear();
        kps2_l_ori.clear();
        kps2_r_ori.clear();

        kps1_l_stereo.clear();
        kps1_r_stereo.clear();
        kps2_l_stereo.clear();
        kps2_r_stereo.clear();

        kps1_l_desc.clear();
        kps2_l_desc.clear();

        /*******************************************************/
        /****** Step 1: Feature and Descriptor Extraction ******/
        /*******************************************************/

        auto t2 = chrono::high_resolution_clock::now();
        cv::Mat desc1_l_, desc1_r_, desc2_l_, desc2_r_;
        if (feature_detector_type == FeatureDetectorType::HLOC)
        {
            cv::Mat kps1_l_, kps1_r_, kps2_l_, kps2_r_;

            rpe::FeatureDetection feature_fetection_srv;
            cv_bridge::CvImage img1_l_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img1_l);
            cv_bridge::CvImage img1_r_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img1_r);
            cv_bridge::CvImage img2_l_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img2_l);
            cv_bridge::CvImage img2_r_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img2_r);

            // Call the HLOC detection service
            feature_fetection_srv.request.img = *(img1_l_msg.toImageMsg());
            if (hloc_feature_detector.call(feature_fetection_srv))
            {
                try
                {
                    kps1_l_ = cv_bridge::toCvCopy(feature_fetection_srv.response.kps)->image;
                    desc1_l_ = cv_bridge::toCvCopy(feature_fetection_srv.response.desc)->image;
                }
                catch (const cv_bridge::Exception &e)
                {
                    LOG(ERROR) << "cv_bridge exception: " << string(e.what());
                    return false;
                }
            }
            else
            {
                LOG(ERROR) << "Failed to call service feature_detection #^#";
                return false;
            }
            feature_fetection_srv.request.img = *(img1_r_msg.toImageMsg());
            if (hloc_feature_detector.call(feature_fetection_srv))
            {
                try
                {
                    kps1_r_ = cv_bridge::toCvCopy(feature_fetection_srv.response.kps)->image;
                    desc1_r_ = cv_bridge::toCvCopy(feature_fetection_srv.response.desc)->image;
                }
                catch (const cv_bridge::Exception &e)
                {
                    LOG(ERROR) << "cv_bridge exception: " << string(e.what());
                    return false;
                }
            }
            else
            {
                LOG(ERROR) << "Failed to call service feature_detection #^#";
                return false;
            }
            feature_fetection_srv.request.img = *(img2_l_msg.toImageMsg());
            if (hloc_feature_detector.call(feature_fetection_srv))
            {
                try
                {
                    kps2_l_ = cv_bridge::toCvCopy(feature_fetection_srv.response.kps)->image;
                    desc2_l_ = cv_bridge::toCvCopy(feature_fetection_srv.response.desc)->image;
                }
                catch (const cv_bridge::Exception &e)
                {
                    LOG(ERROR) << "cv_bridge exception: " << string(e.what());
                    return false;
                }
            }
            else
            {
                LOG(ERROR) << "Failed to call service feature_detection #^#";
                return false;
            }
            feature_fetection_srv.request.img = *(img2_r_msg.toImageMsg());
            if (hloc_feature_detector.call(feature_fetection_srv))
            {
                try
                {
                    kps2_r_ = cv_bridge::toCvCopy(feature_fetection_srv.response.kps)->image;
                    desc2_r_ = cv_bridge::toCvCopy(feature_fetection_srv.response.desc)->image;
                }
                catch (const cv_bridge::Exception &e)
                {
                    LOG(ERROR) << "cv_bridge exception: " << string(e.what());
                    return false;
                }
            }
            else
            {
                LOG(ERROR) << "Failed to call service feature_detection #^#";
                return false;
            }

            CHECK_EQ(kps1_l_.type(), CV_32F) << "Feature detection server returns a matrix of abnormal type #^#";
            CHECK_EQ(kps1_r_.type(), CV_32F) << "Feature detection server returns a matrix of abnormal type #^#";
            CHECK_EQ(kps2_l_.type(), CV_32F) << "Feature detection server returns a matrix of abnormal type #^#";
            CHECK_EQ(kps2_r_.type(), CV_32F) << "Feature detection server returns a matrix of abnormal type #^#";
            for (size_t i = 0; i < kps1_l_.rows; i++)
                kps1_l.emplace_back(cv::KeyPoint(kps1_l_.at<float>(i, 0), kps1_l_.at<float>(i, 1), -1));
            for (size_t i = 0; i < kps1_r_.rows; i++)
                kps1_r.emplace_back(cv::KeyPoint(kps1_r_.at<float>(i, 0), kps1_r_.at<float>(i, 1), -1));
            for (size_t i = 0; i < kps2_l_.rows; i++)
                kps2_l.emplace_back(cv::KeyPoint(kps2_l_.at<float>(i, 0), kps2_l_.at<float>(i, 1), -1));
            for (size_t i = 0; i < kps2_r_.rows; i++)
                kps2_r.emplace_back(cv::KeyPoint(kps2_r_.at<float>(i, 0), kps2_r_.at<float>(i, 1), -1));
        }
        else
        {
            feature_detector->detect(img1_l, kps1_l);
            feature_detector->detect(img1_r, kps1_r);
            feature_detector->detect(img2_l, kps2_l);
            feature_detector->detect(img2_r, kps2_r);

            t2 = chrono::high_resolution_clock::now();

            // Subpixel corner refinement
            if (enable_subpix_corner_refinement)
            {
                vector<cv::Point2f> kps1_l_ref, kps1_r_ref, kps2_l_ref, kps2_r_ref;
                cv::KeyPoint::convert(kps1_l, kps1_l_ref);
                cv::KeyPoint::convert(kps1_r, kps1_r_ref);
                cv::KeyPoint::convert(kps2_l, kps2_l_ref);
                cv::KeyPoint::convert(kps2_r, kps2_r_ref);

                cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);
                cv::cornerSubPix(img1_l, kps1_l_ref, cv::Size(5, 5), cv::Size(-1, -1), criteria);
                cv::cornerSubPix(img1_r, kps1_r_ref, cv::Size(5, 5), cv::Size(-1, -1), criteria);
                cv::cornerSubPix(img2_l, kps2_l_ref, cv::Size(5, 5), cv::Size(-1, -1), criteria);
                cv::cornerSubPix(img2_r, kps2_r_ref, cv::Size(5, 5), cv::Size(-1, -1), criteria);

                cv::KeyPoint::convert(kps1_l_ref, kps1_l);
                cv::KeyPoint::convert(kps1_r_ref, kps1_r);
                cv::KeyPoint::convert(kps2_l_ref, kps2_l);
                cv::KeyPoint::convert(kps2_r_ref, kps2_r);
            }

            descriptor_extractor->compute(img1_l, kps1_l, desc1_l_);
            descriptor_extractor->compute(img1_r, kps1_r, desc1_r_);
            descriptor_extractor->compute(img2_l, kps2_l, desc2_l_);
            descriptor_extractor->compute(img2_r, kps2_r, desc2_r_);
        }

        // Convert descriptor matrix to descriptor vector
        desc1_l.clear();
        desc1_r.clear();
        desc2_l.clear();
        desc2_r.clear();
        for (size_t i = 0; i < desc1_l_.rows; i++)
            desc1_l.emplace_back(desc1_l_.row(i));
        for (size_t i = 0; i < desc1_r_.rows; i++)
            desc1_r.emplace_back(desc1_r_.row(i));
        for (size_t i = 0; i < desc2_l_.rows; i++)
            desc2_l.emplace_back(desc2_l_.row(i));
        for (size_t i = 0; i < desc2_r_.rows; i++)
            desc2_r.emplace_back(desc2_r_.row(i));

        auto t3 = chrono::high_resolution_clock::now();

        // Backup originally extracted features
        kps1_l_ori = kps1_l;
        kps1_r_ori = kps1_r;
        kps2_l_ori = kps2_l;
        kps2_r_ori = kps2_r;

        if (kps1_l.size() < 100 || kps1_r.size() < 100 || kps2_l.size() < 100 || kps2_r.size() < 100)
        {
            LOG(WARNING) << "Only " << kps1_l.size() << "/"
                         << kps1_r.size() << "/"
                         << kps2_l.size() << "/"
                         << kps2_r.size() << " features detected #^#";
            return false;
        }

        /*************************************************/
        /****** Step 2: Stereo and Feature Matching ******/
        /*************************************************/

        extractKpsDepth(kps1_l, kps1_r, desc1_l, desc1_r, kps3d1);
        extractKpsDepth(kps2_l, kps2_r, desc2_l, desc2_r, kps3d2);
        kps3d1_debug = kps3d1;
        kps3d2_debug = kps3d2;

        auto t4 = chrono::high_resolution_clock::now();

        // Backup features after stereo matching
        kps1_l_stereo = kps1_l;
        kps1_r_stereo = kps1_r;
        kps2_l_stereo = kps2_l;
        kps2_r_stereo = kps2_r;

        if (kps1_l.size() < 50 || kps1_r.size() < 50 || kps2_l.size() < 50 || kps2_r.size() < 50)
        {
            LOG(WARNING) << "Only " << kps1_l.size() << "/" << kps2_l.size() << " stereo matches #^#";
            return false;
        }

        matchByDescriptor(kps1_l, kps2_l, desc1_l, desc2_l, kps3d1, kps3d2);

        auto t5 = chrono::high_resolution_clock::now();

        // Backup features after descriptor matching
        kps1_l_desc = kps1_l;
        kps2_l_desc = kps2_l;

        if (kps1_l.size() < 15 || kps2_l.size() < 15)
        {
            LOG(WARNING) << "Only " << kps1_l.size() << " descriptor matches #^#";
            return false;
        }

        /********************************************/
        /****** Step 3: Geometric Verification ******/
        /********************************************/

        bool recover_pose_success = recoverPoseArun(kps3d1, kps3d2, R12, t12);

        auto t6 = chrono::high_resolution_clock::now();

        if (recover_pose_success)
        {
            R12_ = R12;
            t12_ = t12;

            LOG(INFO) << "Recover pose succeeded :)";
        }

        if (feature_detector_type == FeatureDetectorType::HLOC)
        {
            LOG(INFO) << "[Histogram Equalization]: "
                      << chrono::duration_cast<chrono::duration<double>>(t1 - t0).count() * 1000 << " ms";
            LOG(INFO) << "[Kps and Desc Extraction]: "
                      << chrono::duration_cast<chrono::duration<double>>(t3 - t1).count() * 1000 << " ms";
            LOG(INFO) << "[Stereo Matching]: "
                      << chrono::duration_cast<chrono::duration<double>>(t4 - t3).count() * 1000 << " ms";
            LOG(INFO) << "[Descriptor Matching]: "
                      << chrono::duration_cast<chrono::duration<double>>(t5 - t4).count() * 1000 << " ms";
            LOG(INFO) << "[Geometric Verification]: "
                      << chrono::duration_cast<chrono::duration<double>>(t6 - t5).count() * 1000 << " ms";
        }
        else
        {
            LOG(INFO) << "[Histogram Equalization]: "
                      << chrono::duration_cast<chrono::duration<double>>(t1 - t0).count() * 1000 << " ms";
            LOG(INFO) << "[Feature Detection]: "
                      << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() * 1000 << " ms";
            LOG(INFO) << "[Descriptor Extraction]: "
                      << chrono::duration_cast<chrono::duration<double>>(t3 - t2).count() * 1000 << " ms";
            LOG(INFO) << "[Stereo Matching]: "
                      << chrono::duration_cast<chrono::duration<double>>(t4 - t3).count() * 1000 << " ms";
            LOG(INFO) << "[Descriptor Matching]: "
                      << chrono::duration_cast<chrono::duration<double>>(t5 - t4).count() * 1000 << " ms";
            LOG(INFO) << "[Geometric Verification]: "
                      << chrono::duration_cast<chrono::duration<double>>(t6 - t5).count() * 1000 << " ms";
        }

        return recover_pose_success;
    }

    void RPE::extractKpsDepth(vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                              vector<cv::Mat> &desc_l, vector<cv::Mat> &desc_r,
                              vector<Vector3d> &kps3d)
    {
        vector<int> match(kps_l.size(), -1);
        matcher->matchStereo(kps_l, kps_r, desc_l, desc_r, match);

        rearrangeMatchedVec(match, kps_l, kps_r);
        rearrangeMatchedVec(match, desc_l, desc_r);

        // Compute depth
        kps3d.resize(kps_l.size());
        for (size_t i = 0; i < kps_l.size(); i++)
        {
            const double ul = kps_l[i].pt.x;
            const double ur = kps_r[i].pt.x;
            const double vl = kps_l[i].pt.y;

            double disparity = ul - ur;
            double depth = baseline * fx / disparity;

            const double x = (ul - cx) * depth / fx;
            const double y = (vl - cy) * depth / fy;

            kps3d[i] << x, y, depth;
        }
    }

    void RPE::matchByDescriptor(vector<cv::KeyPoint> &kps1, vector<cv::KeyPoint> &kps2,
                                vector<cv::Mat> &desc1, vector<cv::Mat> &desc2,
                                vector<Vector3d> &kps3d1, vector<Vector3d> &kps3d2)
    {
        vector<int> match(kps1.size(), -1);
        matcher->matchByDesc(kps1, kps2, desc1, desc2, match);

        rearrangeMatchedVec(match, kps1, kps2);
        rearrangeMatchedVec(match, desc1, desc2);
        rearrangeMatchedVec(match, kps3d1, kps3d2);
    }

    template <typename T>
    inline void RPE::rearrangeMatchedVec(const vector<int> &match, vector<T> &vec1, vector<T> &vec2)
    {
        vector<T> vec2_temp;
        size_t idx = 0;
        for (size_t i = 0; i < match.size(); i++)
        {
            if (match[i] >= 0)
            {
                // Rearrange elements
                vec1[idx++] = vec1[i]; // Borrowed from VINS-Fusion
                vec2_temp.emplace_back(vec2[match[i]]);
            }
        }
        vec1.resize(idx);
        vec2 = vec2_temp;
    }

    bool RPE::recoverPoseArun(const vector<Vector3d> &kps3d1_, const vector<Vector3d> &kps3d2_,
                              Matrix3d &R12, Vector3d &t12)
    {
        CHECK_EQ(kps3d1_.size(), kps3d2_.size()) << "kps3d1.size() != kps3d2.size() in recoverPoseArun #^#";

        const size_t n_matches = kps3d1_.size();
        opengv::points_t kps3d1, kps3d2;
        kps3d1.resize(n_matches);
        kps3d2.resize(n_matches);
        for (size_t i = 0; i < n_matches; i++)
        {
            kps3d1[i] = kps3d1_[i];
            kps3d2[i] = kps3d2_[i];
        }
        opengv::point_cloud::PointCloudAdapter adapter(kps3d1, kps3d2);

        ransacer->sac_model_ = make_shared<SacProblem>(adapter);
        bool ransac_success = ransacer->computeModel(ransac_verbosity_level);

        if (ransac_success)
        {
            if (ransacer->inliers_.size() >= ransac_min_inliers)
            {
                opengv::transformation_t T12 = ransacer->model_coefficients_;

                R12 = T12.block<3, 3>(0, 0);
                t12 = T12.col(3);

                return true;
            }
            else
            {
                LOG(WARNING) << "Only " << ransacer->inliers_.size() << " RANSAC inliers #^#";
                return false;
            }
        }
        else
        {
            LOG(WARNING) << "RANSAC failed #^#";
            return false;
        }
    }
}