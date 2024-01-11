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

        settings["enable_hloc_matcher"] >> enable_hloc_matcher;
        if (enable_hloc_matcher)
        {
            hloc_matcher = nh.serviceClient<rpe::Matching>("/hloc_matching");
        }
        else
        {
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
                hloc_feature_detector = nh.serviceClient<rpe::FeatureDetection>("/hloc_feature_detection");
                break;
            }
            default:
                LOG(FATAL) << "Unknown feature detector type #^#";
            }

            if (feature_detector_type != FeatureDetectorType::HLOC)
            {
                settings["enable_subpix_corner_refinement"] >> enable_subpix_corner_refinement;

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
        }

        settings["far_pt_thr"] >> far_pt_thr;

        settings["ransac_verbosity_level"] >> ransac_verbosity_level;
        settings["ransac_min_inliers"] >> ransac_min_inliers;

        mono_ransacer = make_unique<opengv::sac::Ransac<SacProblemMono>>();
        settings["mono_ransac_max_iterations"] >> mono_ransacer->max_iterations_;
        settings["mono_ransac_threshold"] >> mono_ransacer->threshold_;

        stereo_ransacer = make_unique<opengv::sac::Ransac<SacProblemStereo>>();
        settings["stereo_ransac_max_iterations"] >> stereo_ransacer->max_iterations_;
        settings["stereo_ransac_threshold"] >> stereo_ransacer->threshold_;

        settings["enable_nlopt"] >> enable_nlopt;

        R_solver = make_unique<RSolver>();
        A_solver = make_unique<ASolver>(settings_path);
    }

    bool RPE::estimate(const cv::Mat &img1_l_, const cv::Mat &img1_r_,
                       const cv::Mat &img2_l_, const cv::Mat &img2_r_,
                       Matrix3d &R12, Vector3d &t12)
    {
        LOG(INFO) << "============= RPE =============";
        auto t0 = chrono::high_resolution_clock::now(), t1 = chrono::high_resolution_clock::now(), t2 = chrono::high_resolution_clock::now(), t3 = chrono::high_resolution_clock::now(), t4 = chrono::high_resolution_clock::now(), t5 = chrono::high_resolution_clock::now();

        img1_l = img1_l_;
        img1_r = img1_r_;
        img2_l = img2_l_;
        img2_r = img2_r_;

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img1_l, img1_l);
        clahe->apply(img1_r, img1_r);
        clahe->apply(img2_l, img2_l);
        clahe->apply(img2_r, img2_r);

        t1 = chrono::high_resolution_clock::now();

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

        kps1_l_matched.clear();
        kps2_l_matched.clear();

        kps1_l_refined.clear();
        kps2_l_refined.clear();

        vector<int> match12, match21;
        if (enable_hloc_matcher)
        {
            /*******************************************************/
            /****** Feature Detection and Matching Using HLOC ******/
            /*******************************************************/

            cv::Mat kps1_l_, kps1_r_, kps2_l_, kps2_r_;
            vector<int> match_stereo1, match_stereo2;

            // Call the HLOC matching service
            rpe::Matching matching_srv;
            cv_bridge::CvImage img1_l_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img1_l);
            cv_bridge::CvImage img1_r_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img1_r);
            cv_bridge::CvImage img2_l_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img2_l);
            cv_bridge::CvImage img2_r_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img2_r);
            matching_srv.request.img1_l = *(img1_l_msg.toImageMsg());
            matching_srv.request.img1_r = *(img1_r_msg.toImageMsg());
            matching_srv.request.img2_l = *(img2_l_msg.toImageMsg());
            matching_srv.request.img2_r = *(img2_r_msg.toImageMsg());

            if (hloc_matcher.call(matching_srv))
            {
                try
                {
                    kps1_l_ = cv_bridge::toCvCopy(matching_srv.response.kps1_l)->image;
                    kps1_r_ = cv_bridge::toCvCopy(matching_srv.response.kps1_r)->image;
                    kps2_l_ = cv_bridge::toCvCopy(matching_srv.response.kps2_l)->image;
                    kps2_r_ = cv_bridge::toCvCopy(matching_srv.response.kps2_r)->image;

                    match_stereo1 = matching_srv.response.match_stereo1.data;
                    match_stereo2 = matching_srv.response.match_stereo2.data;
                    match12 = matching_srv.response.match12.data;
                }
                catch (const cv_bridge::Exception &e)
                {
                    LOG(ERROR) << "cv_bridge exception: " << string(e.what());
                    return false;
                }
            }
            else
            {
                LOG(ERROR) << "Failed to call service hloc_matching #^#";
                return false;
            }

            CHECK_EQ(kps1_l_.type(), CV_32F);
            CHECK_EQ(kps1_r_.type(), CV_32F);
            CHECK_EQ(kps2_l_.type(), CV_32F);
            CHECK_EQ(kps2_r_.type(), CV_32F);
            for (size_t i = 0; i < kps1_l_.rows; i++)
                kps1_l.emplace_back(cv::KeyPoint(kps1_l_.at<float>(i, 0), kps1_l_.at<float>(i, 1), -1));
            for (size_t i = 0; i < kps1_r_.rows; i++)
                kps1_r.emplace_back(cv::KeyPoint(kps1_r_.at<float>(i, 0), kps1_r_.at<float>(i, 1), -1));
            for (size_t i = 0; i < kps2_l_.rows; i++)
                kps2_l.emplace_back(cv::KeyPoint(kps2_l_.at<float>(i, 0), kps2_l_.at<float>(i, 1), -1));
            for (size_t i = 0; i < kps2_r_.rows; i++)
                kps2_r.emplace_back(cv::KeyPoint(kps2_r_.at<float>(i, 0), kps2_r_.at<float>(i, 1), -1));

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

            CHECK_EQ(match_stereo1.size(), kps1_l.size());
            CHECK_EQ(match_stereo2.size(), kps2_l.size());
            CHECK_EQ(match12.size(), kps1_l.size());

            // Compute match2->1 from match1->2
            match21 = vector<int>(kps2_l.size(), -1);
            for (size_t i = 0; i < match12.size(); i++)
                if (match12[i] >= 0)
                    match21[match12[i]] = i;

            // Compute depth1 given stereo matches from HLOC
            extractKpsDepth(match_stereo1, kps1_l, kps1_r, kps3d1);
            kps3d1_stereo = kps3d1;

            size_t idx = 0;
            for (size_t i = 0; i < match_stereo1.size(); i++)
            {
                if (match_stereo1[i] >= 0)
                {
                    // Rearrange elements
                    match12[idx] = match12[i]; // Borrowed from VINS-Fusion
                    if (match12[i] >= 0)
                        match21[match12[i]] = idx++;
                    else
                        idx++;
                }
                else
                {
                    if (match12[i] >= 0)
                        match21[match12[i]] = -1;
                }
            }
            match12.resize(idx);

            // Compute depth2 given stereo matches from HLOC
            extractKpsDepth(match_stereo2, kps2_l, kps2_r, kps3d2);
            kps3d2_stereo = kps3d2;

            idx = 0;
            for (size_t i = 0; i < match_stereo2.size(); i++)
            {
                if (match_stereo2[i] >= 0)
                {
                    // Rearrange elements
                    match21[idx] = match21[i]; // Borrowed from VINS-Fusion
                    if (match21[i] >= 0)
                        match12[match21[i]] = idx++;
                    else
                        idx++;
                }
                else
                {
                    if (match21[i] >= 0)
                        match12[match21[i]] = -1;
                }
            }
            match21.resize(idx);

            for (size_t i = 0; i < match12.size(); i++)
                if (match12[i] >= 0)
                    CHECK_EQ(match21[match12[i]], i);
            for (size_t i = 0; i < match21.size(); i++)
                if (match21[i] >= 0)
                    CHECK_EQ(match12[match21[i]], i);

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

            // Filter out far points
            CHECK_EQ(match12.size(), kps3d1.size());
            for (size_t i = 0; i < match12.size(); i++)
                if (match12[i] >= 0)
                    if (kps3d1[i].z() > far_pt_thr || kps3d2[match12[i]].z() > far_pt_thr)
                        match12[i] = -1;

            rearrangeMatchedVec(match12, kps1_l, kps2_l);
            rearrangeMatchedVec(match12, kps3d1, kps3d2);

            // Backup features after descriptor matching
            kps1_l_matched = kps1_l;
            kps2_l_matched = kps2_l;

            if (kps1_l.size() < 15 || kps2_l.size() < 15)
            {
                LOG(WARNING) << "Only " << kps1_l.size() << " descriptor matches #^#";
                return false;
            }

            t5 = chrono::high_resolution_clock::now();
        }
        else
        {
            /***********************************************/
            /****** Feature and Descriptor Extraction ******/
            /***********************************************/

            cv::Mat desc1_l_, desc1_r_, desc2_l_, desc2_r_;
            if (feature_detector_type == FeatureDetectorType::HLOC)
            {
                cv::Mat kps1_l_, kps1_r_, kps2_l_, kps2_r_;

                // Call the HLOC detection service
                rpe::FeatureDetection feature_detection_srv;
                cv_bridge::CvImage img1_l_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img1_l);
                cv_bridge::CvImage img1_r_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img1_r);
                cv_bridge::CvImage img2_l_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img2_l);
                cv_bridge::CvImage img2_r_msg = cv_bridge::CvImage(std_msgs::Header(), image_encodings::MONO8, img2_r);

                feature_detection_srv.request.img = *(img1_l_msg.toImageMsg());
                if (hloc_feature_detector.call(feature_detection_srv))
                {
                    try
                    {
                        kps1_l_ = cv_bridge::toCvCopy(feature_detection_srv.response.kps)->image;
                        desc1_l_ = cv_bridge::toCvCopy(feature_detection_srv.response.desc)->image;
                    }
                    catch (const cv_bridge::Exception &e)
                    {
                        LOG(ERROR) << "cv_bridge exception: " << string(e.what());
                        return false;
                    }
                }
                else
                {
                    LOG(ERROR) << "Failed to call service hloc_feature_detection #^#";
                    return false;
                }
                feature_detection_srv.request.img = *(img1_r_msg.toImageMsg());
                if (hloc_feature_detector.call(feature_detection_srv))
                {
                    try
                    {
                        kps1_r_ = cv_bridge::toCvCopy(feature_detection_srv.response.kps)->image;
                        desc1_r_ = cv_bridge::toCvCopy(feature_detection_srv.response.desc)->image;
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
                feature_detection_srv.request.img = *(img2_l_msg.toImageMsg());
                if (hloc_feature_detector.call(feature_detection_srv))
                {
                    try
                    {
                        kps2_l_ = cv_bridge::toCvCopy(feature_detection_srv.response.kps)->image;
                        desc2_l_ = cv_bridge::toCvCopy(feature_detection_srv.response.desc)->image;
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
                feature_detection_srv.request.img = *(img2_r_msg.toImageMsg());
                if (hloc_feature_detector.call(feature_detection_srv))
                {
                    try
                    {
                        kps2_r_ = cv_bridge::toCvCopy(feature_detection_srv.response.kps)->image;
                        desc2_r_ = cv_bridge::toCvCopy(feature_detection_srv.response.desc)->image;
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

                CHECK_EQ(kps1_l_.type(), CV_32F);
                CHECK_EQ(kps1_r_.type(), CV_32F);
                CHECK_EQ(kps2_l_.type(), CV_32F);
                CHECK_EQ(kps2_r_.type(), CV_32F);
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

            t3 = chrono::high_resolution_clock::now();

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

            /*****************************************/
            /****** Stereo and Feature Matching ******/
            /*****************************************/

            extractKpsDepth(kps1_l, kps1_r, desc1_l, desc1_r, kps3d1);
            extractKpsDepth(kps2_l, kps2_r, desc2_l, desc2_r, kps3d2);
            kps3d1_stereo = kps3d1;
            kps3d2_stereo = kps3d2;

            t4 = chrono::high_resolution_clock::now();

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

            t5 = chrono::high_resolution_clock::now();

            // Backup features after descriptor matching
            kps1_l_matched = kps1_l;
            kps2_l_matched = kps2_l;

            if (kps1_l.size() < 15 || kps2_l.size() < 15)
            {
                LOG(WARNING) << "Only " << kps1_l.size() << " descriptor matches #^#";
                return false;
            }
        }

        /************************************/
        /****** Geometric Verification ******/
        /************************************/

        bool recover_pose_success = false;
        if (geometricVerificationNister(kps1_l, kps2_l, R12_mono))
        {
            if (recoverPose(kps3d1, kps3d2, R12_mono, R12_stereo, t12_stereo))
            {
                recover_pose_success = true;

                R12 = R12_stereo;
                t12 = t12_stereo;

                LOG(INFO) << "RANSAC pose recovery succeeded :)";
            }
        }

        auto t6 = chrono::high_resolution_clock::now();

        /****************************************************/
        /****** Improve Matching with Pose Information ******/
        /****************************************************/

        CHECK_EQ(match12.size(), kps3d1_stereo.size());
        CHECK_EQ(match21.size(), kps3d2_stereo.size());
        alternateOpt(kps3d1_stereo, kps3d2_stereo, match12, R12, t12);

        vector<cv::KeyPoint> kps1_l_refined_, kps2_l_refined_;
        kps1_l_refined_ = kps1_l_stereo;
        kps2_l_refined_ = kps2_l_stereo;
        rearrangeMatchedVec(match12, kps1_l_refined_, kps2_l_refined_);

        // Backup features after alternate optimization
        kps1_l_refined = kps1_l_refined_;
        kps2_l_refined = kps2_l_refined_;

        // vector<int> match12_(kps3d1.size());
        // for (size_t i = 0; i < match12_.size(); i++)
        //     match12_[i] = i;
        // alternateOpt(kps3d1, kps3d2, match12_, R12, t12);

        // rearrangeMatchedVec(match12_, kps1_l, kps2_l);

        // // Backup features after alternate optimization
        // kps1_l_refined = kps1_l;
        // kps2_l_refined = kps2_l;

        /********************************/
        /****** Print Running Time ******/
        /********************************/

        if (enable_hloc_matcher)
        {
            LOG(INFO) << "[Histogram Equalization]: "
                      << chrono::duration_cast<chrono::duration<double>>(t1 - t0).count() * 1000 << " ms";
            LOG(INFO) << "[HLOC Matching]: "
                      << chrono::duration_cast<chrono::duration<double>>(t5 - t1).count() * 1000 << " ms";
            LOG(INFO) << "[Geometric Verification]: "
                      << chrono::duration_cast<chrono::duration<double>>(t6 - t5).count() * 1000 << " ms";
        }
        else
        {
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
        }

        return recover_pose_success;
    }

    void RPE::extractKpsDepth(vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                              vector<cv::Mat> &desc_l, vector<cv::Mat> &desc_r,
                              vector<Vector3d> &kps3d)
    {
        vector<int> stereo_match(kps_l.size(), -1);
        matcher->matchStereo(kps_l, kps_r, desc_l, desc_r, stereo_match);

        rearrangeMatchedVec(stereo_match, kps_l, kps_r);
        rearrangeMatchedVec(stereo_match, desc_l, desc_r);

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

    void RPE::extractKpsDepth(vector<int> &stereo_match, vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                              vector<Vector3d> &kps3d)
    {
        // Filter out points with inf depth
        for (size_t i = 0; i < kps_l.size(); i++)
        {
            if (stereo_match[i] >= 0)
            {
                const double ul = kps_l[i].pt.x;
                const double ur = kps_r[stereo_match[i]].pt.x;

                double disparity = ul - ur;
                if (-1e-5 < disparity && disparity < 1e-5)
                    stereo_match[i] = -1;
            }
        }

        rearrangeMatchedVec(stereo_match, kps_l, kps_r);

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
    inline void RPE::rearrangeMatchedVec(const vector<int> &match12, vector<T> &vec1, vector<T> &vec2)
    {
        vector<T> vec2_temp;
        size_t idx = 0;
        for (size_t i = 0; i < match12.size(); i++)
        {
            if (match12[i] >= 0)
            {
                // Rearrange elements
                vec1[idx++] = vec1[i]; // Borrowed from VINS-Fusion
                vec2_temp.emplace_back(vec2[match12[i]]);
            }
        }
        vec1.resize(idx);
        vec2 = vec2_temp;
    }

    bool RPE::geometricVerificationNister(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2, Matrix3d &R12_mono)
    {
        CHECK_EQ(kps1.size(), kps2.size());

        const size_t n_matches = kps1.size();
        opengv::bearingVectors_t bearing_vecs1, bearing_vecs2;
        bearing_vecs1.resize(n_matches);
        bearing_vecs2.resize(n_matches);
        for (size_t i = 0; i < n_matches; i++)
        {
            bearing_vecs1[i] = opengv::bearingVector_t(kps1[i].pt.x, kps1[i].pt.y, 1.0).normalized();
            bearing_vecs2[i] = opengv::bearingVector_t(kps2[i].pt.x, kps2[i].pt.y, 1.0).normalized();
        }
        opengv::relative_pose::CentralRelativeAdapter adapter(bearing_vecs1, bearing_vecs2);

        mono_ransacer->sac_model_ = make_shared<SacProblemMono>(adapter, SacProblemMono::Algorithm::NISTER);
        bool ransac_success = mono_ransacer->computeModel(ransac_verbosity_level);

        if (ransac_success)
        {
            if (mono_ransacer->inliers_.size() >= ransac_min_inliers)
            {
                LOG(INFO) << "Mono RANSAC has " << mono_ransacer->inliers_.size() << " inliers";

                opengv::transformation_t T12 = mono_ransacer->model_coefficients_;
                R12_mono = T12.block<3, 3>(0, 0);

                return true;
            }
            else
            {
                LOG(WARNING) << "Only " << mono_ransacer->inliers_.size() << " mono RANSAC inliers #^#";
                return false;
            }
        }
        else
        {
            LOG(WARNING) << "Mono RANSAC failed #^#";
            return false;
        }
    }

    bool RPE::recoverPose(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2, const Matrix3d &R12_mono,
                          Matrix3d &R12_stereo, Vector3d &t12_stereo)
    {
        CHECK_EQ(kps3d1.size(), kps3d2.size());

        // Compute transform using Arun's 3-point method + RANSAC
        const size_t n_matches = kps3d1.size();
        opengv::points_t kps3d1_, kps3d2_;
        kps3d1_.resize(n_matches);
        kps3d2_.resize(n_matches);
        for (size_t i = 0; i < n_matches; i++)
        {
            kps3d1_[i] = kps3d1[i];
            kps3d2_[i] = kps3d2[i];
        }
        opengv::point_cloud::PointCloudAdapter adapter(kps3d1_, kps3d2_);
        adapter.setR12(R12_mono);

        stereo_ransacer->sac_model_ = make_shared<SacProblemStereo>(adapter);
        bool ransac_success = stereo_ransacer->computeModel(ransac_verbosity_level);
        const size_t n_inliers = stereo_ransacer->inliers_.size();

        Matrix3d R12_;
        Vector3d t12_;
        if (ransac_success)
        {
            if (n_inliers >= ransac_min_inliers)
            {
                LOG(INFO) << "Stereo RANSAC has " << n_inliers << " inliers";

                opengv::transformation_t T12 = stereo_ransacer->model_coefficients_;
                R12_ = T12.block<3, 3>(0, 0);
                t12_ = T12.col(3);
            }
            else
            {
                LOG(WARNING) << "Only " << n_inliers << " stereo RANSAC inliers #^#";
                return false;
            }
        }
        else
        {
            LOG(WARNING) << "Stereo RANSAC failed #^#";
            return false;
        }

        if (enable_nlopt)
        {
            // Compute transform using non-linear optimization with RANSAC inliers
            opengv::points_t kps3d1_inlier, kps3d2_inlier;
            kps3d1_inlier.resize(n_inliers);
            kps3d2_inlier.resize(n_inliers);
            for (size_t i = 0; i < n_inliers; i++)
            {
                kps3d1_inlier[i] = kps3d1[stereo_ransacer->inliers_[i]];
                kps3d2_inlier[i] = kps3d2[stereo_ransacer->inliers_[i]];
            }
            opengv::point_cloud::PointCloudAdapter adapter_inlier(kps3d1_inlier, kps3d2_inlier);
            adapter_inlier.setR12(R12_mono);

            opengv::transformation_t T12 = opengv::point_cloud::optimize_nonlinear(adapter_inlier);
            R12_stereo = T12.block<3, 3>(0, 0);
            t12_stereo = T12.col(3);
        }
        else
        {
            R12_stereo = R12_;
            t12_stereo = t12_;
        }
        return true;
    }

    void RPE::alternateOpt(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2,
                           vector<int> &match12, Matrix3d &R12, Vector3d &t12)
    {
        // Convert the match vector to match matrix
        const size_t N1 = kps3d1.size();
        const size_t N2 = kps3d2.size();
        MatrixXd A(N2, N1), A_tilde(N2, N1);
        A.setZero();
        A_tilde.setZero();
        for (size_t i = 0; i < N1; i++)
            for (size_t j = 0; j < N2; j++)
                if (match12[i] == j)
                {
                    A(j, i) = 1;
                    A_tilde(j, i) = 1;
                }

        for (size_t iteration = 0; iteration < 1; iteration++)
        {
            A_solver->solve(kps3d1, kps3d2, R12, t12, A_tilde, A);
        }

        // Convert the match matrix back to match vector
        match12 = vector<int>(N1, -1);
        for (size_t i = 0; i < N1; i++)
            for (size_t j = 0; j < N2; j++)
                if (A(j, i))
                {
                    match12[i] = j;
                    break;
                }
    }
}