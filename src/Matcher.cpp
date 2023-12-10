#include "RPE.hpp"
#include "Matcher.hpp"

namespace RPE
{
    Matcher::Matcher(string settings_path)
    {
        cv::FileStorage settings(settings_path, cv::FileStorage::READ);

        settings["GRB_log_to_console"] >> GRB_log_to_console;
        settings["fx"] >> fx;

        settings["matcher_type"] >> matcher_type;

        RPE::DescriptorExtractorType desc_type;
        settings["descriptor_extractor_type"] >> desc_type;

        switch (matcher_type)
        {
        case MatcherType::CV_NN:
        {
            switch (desc_type)
            {
            case RPE::DescriptorExtractorType::SIFT:
            {
                norm_type = cv::NORM_L1;

                int descriptor_type = settings["SIFT.descriptor_type"];
                switch (descriptor_type)
                {
                case 0: // CV_8U
                    settings["SIFT.stereo_desc_dist_th_CV_8U"] >> stereo_desc_dist_th;
                    break;
                case 5: // CV_32F
                    settings["SIFT.stereo_desc_dist_th_CV_32F"] >> stereo_desc_dist_th;
                    break;
                default:
                    LOG(FATAL) << "Unknown descriptor element type for SIFT #^#";
                }
                break;
            }
            case RPE::DescriptorExtractorType::ORB:
            {
                settings["ORB.stereo_desc_dist_th"] >> stereo_desc_dist_th;

                int WTA_K = settings["ORB.WTA_K"];
                switch (WTA_K)
                {
                case 2:
                    norm_type = cv::NORM_HAMMING;
                    break;
                case 3:
                case 4:
                    norm_type = cv::NORM_HAMMING2;
                    break;
                default:
                    LOG(FATAL) << "Invalid WTA_K for ORB #^#";
                }
                break;
            }
            default:
                LOG(FATAL) << "Unknown descriptor extractor type #^#";
            }

            settings["enable_ratio_test"] >> enable_ratio_test;
            settings["lowe_ratio"] >> lowe_ratio;

            matcher_cv = cv::BFMatcher::create(norm_type);

            break;
        }
        case MatcherType::GNN:
        {
            switch (desc_type)
            {
            case RPE::DescriptorExtractorType::SIFT:
            {
                norm_type = cv::NORM_L1;

                int descriptor_type = settings["SIFT.descriptor_type"];
                switch (descriptor_type)
                {
                case 0: // CV_8U
                    settings["SIFT.stereo_desc_dist_th_CV_8U"] >> stereo_desc_dist_th;
                    settings["SIFT.desc_dist_th_CV_8U"] >> desc_dist_th;
                    break;
                case 5: // CV_32F
                    settings["SIFT.stereo_desc_dist_th_CV_32F"] >> stereo_desc_dist_th;
                    settings["SIFT.desc_dist_th_CV_32F"] >> desc_dist_th;
                    break;
                default:
                    LOG(FATAL) << "Unknown descriptor element type for SIFT #^#";
                }
                break;
            }
            case RPE::DescriptorExtractorType::ORB:
            {
                settings["ORB.stereo_desc_dist_th"] >> stereo_desc_dist_th;
                settings["ORB.desc_dist_th"] >> desc_dist_th;

                int WTA_K = settings["ORB.WTA_K"];
                switch (WTA_K)
                {
                case 2:
                    norm_type = cv::NORM_HAMMING;
                    break;
                case 3:
                case 4:
                    norm_type = cv::NORM_HAMMING2;
                    break;
                default:
                    LOG(FATAL) << "Invalid WTA_K for ORB #^#";
                }
                break;
            }
            default:
                LOG(FATAL) << "Unknown descriptor extractor type #^#";
            }

            settings["enable_consistency"] >> enable_consistency;
            settings["weight_desc_vs_consis"] >> weight_desc_vs_consis;
            settings["weight_len_vs_ang"] >> weight_len_vs_ang;
            settings["diff_len_th"] >> diff_len_th;
            settings["diff_ang_th"] >> diff_ang_th;
            settings["GRB_mip_focus"] >> GRB_mip_focus;
            settings["GRB_heuristics"] >> GRB_heuristics;
            settings["GRB_presolve"] >> GRB_presolve;
            settings["GRB_presparsify"] >> GRB_presparsify;
            settings["GRB_time_limit"] >> GRB_time_limit;

            break;
        }
        default:
            LOG(FATAL) << "Unknown matcher type #^#";
        }
    }

    void Matcher::matchStereo(const vector<cv::KeyPoint> &kps_l, const vector<cv::KeyPoint> &kps_r,
                              const vector<cv::Mat> &desc_l, const vector<cv::Mat> &desc_r,
                              vector<int> &match)
    {
        const size_t Nl = kps_l.size(), Nr = kps_r.size();
        vector<double> desc_dist(Nl * Nr), desc_score(Nl * Nr), x_diff(Nl * Nr), y_diff(Nl * Nr);

        // Compute descriptor score and difference in y
        for (size_t i = 0; i < Nl; i++)
        {
            for (size_t j = 0; j < Nr; j++)
            {
                desc_dist[i * Nr + j] = cv::norm(desc_l[i], desc_r[j], norm_type);

                desc_score[i * Nr + j] = stereo_desc_dist_th - desc_dist[i * Nr + j];

                x_diff[i * Nr + j] = kps_l[i].pt.x - kps_r[j].pt.x;

                y_diff[i * Nr + j] = kps_l[i].pt.y - kps_r[j].pt.y;
            }
        }

        /*******************************************************************/
        /****** Solve the stereo matching by global nearest neighbour ******/
        /*******************************************************************/

        try
        {
            GRBEnv env;
            GRBModel model(env);
            model.set(GRB_StringAttr_ModelName, "stereo_matching");
            model.set(GRB_IntParam_LogToConsole, GRB_log_to_console);

            vector<GRBVar> match_mat(Nl * Nr);
            for (size_t i = 0; i < Nl; i++)
            {
                for (size_t j = 0; j < Nr; j++)
                {
                    if (desc_score[i * Nr + j] > 0)
                        match_mat[i * Nr + j] = model.addVar(
                            0.0, 1.0, desc_score[i * Nr + j], GRB_BINARY, "match_" + to_string(i) + to_string(j));
                }
            }

            for (size_t i = 0; i < Nl; i++)
            {
                GRBLinExpr row_sum = 0;
                for (size_t j = 0; j < Nr; j++)
                    if (desc_score[i * Nr + j] > 0)
                        row_sum += match_mat[i * Nr + j];
                model.addConstr(row_sum <= 1, "row_sum_" + to_string(i));
            }

            for (size_t j = 0; j < Nr; j++)
            {
                GRBLinExpr col_sum = 0;
                for (size_t i = 0; i < Nl; i++)
                    if (desc_score[i * Nr + j] > 0)
                        col_sum += match_mat[i * Nr + j];
                model.addConstr(col_sum <= 1, "col_sum_" + to_string(j));
            }

            for (size_t i = 0; i < Nl; i++)
            {
                for (size_t j = 0; j < Nr; j++)
                {
                    if (desc_score[i * Nr + j] <= 0)
                        continue;

                    model.addConstr(x_diff[i * Nr + j] * match_mat[i * Nr + j] <= fx, "x_diff_upper_" + to_string(i) + to_string(j));
                    model.addConstr(0 <= x_diff[i * Nr + j] * match_mat[i * Nr + j], "x_diff_lower_" + to_string(i) + to_string(j));
                    model.addConstr(y_diff[i * Nr + j] * match_mat[i * Nr + j] <= 2, "y_diff_upper_" + to_string(i) + to_string(j));
                    model.addConstr(-2 <= y_diff[i * Nr + j] * match_mat[i * Nr + j], "y_diff_lower_" + to_string(i) + to_string(j));
                }
            }

            model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
            model.optimize();

            // Parse solution
            for (size_t i = 0; i < Nl; i++)
            {
                for (size_t j = 0; j < Nr; j++)
                {
                    if (desc_score[i * Nr + j] <= 0)
                        continue;

                    if (match_mat[i * Nr + j].get(GRB_DoubleAttr_X)) // kps_l[i] <---match---> kps_r[j]
                    {
                        match[i] = j;
                        break;
                    }
                }
            }
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

    void Matcher::matchByDesc(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2,
                              const vector<cv::Mat> &desc1, const vector<cv::Mat> &desc2,
                              vector<int> &match)
    {
        switch (matcher_type)
        {
        case MatcherType::CV_NN:
        {
            matchByDescCV(kps1, kps2, desc1, desc2, match);
            break;
        }
        case MatcherType::GNN:
        {
            matchByDescGNN(kps1, kps2, desc1, desc2, match);
            break;
        }
        default:
            LOG(FATAL) << "Unknown matcher type #^#";
        }
    }

    void Matcher::matchByDescCV(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2,
                                const vector<cv::Mat> &desc1, const vector<cv::Mat> &desc2,
                                vector<int> &match)
    {
        CHECK_EQ(desc1[0].cols, desc2[0].cols) << "desc1.cols != desc2.cols in matchByDescCV #^#";

        vector<vector<cv::DMatch>> matches;
        size_t desc1_mat_rows = desc1.size();
        size_t desc2_mat_rows = desc2.size();
        size_t desc_mat_cols = desc1[0].cols;
        int desc_ele_type = desc1[0].type();

        // Convert descriptor vector to descriptor matrix
        cv::Mat desc1_mat(desc1_mat_rows, desc_mat_cols, desc_ele_type);
        cv::Mat desc2_mat(desc2_mat_rows, desc_mat_cols, desc_ele_type);
        for (size_t i = 0; i < desc1_mat_rows; i++)
            desc1[i].row(0).copyTo(desc1_mat.row(i));
        for (size_t i = 0; i < desc2_mat_rows; i++)
            desc2[i].row(0).copyTo(desc2_mat.row(i));

        matcher_cv->knnMatch(desc1_mat, desc2_mat, matches, 2u);

        if (enable_ratio_test)
        {
            size_t n_screened_out_by_ratio_test = 0;
            for (size_t i = 0; i < matches.size(); i++)
            {
                const vector<cv::DMatch> &match_ = matches[i];
                if (match_.size() < 2)
                    continue;

                // Lowe's ratio test
                if (match_[0].distance < lowe_ratio * match_[1].distance)
                    match[match_[0].queryIdx] = match_[0].trainIdx;
                else
                    n_screened_out_by_ratio_test++;
            }
            LOG(INFO) << n_screened_out_by_ratio_test << " matches screened out by ratio test.";
        }
        else
            for (size_t i = 0; i < matches.size(); i++)
            {
                const vector<cv::DMatch> &match_ = matches[i];
                match[match_[0].queryIdx] = match_[0].trainIdx;
            }
    }

    void Matcher::matchByDescGNN(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2,
                                 const vector<cv::Mat> &desc1, const vector<cv::Mat> &desc2,
                                 vector<int> &match)
    {
        const size_t N1 = kps1.size(), N2 = kps2.size();
        vector<double> desc_dist(N1 * N2), desc_score(N1 * N2), consistency;

        // Compute descriptor score
        double min_desc = 23333, max_desc = 0, avg_desc = 0;
        size_t idx = 0;
        for (size_t i = 0; i < N1; i++)
        {
            for (size_t j = 0; j < N2; j++)
            {
                desc_dist[i * N2 + j] = cv::norm(desc1[i], desc2[j], norm_type);

                if (desc_dist[i * N2 + j] < min_desc)
                    min_desc = desc_dist[i * N2 + j];
                if (desc_dist[i * N2 + j] > max_desc)
                    max_desc = desc_dist[i * N2 + j];
                avg_desc = (idx * avg_desc + desc_dist[i * N2 + j]) / (idx + 1);
                idx++;

                desc_score[i * N2 + j] = weight_desc_vs_consis *
                                         1.0 / desc_dist_th *
                                         (desc_dist_th - desc_dist[i * N2 + j]);
            }
        }
        // Descriptor distance distribution
        LOG(INFO) << "DDD: " << min_desc << " " << avg_desc << " " << max_desc;

        // Compute consistency
        if (enable_consistency)
        {
            for (size_t a = 0; a < N1; a++)
            {
                for (size_t c = 0; c < N2; c++)
                {
                    if (desc_score[a * N2 + c] <= 0)
                        continue;

                    const cv::Point &A = kps1[a].pt;
                    const cv::Point &C = kps2[c].pt;

                    for (size_t b = 0; b < N1; b++)
                    {
                        if (b == a)
                            continue;

                        for (size_t d = 0; d < N2; d++)
                        {
                            if (desc_score[b * N2 + d] <= 0)
                                continue;

                            if (d == c)
                                continue;

                            const cv::Point &B = kps1[b].pt;
                            const cv::Point &D = kps2[d].pt;

                            cv::Point AB = B - A, CD = D - C;

                            double diff_len = abs(norm(AB) - norm(CD)) / (norm(AB) + norm(CD));
                            double diff_cos = AB.dot(CD) / (norm(AB) * norm(CD));

                            // Robust acos function
                            double diff_ang;
                            if (diff_cos >= 1.0)
                                diff_ang = 0.0;
                            else if (diff_cos <= -1.0)
                                diff_ang = 180.0;
                            else
                                diff_ang = acos(diff_cos) * 180.0 / CV_PI;

                            consistency.push_back((1.0 - weight_desc_vs_consis) *
                                                  (weight_len_vs_ang * 1.0 / diff_len_th * (diff_len_th - diff_len) +
                                                   (1.0 - weight_len_vs_ang) * 1.0 / diff_ang_th * (diff_ang_th - diff_ang)));

                            CHECK(!isnan(consistency.back()) && !isinf(consistency.back())) << "Consistency = Nan or Inf #^#";
                        }
                    }
                }
            }
        }

        /**********************************************************************************/
        /****** Solve the descriptor matching by consistent global nearest neighbour ******/
        /**********************************************************************************/

        try
        {
            GRBEnv env;
            GRBModel model_init(env);
            model_init.set(GRB_StringAttr_ModelName, "desc_matching_init");
            model_init.set(GRB_IntParam_LogToConsole, GRB_log_to_console);

            vector<GRBVar> match_init(N1 * N2);
            for (size_t i = 0; i < N1; i++)
            {
                for (size_t j = 0; j < N2; j++)
                {
                    if (desc_score[i * N2 + j] > 0)
                        match_init[i * N2 + j] = model_init.addVar(
                            0.0, 1.0, desc_score[i * N2 + j], GRB_BINARY, "match_" + to_string(i) + to_string(j));
                }
            }

            for (size_t i = 0; i < N1; i++)
            {
                GRBLinExpr row_sum = 0;
                for (size_t j = 0; j < N2; j++)
                    if (desc_score[i * N2 + j] > 0)
                        row_sum += match_init[i * N2 + j];
                model_init.addConstr(row_sum <= 1, "row_sum_" + to_string(i));
            }

            for (size_t j = 0; j < N2; j++)
            {
                GRBLinExpr col_sum = 0;
                for (size_t i = 0; i < N1; i++)
                    if (desc_score[i * N2 + j] > 0)
                        col_sum += match_init[i * N2 + j];
                model_init.addConstr(col_sum <= 1, "col_sum_" + to_string(j));
            }

            model_init.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
            model_init.optimize();

            if (enable_consistency)
            {
                vector<GRBVar> match_(N1 * N2);
                GRBModel model(env);
                model.set(GRB_StringAttr_ModelName, "desc_matching");
                model.set(GRB_IntParam_LogToConsole, GRB_log_to_console);

                GRBQuadExpr score = 0;
                for (size_t i = 0; i < N1; i++)
                {
                    for (size_t j = 0; j < N2; j++)
                    {
                        if (desc_score[i * N2 + j] <= 0)
                            continue;

                        match_[i * N2 + j] = model.addVar(
                            0.0, 1.0, 0.0, GRB_BINARY, "match_" + to_string(i) + to_string(j));

                        score += desc_score[i * N2 + j] * match_[i * N2 + j];
                    }
                }

                size_t idx = 0;
                for (size_t a = 0; a < N1; a++)
                {
                    for (size_t c = 0; c < N2; c++)
                    {
                        if (desc_score[a * N2 + c] <= 0)
                            continue;

                        for (size_t b = 0; b < N1; b++)
                        {
                            if (b == a)
                                continue;

                            for (size_t d = 0; d < N2; d++)
                            {
                                if (desc_score[b * N2 + d] <= 0)
                                    continue;

                                if (d == c)
                                    continue;

                                score += consistency[idx++] *
                                         match_[a * N2 + c] * match_[b * N2 + d];
                            }
                        }
                    }
                }
                model.setObjective(score, GRB_MAXIMIZE);

                for (size_t i = 0; i < N1; i++)
                {
                    GRBLinExpr row_sum = 0;
                    for (size_t j = 0; j < N2; j++)
                        if (desc_score[i * N2 + j] > 0)
                            row_sum += match_[i * N2 + j];
                    model.addConstr(row_sum <= 1, "row_sum_" + to_string(i));
                }

                for (size_t j = 0; j < N2; j++)
                {
                    GRBLinExpr col_sum = 0;
                    for (size_t i = 0; i < N1; i++)
                        if (desc_score[i * N2 + j] > 0)
                            col_sum += match_[i * N2 + j];
                    model.addConstr(col_sum <= 1, "col_sum_" + to_string(j));
                }

                for (size_t i = 0; i < N1; i++)
                {
                    for (size_t j = 0; j < N2; j++)
                    {
                        if (desc_score[i * N2 + j] > 0)
                            match_[i * N2 + j].set(GRB_DoubleAttr_Start, match_init[i * N2 + j].get(GRB_DoubleAttr_X));
                    }
                }

                // The MIPFocus parameter allows you to modify your high-level solution strategy, depending
                // on your goals. By default, the Gurobi MIP solver strikes a balance between finding new feasible
                // solutions and proving that the current solution is optimal. If you are more interested in finding
                // feasible solutions quickly, you can select MIPFocus=1. If you believe the solver is having no trouble
                // finding good quality solutions, and wish to focus more attention on proving optimality, select
                // MIPFocus=2. If the best objective bound is moving very slowly (or not at all), you may want to
                // try MIPFocus=3 to focus on the bound.
                model.set(GRB_IntParam_MIPFocus, GRB_mip_focus);

                // The root relaxation in a MIP model can sometimes be quite expensive to solve. If you find that a lot
                // of time is spent here, consider using the Method parameter to select a different continuous algorithm
                // for the root. For example, Method=2 would select the parallel barrier algorithm at the root, and
                // Method=3 would select the concurrent solver. Note that you can choose a different algorithm for
                // the MIP node relaxations using the NodeMethod parameter, but it is rarely beneficial to change
                // this from the default (dual simplex).
                model.set(GRB_IntParam_Method, -1);

                // Determines the amount of time spent in MIP heuristics. You can think of the value as the
                // desired fraction of total MIP runtime devoted to heuristics (so by default, we aim to spend 5% of
                // runtime on heuristics). Larger values produce more and better feasible solutions, at a cost of slower
                // progress in the best bound.
                model.set(GRB_DoubleParam_Heuristics, GRB_heuristics);

                // Controls the presolve level. A value of -1 corresponds to an automatic setting. Other options
                // are off (0), conservative (1), or aggressive (2). More aggressive application of presolve takes more
                // time, but can sometimes lead to a significantly tighter model.
                model.set(GRB_IntParam_Presolve, GRB_presolve);

                // Controls the presolve sparsify reduction. This reduction can sometimes significantly reduce the
                // number of non-zero values in the presolved model. Value 0 shuts off the reduction, while value 1
                // forces it on for mixed integer programming (MIP) models.
                model.set(GRB_IntParam_PreSparsify, GRB_presparsify);

                // Sets the strategy for handling non-convex quadratic objectives or non-convex quadratic con-
                // straints. With setting 0, an error is reported if the original user model contains non-convex
                // quadratic constructs, except for Q matrix linearization controlled by the PreQLinearize param-
                // eter. With setting 1, an error is reported if non-convex quadratic constructs could not be discarded
                // or linearized during presolve. With setting 2, non-convex quadratic problems are solved by means
                // of translating them into bilinear form and applying spatial branching. The default -1 setting is
                // currently equivalent to 1, and may change in future releases to be equivalent to 2.
                model.set(GRB_IntParam_NonConvex, 2);

                model.set(GRB_DoubleParam_TimeLimit, GRB_time_limit);

                model.optimize();

                // Parse solution of model
                for (size_t i = 0; i < N1; i++)
                {
                    for (size_t j = 0; j < N2; j++)
                    {
                        if (desc_score[i * N2 + j] <= 0)
                            continue;

                        if (match_[i * N2 + j].get(GRB_DoubleAttr_X)) // kps1[i] <---match---> kps2[j]
                        {
                            match[i] = j;
                            break;
                        }
                    }
                }
            }
            else
            {
                // Parse solution of model_init
                for (size_t i = 0; i < N1; i++)
                {
                    for (size_t j = 0; j < N2; j++)
                    {
                        if (desc_score[i * N2 + j] <= 0)
                            continue;

                        if (match_init[i * N2 + j].get(GRB_DoubleAttr_X)) // kps1[i] <---match---> kps2[j]
                        {
                            match[i] = j;
                            break;
                        }
                    }
                }
            }
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
}