#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "matching2D.hpp"

using namespace std;

// ---------------------------------------------------------------------------
// Helper: determine descriptor norm category
// ---------------------------------------------------------------------------
bool isBinaryDescriptor(const string &descriptorType)
{
    // SIFT uses floating-point descriptors (L2); everything else is binary (Hamming).
    return descriptorType != "SIFT";
}

// ---------------------------------------------------------------------------
// 5. Match descriptors
// ---------------------------------------------------------------------------
void matchDescriptors(vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches,
                      const string &descriptorType, const string &matcherType,
                      const string &selectorType)
{
    const bool binary  = isBinaryDescriptor(descriptorType);
    const int  normType = binary ? cv::NORM_HAMMING : cv::NORM_L2;

    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (matcherType == "MAT_BF")
    {
        matcher = cv::BFMatcher::create(normType, /*crossCheck=*/false);
    }
    else if (matcherType == "MAT_FLANN")
    {
        if (binary)
        {
            // LSH index is required for binary (Hamming-distance) descriptors.
            // Using the default KD-Tree with binary descriptors would produce incorrect results.
            matcher = cv::makePtr<cv::FlannBasedMatcher>(
                cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        }
        else
        {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }
    }
    else
    {
        throw invalid_argument("matchDescriptors: unknown matcherType '" + matcherType + "'");
    }

    // Perform matching.
    if (selectorType == "SEL_NN")
    {
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType == "SEL_KNN")
    {
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // Lowe's ratio test: discard ambiguous matches.
        const float ratio_thresh = 0.8f;
        for (const auto &m : knn_matches)
        {
            if (m[0].distance < ratio_thresh * m[1].distance)
                matches.push_back(m[0]);
        }
    }
    else
    {
        throw invalid_argument("matchDescriptors: unknown selectorType '" + selectorType + "'");
    }
}

// ---------------------------------------------------------------------------
// 4. Compute descriptors
// ---------------------------------------------------------------------------
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, const string &descriptorType)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType == "BRISK")
    {
        // Binary Robust Invariant Scalable Keypoints
        extractor = cv::BRISK::create(/*threshold=*/30, /*octaves=*/3, /*patternScale=*/1.0f);
    }
    else if (descriptorType == "ORB")
    {
        // Oriented FAST + Rotated BRIEF -- parameters are shared with the ORB detector.
        extractor = cv::ORB::create(
            /*nfeatures=*/500, /*scaleFactor=*/1.2f, /*nlevels=*/8,
            /*edgeThreshold=*/31, /*firstLevel=*/0, /*WTA_K=*/2,
            cv::ORB::HARRIS_SCORE, /*patchSize=*/31, /*fastThreshold=*/20);
    }
    else if (descriptorType == "AKAZE")
    {
        // AKAZE descriptor -- must be paired with the AKAZE detector.
        extractor = cv::AKAZE::create(
            cv::AKAZE::DESCRIPTOR_MLDB, /*size=*/0, /*channels=*/3,
            /*threshold=*/0.001f, /*nOctaves=*/4, /*nOctaveLayers=*/4,
            cv::KAZE::DIFF_PM_G2);
    }
    else if (descriptorType == "SIFT")
    {
#if HAS_XFEATURES2D
        extractor = cv::xfeatures2d::SIFT::create();
#else
        throw runtime_error("descKeypoints: SIFT requires opencv-contrib (xfeatures2d).");
#endif
    }
    else if (descriptorType == "BRIEF")
    {
#if HAS_XFEATURES2D
        // Not rotation-invariant by default; fast and compact (32-byte).
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(/*bytes=*/32);
#else
        throw runtime_error("descKeypoints: BRIEF requires opencv-contrib (xfeatures2d).");
#endif
    }
    else if (descriptorType == "FREAK")
    {
#if HAS_XFEATURES2D
        extractor = cv::xfeatures2d::FREAK::create();
#else
        throw runtime_error("descKeypoints: FREAK requires opencv-contrib (xfeatures2d).");
#endif
    }
    else
    {
        throw invalid_argument("descKeypoints: unknown descriptorType '" + descriptorType + "'");
    }

    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t << " ms" << endl;
}

// ---------------------------------------------------------------------------
// 1. Unified keypoint detector
//    Handles: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
// ---------------------------------------------------------------------------
void detKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                  const string &detectorType, bool bVis)
{
    double t = (double)cv::getTickCount();

    if (detectorType == "SHITOMASI")
    {
        const int   blockSize    = 4;
        const double maxOverlap  = 0.0;
        const double minDistance = (1.0 - maxOverlap) * blockSize;
        const int   maxCorners   = img.rows * img.cols / max(1.0, minDistance);

        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(img, corners, maxCorners,
                                /*qualityLevel=*/0.01, minDistance,
                                cv::Mat(), blockSize, /*useHarris=*/false, /*k=*/0.04);
        for (const auto &c : corners)
        {
            cv::KeyPoint kp;
            kp.pt   = c;
            kp.size = blockSize;
            keypoints.push_back(kp);
        }
    }
    else if (detectorType == "HARRIS")
    {
        const int    blockSize   = 2;
        const int    apertureSize = 3;
        const int    minResponse = 100;
        const double k           = 0.04;
        const double maxOverlap  = 0.0;

        cv::Mat harrisRes, harrisNorm;
        cv::cornerHarris(img, harrisRes, blockSize, apertureSize, k);
        cv::normalize(harrisRes, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_32F);

        for (int j = 0; j < harrisNorm.rows; ++j)
        {
            for (int i = 0; i < harrisNorm.cols; ++i)
            {
                int response = (int)harrisNorm.at<float>(j, i);
                if (response <= minResponse) continue;

                cv::KeyPoint kp(cv::Point2f((float)i, (float)j),
                                (float)(2 * apertureSize), -1, response);
                bool bOverlap = false;
                for (auto &existing : keypoints)
                {
                    if (cv::KeyPoint::overlap(kp, existing) > maxOverlap)
                    {
                        bOverlap = true;
                        if (kp.response > existing.response)
                            existing = kp;
                        break;
                    }
                }
                if (!bOverlap)
                    keypoints.push_back(kp);
            }
        }
    }
    else
    {
        // Modern OpenCV detector selected by name.
        cv::Ptr<cv::FeatureDetector> detector;
        if (detectorType == "FAST")
        {
            // Features from Accelerated Segment Test.
            detector = cv::FastFeatureDetector::create(
                /*threshold=*/30, /*NMS=*/true, cv::FastFeatureDetector::TYPE_9_16);
        }
        else if (detectorType == "BRISK")
        {
            // Multi-scale FAST with scale and rotation invariance.
            detector = cv::BRISK::create(/*threshold=*/30, /*octaves=*/3, /*patternScale=*/1.0f);
        }
        else if (detectorType == "ORB")
        {
            // oFAST keypoints + rBRIEF descriptors.
            detector = cv::ORB::create(
                /*nfeatures=*/500, /*scaleFactor=*/1.2f, /*nlevels=*/8,
                /*edgeThreshold=*/31, /*firstLevel=*/0, /*WTA_K=*/2,
                cv::ORB::HARRIS_SCORE, /*patchSize=*/31, /*fastThreshold=*/20);
        }
        else if (detectorType == "AKAZE")
        {
            detector = cv::AKAZE::create(
                cv::AKAZE::DESCRIPTOR_MLDB, /*size=*/0, /*channels=*/3,
                /*threshold=*/0.001f, /*nOctaves=*/4, /*nOctaveLayers=*/4,
                cv::KAZE::DIFF_PM_G2);
        }
        else if (detectorType == "SIFT")
        {
#if HAS_XFEATURES2D
            detector = cv::xfeatures2d::SIFT::create();
#else
            throw runtime_error("detKeypoints: SIFT requires opencv-contrib (xfeatures2d).");
#endif
        }
        else
        {
            throw invalid_argument("detKeypoints: unknown detectorType '" + detectorType + "'");
        }
        detector->detect(img, keypoints);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size()
         << " keypoints in " << 1000 * t << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        const string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}