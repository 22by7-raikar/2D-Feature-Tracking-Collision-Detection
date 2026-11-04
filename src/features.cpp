#include "features.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>
#if HAS_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

namespace
{

/// Create an OpenCV feature detector/extractor by name, or nullptr if unknown.
/// The same parameters are used whether BRISK / ORB / AKAZE / SIFT act as detector or descriptor.
cv::Ptr<cv::Feature2D> makeFeature2D(const std::string &type)
{
    if (type == "FAST")  return cv::FastFeatureDetector::create(/*thresh=*/30, /*nms=*/true);
    if (type == "BRISK") return cv::BRISK::create(/*thresh=*/30, /*octaves=*/3);
    if (type == "ORB")   return cv::ORB::create(/*nfeatures=*/500);
    if (type == "AKAZE") return cv::AKAZE::create();
    if (type == "SIFT")  return cv::SIFT::create();
#if HAS_XFEATURES2D
    if (type == "BRIEF") return cv::xfeatures2d::BriefDescriptorExtractor::create();
    if (type == "FREAK") return cv::xfeatures2d::FREAK::create();
#endif
    return nullptr;
}

/// Construct each detector/extractor once and reuse it, so per-frame timings measure the
/// work itself rather than setup (BRISK, for example, precomputes its sampling pattern).
/// `role` keeps detector and extractor instances separate.
cv::Ptr<cv::Feature2D> cachedFeature2D(const std::string &role, const std::string &type)
{
    static std::map<std::string, cv::Ptr<cv::Feature2D>> cache;
    const std::string key = role + ":" + type;
    auto it = cache.find(key);
    if (it == cache.end())
        it = cache.emplace(key, makeFeature2D(type)).first;
    return it->second;
}

std::vector<cv::KeyPoint> detectShiTomasi(const cv::Mat &img)
{
    const int blockSize = 4;
    const double minDistance = blockSize;
    const int maxCorners = static_cast<int>(img.rows * img.cols / minDistance);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, /*qualityLevel=*/0.01, minDistance,
                            cv::noArray(), blockSize, /*useHarris=*/false);

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(corners.size());
    for (const auto &c : corners)
        keypoints.emplace_back(c, static_cast<float>(blockSize));
    return keypoints;
}

std::vector<cv::KeyPoint> detectHarris(const cv::Mat &img)
{
    const int blockSize = 2, apertureSize = 3, minResponse = 100;
    const double k = 0.04;

    cv::Mat response, responseNorm;
    cv::cornerHarris(img, response, blockSize, apertureSize, k);
    cv::normalize(response, responseNorm, 0, 255, cv::NORM_MINMAX, CV_32F);

    // Non-maximum suppression: among overlapping candidates keep the strongest.
    std::vector<cv::KeyPoint> keypoints;
    for (int y = 0; y < responseNorm.rows; ++y)
    {
        for (int x = 0; x < responseNorm.cols; ++x)
        {
            const float r = responseNorm.at<float>(y, x);
            if (r <= minResponse)
                continue;

            cv::KeyPoint candidate(cv::Point2f(x, y), 2.f * apertureSize, -1, r);
            auto overlapping = std::find_if(keypoints.begin(), keypoints.end(),
                [&](const cv::KeyPoint &kp) { return cv::KeyPoint::overlap(candidate, kp) > 0.f; });

            if (overlapping == keypoints.end())
                keypoints.push_back(candidate);
            else if (candidate.response > overlapping->response)
                *overlapping = candidate;
        }
    }
    return keypoints;
}

} // namespace

bool isUnsupportedCombination(const std::string &detectorType, const std::string &descriptorType)
{
    // AKAZE descriptors need AKAZE keypoints (they rely on its scale-space octave info).
    if (descriptorType == "AKAZE" && detectorType != "AKAZE")
        return true;
    // ORB descriptors fail on SIFT keypoints (OpenCV tries a huge allocation and throws).
    if (descriptorType == "ORB" && detectorType == "SIFT")
        return true;
    return false;
}

std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &img, const std::string &detectorType)
{
    if (detectorType == "SHITOMASI") return detectShiTomasi(img);
    if (detectorType == "HARRIS")    return detectHarris(img);

    auto detector = detectorType == "BRIEF" || detectorType == "FREAK"
                        ? nullptr // descriptor-only algorithms
                        : cachedFeature2D("detector", detectorType);
    if (!detector)
        throw std::invalid_argument("unknown detector type '" + detectorType + "'");

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);
    return keypoints;
}

cv::Mat describeKeypoints(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints,
                          const std::string &descriptorType)
{
    auto extractor = descriptorType == "FAST" ? nullptr // detector-only algorithm
                                              : cachedFeature2D("extractor", descriptorType);
    if (!extractor)
    {
        if (descriptorType == "BRIEF" || descriptorType == "FREAK")
            throw std::runtime_error(descriptorType + " requires OpenCV built with opencv_contrib (xfeatures2d)");
        throw std::invalid_argument("unknown descriptor type '" + descriptorType + "'");
    }

    cv::Mat descriptors;
    extractor->compute(img, keypoints, descriptors);
    return descriptors;
}

std::vector<cv::DMatch> matchDescriptors(const cv::Mat &descPrev, const cv::Mat &descCurr,
                                         const std::string &descriptorType,
                                         const std::string &matcherType,
                                         const std::string &selectorType)
{
    std::vector<cv::DMatch> matches;
    if (descPrev.empty() || descCurr.empty())
        return matches;

    // SIFT produces float descriptors (L2 distance); all others here are binary (Hamming).
    const bool binary = descriptorType != "SIFT";

    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (matcherType == "MAT_BF")
        matcher = cv::BFMatcher::create(binary ? cv::NORM_HAMMING : cv::NORM_L2);
    else if (matcherType == "MAT_FLANN")
        matcher = binary ? cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))
                         : cv::FlannBasedMatcher::create();
    else
        throw std::invalid_argument("unknown matcher type '" + matcherType + "'");

    if (selectorType == "SEL_NN")
    {
        matcher->match(descPrev, descCurr, matches);
    }
    else if (selectorType == "SEL_KNN")
    {
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(descPrev, descCurr, knn, 2);

        // Lowe's ratio test: keep a match only if it is clearly better than the runner-up.
        const float ratio = 0.8f;
        for (const auto &m : knn)
            if (m.size() >= 2 && m[0].distance < ratio * m[1].distance)
                matches.push_back(m[0]);
    }
    else
    {
        throw std::invalid_argument("unknown selector type '" + selectorType + "'");
    }
    return matches;
}
