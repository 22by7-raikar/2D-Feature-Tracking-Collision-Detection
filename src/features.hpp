#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

/// Detector names accepted by detectKeypoints().
inline const std::vector<std::string> kDetectorTypes =
    {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};

/// Descriptor names accepted by describeKeypoints().
/// BRIEF and FREAK are only available when OpenCV was built with xfeatures2d.
inline const std::vector<std::string> kDescriptorTypes = {"BRISK", "ORB", "AKAZE", "SIFT",
#if HAS_XFEATURES2D
                                                          "BRIEF", "FREAK"
#endif
};

/// True if this detector/descriptor pair is known not to work in OpenCV.
bool isUnsupportedCombination(const std::string &detectorType,
                              const std::string &descriptorType);

/// Detect keypoints in a grayscale image.
/// @throws std::invalid_argument for an unknown detector type.
std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &img, const std::string &detectorType);

/// Compute one descriptor row per keypoint. The extractor may drop keypoints
/// it cannot describe (e.g. near the image border), so `keypoints` is updated.
/// @throws std::invalid_argument for an unknown descriptor type,
///         std::runtime_error if the descriptor needs xfeatures2d and it is missing.
cv::Mat describeKeypoints(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints,
                          const std::string &descriptorType);

/// Match descriptors of the previous frame (query) against the current frame (train).
/// @param matcherType  "MAT_BF" (brute force) or "MAT_FLANN"
/// @param selectorType "SEL_NN" (best match) or "SEL_KNN" (k=2 with Lowe's ratio test, 0.8)
/// @throws std::invalid_argument for an unknown matcher or selector type.
std::vector<cv::DMatch> matchDescriptors(const cv::Mat &descPrev, const cv::Mat &descCurr,
                                         const std::string &descriptorType,
                                         const std::string &matcherType,
                                         const std::string &selectorType);

#endif // FEATURES_HPP
