#ifndef matching2D_hpp
#define matching2D_hpp

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

// Try to include xfeatures2d for SIFT, BRIEF, FREAK support
// Available in OpenCV >= 4.3 or with opencv-contrib module
#ifdef __has_include
  #if __has_include(<opencv2/xfeatures2d.hpp>)
    #include <opencv2/xfeatures2d.hpp>
    #define HAS_XFEATURES2D 1
  #elif __has_include(<opencv2/xfeatures2d/nonfree.hpp>)
    #include <opencv2/xfeatures2d/nonfree.hpp>
    #define HAS_XFEATURES2D 1
  #else
    #define HAS_XFEATURES2D 0
  #endif
#else
  #define HAS_XFEATURES2D 0
#endif

#include "dataStructures.h"

// Returns true when the descriptor encodes binary patterns (Hamming norm).
// Returns false for float-valued descriptors (L2 norm, e.g. SIFT).
bool isBinaryDescriptor(const std::string &descriptorType);

// Single entry point for all detectors: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT.
// Throws std::invalid_argument on unknown detectorType,
// std::runtime_error  if a contrib-only detector is missing.
void detKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                  const std::string &detectorType, bool bVis = false);

// Compute descriptors for the given keypoints.
// Throws std::invalid_argument on unknown descriptorType,
// std::runtime_error  if a contrib-only descriptor is missing.
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, const std::string &descriptorType);

// Match descriptors between two frames.
// Throws std::invalid_argument on unknown matcherType / selectorType.
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                      const std::string &descriptorType,
                      const std::string &matcherType,
                      const std::string &selectorType);

#endif /* matching2D_hpp */
