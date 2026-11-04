#ifndef TTC_HPP
#define TTC_HPP

#include <vector>

#include <opencv2/core.hpp>

/// Estimate time-to-collision (seconds) from the change in scale of matched keypoints.
///
/// For every pair of matches, the ratio of their pixel distance in the current frame to
/// that in the previous frame measures how much the object grew. With a constant-velocity
/// model, TTC = -dt / (1 - medianRatio). The median makes the estimate robust to outlier
/// matches.
///
/// @param kptsPrev, kptsCurr  keypoints of the previous and current frame
/// @param matches            matches with queryIdx -> kptsPrev and trainIdx -> kptsCurr
/// @param dt                 time between the two frames in seconds
/// @param minDist            ignore pairs closer than this in the current frame (pixels),
///                           since short distances amplify keypoint noise
/// @return TTC in seconds, or NaN if it cannot be estimated (too few pairs, or no scale change).
double computeTTCCamera(const std::vector<cv::KeyPoint> &kptsPrev,
                        const std::vector<cv::KeyPoint> &kptsCurr,
                        const std::vector<cv::DMatch> &matches,
                        double dt, double minDist = 50.0);

#endif // TTC_HPP
