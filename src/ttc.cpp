#include "ttc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

double computeTTCCamera(const std::vector<cv::KeyPoint> &kptsPrev,
                        const std::vector<cv::KeyPoint> &kptsCurr,
                        const std::vector<cv::DMatch> &matches,
                        double dt, double minDist)
{
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> ratios;

    for (size_t i = 0; i < matches.size(); ++i)
    {
        const cv::Point2f &prevA = kptsPrev[matches[i].queryIdx].pt;
        const cv::Point2f &currA = kptsCurr[matches[i].trainIdx].pt;

        for (size_t j = i + 1; j < matches.size(); ++j)
        {
            const double distCurr = cv::norm(currA - kptsCurr[matches[j].trainIdx].pt);
            const double distPrev = cv::norm(prevA - kptsPrev[matches[j].queryIdx].pt);
            if (distCurr >= minDist && distPrev > eps)
                ratios.push_back(distCurr / distPrev);
        }
    }

    if (ratios.empty())
        return std::numeric_limits<double>::quiet_NaN();

    auto mid = ratios.begin() + ratios.size() / 2;
    std::nth_element(ratios.begin(), mid, ratios.end());
    const double medianRatio = *mid;

    if (std::abs(1.0 - medianRatio) < eps)
        return std::numeric_limits<double>::quiet_NaN();
    return -dt / (1.0 - medianRatio);
}
