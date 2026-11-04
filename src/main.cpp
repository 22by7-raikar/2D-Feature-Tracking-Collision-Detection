// Benchmark keypoint detectors and descriptors on a KITTI sequence and estimate
// camera-only time-to-collision (TTC) with the preceding vehicle.
//
// Per frame: load -> detect -> keep keypoints in vehicle ROI -> describe
//            -> match against previous frame -> estimate TTC -> log.

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "features.hpp"
#include "ttc.hpp"

namespace fs = std::filesystem;

namespace
{

struct Config
{
    fs::path dataDir = fs::path(PROJECT_ROOT) / "images/KITTI/2011_09_26/image_00/data";
    fs::path outDir  = fs::path(PROJECT_ROOT) / "results";
    int numFrames = 10;
    double frameDt = 0.1;                            // KITTI camera runs at 10 Hz
    cv::Rect vehicleRoi{535, 180, 180, 150};         // preceding vehicle in these frames
    std::string matcherType = "MAT_BF";
    std::string selectorType = "SEL_KNN";
    bool saveImages = false;
    std::vector<std::string> detectors = kDetectorTypes;
    std::vector<std::string> descriptors = kDescriptorTypes;
};

struct Frame
{
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

double msSince(int64 ticks)
{
    return 1000.0 * (cv::getTickCount() - ticks) / cv::getTickFrequency();
}

std::string upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}

cv::Mat loadGray(const fs::path &dir, int index)
{
    std::ostringstream name;
    name << std::setfill('0') << std::setw(10) << index << ".png";
    const fs::path path = dir / name.str();
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (img.empty())
        throw std::runtime_error("could not read " + path.string());
    return img;
}

/// Run one detector/descriptor pair over the sequence and append one CSV row per frame.
void runCombination(const Config &cfg, const std::string &det, const std::string &desc,
                    std::ostream &csv)
{
    std::deque<Frame> buffer; // only the previous and current frame are needed
    std::vector<double> ttcs;
    size_t totalMatches = 0;

    for (int i = 0; i < cfg.numFrames; ++i)
    {
        Frame frame;
        frame.image = loadGray(cfg.dataDir, i);

        int64 t = cv::getTickCount();
        frame.keypoints = detectKeypoints(frame.image, det);
        const double detectMs = msSince(t);

        auto &kps = frame.keypoints;
        kps.erase(std::remove_if(kps.begin(), kps.end(),
                                 [&](const cv::KeyPoint &kp) { return !cfg.vehicleRoi.contains(kp.pt); }),
                  kps.end());

        t = cv::getTickCount();
        frame.descriptors = describeKeypoints(frame.image, frame.keypoints, desc);
        const double describeMs = msSince(t);

        double meanSize = 0.0;
        for (const auto &kp : frame.keypoints)
            meanSize += kp.size;
        if (!frame.keypoints.empty())
            meanSize /= frame.keypoints.size();

        buffer.push_back(std::move(frame));
        if (buffer.size() > 2)
            buffer.pop_front();

        csv << i << ',' << det << ',' << desc << ',' << buffer.back().keypoints.size() << ','
            << std::fixed << std::setprecision(2) << meanSize << ',' << detectMs << ',' << describeMs << ',';

        if (buffer.size() < 2)
        {
            csv << ",\n"; // no previous frame: no matches or TTC
            continue;
        }

        const Frame &prev = buffer.front();
        const Frame &curr = buffer.back();
        const auto matches = matchDescriptors(prev.descriptors, curr.descriptors, desc,
                                              cfg.matcherType, cfg.selectorType);
        const double ttc = computeTTCCamera(prev.keypoints, curr.keypoints, matches, cfg.frameDt);

        totalMatches += matches.size();
        if (std::isfinite(ttc))
            ttcs.push_back(ttc);
        csv << matches.size() << ',';
        if (std::isfinite(ttc))
            csv << ttc;
        csv << '\n';

        if (cfg.saveImages)
        {
            cv::Mat vis;
            cv::drawMatches(prev.image, prev.keypoints, curr.image, curr.keypoints, matches, vis);
            const fs::path dir = cfg.outDir / "matches";
            fs::create_directories(dir);
            cv::imwrite((dir / (det + "_" + desc + "_" + std::to_string(i - 1) + "_" +
                                std::to_string(i) + ".png")).string(), vis);
        }
    }

    std::cout << std::left << std::setw(18) << (det + "/" + desc)
              << " mean matches " << std::fixed << std::setprecision(1) << std::setw(7)
              << totalMatches / double(cfg.numFrames - 1);
    if (!ttcs.empty())
    {
        std::sort(ttcs.begin(), ttcs.end());
        std::cout << " median TTC " << ttcs[ttcs.size() / 2] << " s";
    }
    std::cout << '\n';
}

void usage()
{
    std::cerr << "Usage: 2D_feature_tracking [--detector D] [--descriptor D]\n"
                 "                           [--matcher MAT_BF|MAT_FLANN] [--selector SEL_NN|SEL_KNN]\n"
                 "                           [--data-dir DIR] [--out-dir DIR] [--save]\n"
                 "  Detectors:   SHITOMASI HARRIS FAST BRISK ORB AKAZE SIFT (default: all)\n"
                 "  Descriptors: BRISK ORB AKAZE SIFT BRIEF* FREAK* (default: all available)\n"
                 "  * needs OpenCV built with opencv_contrib (xfeatures2d)\n";
}

} // namespace

int main(int argc, char *argv[])
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if      (arg == "--detector"   && hasValue) cfg.detectors   = {upper(argv[++i])};
        else if (arg == "--descriptor" && hasValue) cfg.descriptors = {upper(argv[++i])};
        else if (arg == "--matcher"    && hasValue) cfg.matcherType  = upper(argv[++i]);
        else if (arg == "--selector"   && hasValue) cfg.selectorType = upper(argv[++i]);
        else if (arg == "--data-dir"   && hasValue) cfg.dataDir = argv[++i];
        else if (arg == "--out-dir"    && hasValue) cfg.outDir  = argv[++i];
        else if (arg == "--save")                   cfg.saveImages = true;
        else { usage(); return arg == "--help" || arg == "-h" ? 0 : 1; }
    }

    for (const auto &[names, known] : {std::pair{cfg.detectors, kDetectorTypes},
                                       std::pair{cfg.descriptors, kDescriptorTypes}})
    {
        for (const auto &n : names)
        {
            if (std::find(known.begin(), known.end(), n) == known.end())
            {
                std::cerr << "unknown or unavailable type '" << n << "'\n";
                usage();
                return 1;
            }
        }
    }

    fs::create_directories(cfg.outDir);
    const fs::path csvPath = cfg.outDir / "results.csv";
    std::ofstream csv(csvPath);
    if (!csv)
    {
        std::cerr << "could not write " << csvPath << '\n';
        return 1;
    }
    csv << "Frame,Detector,Descriptor,Keypoints,MeanKeypointSize,DetectMs,DescribeMs,Matches,TTC\n";

    for (const auto &det : cfg.detectors)
    {
        for (const auto &desc : cfg.descriptors)
        {
            if (isUnsupportedCombination(det, desc))
                continue;
            try
            {
                runCombination(cfg, det, desc, csv);
            }
            catch (const std::exception &e)
            {
                // One failing pair (e.g. BRIEF without xfeatures2d) should not stop the benchmark.
                std::cerr << "skipped " << det << "/" << desc << ": " << e.what() << '\n';
            }
        }
    }

    std::cout << "Results written to " << csvPath << '\n';
    return 0;
}
