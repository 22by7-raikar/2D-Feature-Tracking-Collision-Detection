/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>        // #8: O(1) pop_front for ring buffer
#include <algorithm>    // #9: remove_if  #16: transform
#include <cctype>       // #16: toupper
#include <stdexcept>    // #7: runtime_error
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

// ---------------------------------------------------------------------------
// Named constant for the preceding-vehicle ROI
// ---------------------------------------------------------------------------
static const cv::Rect kVehicleROI(535, 180, 180, 150);

// ---------------------------------------------------------------------------
// Normalise a detector/descriptor string to UPPERCASE in-place.
// ---------------------------------------------------------------------------
static string toUpperCase(string s)
{
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c){ return (char)toupper(c); });
    return s;
}

// ---------------------------------------------------------------------------
// Load a single image as grayscale; throws std::runtime_error on failure.
// ---------------------------------------------------------------------------
static cv::Mat loadGrayscaleImage(const string &path)
{
    cv::Mat img = cv::imread(path);
    if (img.empty())
        throw runtime_error("loadGrayscaleImage: could not open '" + path + "'");
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// ---------------------------------------------------------------------------
// Detect keypoints and restrict them to the vehicle ROI.
// ---------------------------------------------------------------------------
static void detectAndFilterKeypoints(cv::Mat &img,
                                     const string &detectorType,
                                     vector<cv::KeyPoint> &keypoints,
                                     bool bFocusOnVehicle)
{
    detKeypoints(keypoints, img, detectorType, /*bVis=*/false);

    if (bFocusOnVehicle)
    {
        // Erase-remove idiom -- O(n) instead of the O(n^2) manual-erase loop.
        keypoints.erase(
            remove_if(keypoints.begin(), keypoints.end(),
                      [](const cv::KeyPoint &kp){ return !kVehicleROI.contains(kp.pt); }),
            keypoints.end());
    }
}

// ---------------------------------------------------------------------------
// Log per-frame keypoint statistics (uses '\n', not std::endl).
// ---------------------------------------------------------------------------
static void logKeypointStats(ofstream &log, size_t imgIndex,
                             const string &detectorType,
                             const vector<cv::KeyPoint> &keypoints)
{
    float minSz = numeric_limits<float>::max(), maxSz = 0.f, meanSz = 0.f;
    for (const auto &kp : keypoints)
    {
        minSz  = min(minSz,  kp.size);
        maxSz  = max(maxSz,  kp.size);
        meanSz += kp.size;
    }
    if (!keypoints.empty())  meanSz /= (float)keypoints.size();
    else                     minSz  = 0.f;

    log << imgIndex << "," << detectorType << "," << keypoints.size()
        << "," << minSz << "," << maxSz << "," << meanSz << "\n"; // #11

    cout << "Image " << imgIndex << " - " << detectorType << ": "
         << keypoints.size() << " keypoints"
         << "  (Min: " << minSz << "  Max: " << maxSz << "  Mean: " << meanSz << ")\n";
}

// ---------------------------------------------------------------------------
// Full pipeline for one detector + descriptor combination.
// ---------------------------------------------------------------------------
static void runCombination(const string &detectorType,
                           const string &descriptorType,
                           const string &matcherType,
                           const string &selectorType,
                           const string &imgBasePath,
                           const string &imgPrefix,
                           const string &imgFileType,
                           int imgStartIndex,
                           int imgEndIndex,
                           int imgFillWidth,
                           int dataBufferSize,
                           bool bFocusOnVehicle,
                           bool bSaveImages,   // # Save images with keypoints drawn
                           ofstream &keypointLog,
                           ofstream &matchLog)
{
    deque<DataFrame> dataBuffer; // Deque gives O(1) pop_front

    for (size_t imgIndex = 0;
         imgIndex <= (size_t)(imgEndIndex - imgStartIndex);
         ++imgIndex)
    {
        /* --- 1. Load image --- */
        ostringstream num;
        num << setfill('0') << setw(imgFillWidth) << imgStartIndex + (int)imgIndex;
        const string imgPath = imgBasePath + imgPrefix + num.str() + imgFileType;

        cv::Mat imgGray = loadGrayscaleImage(imgPath); // Load a single image as grayscale; throws std::runtime_error on failure

        /* --- 2. Ring buffer (O(1) pop_front) --- */  // Deque gives O(1) pop_front
        DataFrame frame;
        frame.cameraImg = imgGray;
        if ((int)dataBuffer.size() == dataBufferSize)
            dataBuffer.pop_front();
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* --- 3. Detect & filter keypoints --- */
        vector<cv::KeyPoint> keypoints;
        detectAndFilterKeypoints(dataBuffer.back().cameraImg,
                                 detectorType, keypoints, bFocusOnVehicle);
        logKeypointStats(keypointLog, imgIndex, detectorType, keypoints);
        dataBuffer.back().keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* --- 4. Extract descriptors --- */
        cv::Mat descriptors;
        descKeypoints(dataBuffer.back().keypoints,
                      dataBuffer.back().cameraImg,
                      descriptors, descriptorType);
        dataBuffer.back().descriptors = descriptors;
        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        /* --- 5. Match (requires >= 2 frames) --- */
        if ((int)dataBuffer.size() > 1)
        {
            vector<cv::DMatch> matches;
            matchDescriptors(dataBuffer[dataBuffer.size() - 2].keypoints,
                             dataBuffer.back().keypoints,
                             dataBuffer[dataBuffer.size() - 2].descriptors,
                             dataBuffer.back().descriptors,
                             matches, descriptorType, matcherType, selectorType);

            dataBuffer.back().kptMatches = matches;

            matchLog << imgIndex << "," << detectorType << ","    // #11
                     << descriptorType << "," << matches.size() << "\n";
            cout << "Image " << imgIndex << " - " << detectorType << "/"
                 << descriptorType << ": " << matches.size() << " matches\n";
            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            /* --- 6. Optionally save visualisation --- */
            if (bSaveImages)
            {
                cv::Mat matchImg;
                cv::drawMatches(dataBuffer[dataBuffer.size() - 2].cameraImg,
                                dataBuffer[dataBuffer.size() - 2].keypoints,
                                dataBuffer.back().cameraImg,
                                dataBuffer.back().keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(),
                                cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                ostringstream ss;
                ss << "../images/outputs/match_" << detectorType << "_"
                   << descriptorType << "_frames_"
                   << (imgIndex - 1) << "_" << imgIndex << ".png";
                cv::imwrite(ss.str(), matchImg);
            }
        }
    } // eof image loop
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    /* --- Defaults (overridable via CLI) --- */
    string singleDetector;    // empty -> test all detectors
    string singleDescriptor;  // empty -> test all descriptors
    string matcherType  = "MAT_BF";
    string selectorType = "SEL_KNN";
    bool   bSaveImages  = false; // off by default -- avoids 300+ output files

    /* --- CLI argument parsing --- */
    //   Usage: ./2D_feature_tracking [--detector D] [--descriptor D]
    //                                [--matcher M] [--selector S] [--save]
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if      (arg == "--detector"   && i + 1 < argc) singleDetector   = toUpperCase(argv[++i]);
        else if (arg == "--descriptor" && i + 1 < argc) singleDescriptor = toUpperCase(argv[++i]);
        else if (arg == "--matcher"    && i + 1 < argc) matcherType      = toUpperCase(argv[++i]);
        else if (arg == "--selector"   && i + 1 < argc) selectorType     = toUpperCase(argv[++i]);
        else if (arg == "--save")                        bSaveImages      = true;
        else { cerr << "Unknown argument: " << arg
                    << "\nUsage: ./2D_feature_tracking [--detector D] [--descriptor D]"
                       " [--matcher M] [--selector S] [--save]\n"; return 1; }
    }

    /* --- Image source configuration --- */
    const string dataPath    = "../";
    const string imgBasePath = dataPath + "images/";
    const string imgPrefix   = "KITTI/2011_09_26/image_00/data/000000";
    const string imgFileType = ".png";
    const int imgStartIndex  = 0;
    const int imgEndIndex    = 9;    // 10 images total
    const int imgFillWidth   = 4;
    const int dataBufferSize = 2;
    const bool bFocusOnVehicle = true;

    /* --- Determine which combinations to run --- */
    vector<string> detectorTypes   = {"SHITOMASI","HARRIS","FAST","BRISK","ORB","AKAZE","SIFT"};
    vector<string> descriptorTypes = {"BRISK","ORB","AKAZE","SIFT","BRIEF","FREAK"};
    if (!singleDetector.empty())   detectorTypes   = {singleDetector};
    if (!singleDescriptor.empty()) descriptorTypes = {singleDescriptor};

    /* --- Open log files --- */
    ofstream keypointLog("../keypoint_log.csv");
    ofstream matchLog("../match_log.csv");
    keypointLog << "ImageIndex,DetectorType,NumKeypoints,MinSize,MaxSize,MeanSize\n"; // #11
    matchLog    << "ImageIndex,DetectorType,DescriptorType,NumMatches\n";

    /* --- Main loop --- */
    for (const string &det : detectorTypes)
    {
        for (const string &desc : descriptorTypes)
        {
            // AKAZE descriptors only work with the AKAZE detector.
            if (desc == "AKAZE" && det != "AKAZE") continue;

            cout << "\n========================================\n"
                 << "Testing: " << det << " + " << desc << "\n"
                 << "========================================" << endl;

            try
            {
                runCombination(det, desc, matcherType, selectorType,
                               imgBasePath, imgPrefix, imgFileType,
                               imgStartIndex, imgEndIndex, imgFillWidth,
                               dataBufferSize, bFocusOnVehicle, bSaveImages,
                               keypointLog, matchLog);
            }
            catch (const exception &e)
            {
                // #7: errors in one combination don't abort the whole benchmark.
                cerr << "[ERROR] " << det << "+" << desc << ": " << e.what() << "\n";
            }
        }
    }

    keypointLog.close();
    matchLog.close();

    cout << "\n=== Analysis Complete ===\n"
         << "Keypoint log : ../keypoint_log.csv\n"
         << "Match log    : ../match_log.csv\n";
    if (bSaveImages)
        cout << "Match images : ../images/outputs/\n";

    return 0;
}

