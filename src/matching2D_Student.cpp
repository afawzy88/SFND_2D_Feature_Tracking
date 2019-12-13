#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the Harric corners detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
    // visualize keypoints
    string windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 2);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

// Detect keypoints in image using the FAST detector
void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::FAST(img, keypoints, 30, true, cv::FastFeatureDetector::DetectorType::TYPE_9_16);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST Detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    if (bVis)
    {
    // visualize keypoints
    string windowName = "FAST Deteection Results";
    cv::namedWindow(windowName, 3);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

// Detect keypoints in image using the BRISK detector
void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
    // visualize keypoints
    string windowName = "BRISK Detection Results";
    cv::namedWindow(windowName, 4);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

// Detect keypoints in image using the ORB detector
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(10000);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
    // visualize keypoints
    string windowName = "ORB Detection Results";
    cv::namedWindow(windowName, 4);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

// Detect keypoints in image using the AKAZE detector
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
    // visualize keypoints
    string windowName = "AKAZE Detection Results";
    cv::namedWindow(windowName, 4);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

// Detect keypoints in image using the FREAK detector
void detKeypointsFREAK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::FREAK::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FREAK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
    // visualize keypoints
    string windowName = "FREAK Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

// Detect keypoints in image using the SIFT detector
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
    // visualize keypoints
    string windowName = "SIFT Detection Results";
    cv::namedWindow(windowName, 6);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    cv::waitKey(0);
    }
}

//The placeholder for all detection functions
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    if (detectorType.compare("SHITOMASI") == 0)
    {
        detKeypointsShiTomasi(keypoints, img, bVis);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        detKeypointsHarris(keypoints, img, bVis);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        detKeypointsFAST(keypoints, img, bVis);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detKeypointsBRISK(keypoints, img, bVis);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detKeypointsORB(keypoints, img, bVis);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detKeypointsAKAZE(keypoints, img, bVis);
    }
    else if (detectorType.compare("FREAK") == 0)
    {
        detKeypointsFREAK(keypoints, img, bVis);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detKeypointsSIFT(keypoints, img, bVis);
    }
}
