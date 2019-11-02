#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;

namespace auxiliary{

class OpticalFlow{
public:
    OpticalFlow(bool Display);
    bool GetImage(Mat &buf);
    std::string ReturnDisplayName();
    void FindFeaturePoints();
    Point2f OpticalTracking();
    void Update();
    void PointVectorErr(const std::vector<Point2f> a,const std::vector<Point2f> b,std::vector<Point2f> &d); 
    void JudgmentPoint(const std::vector<Point2f> err, std::vector<bool>  &isNice);
    int ReturnTrackPointsSize();
    bool ReturnisFindFeature();
    bool ReturnDisplay();
    void DetectShadow(const std::vector<Point2i> featurepoint);
    //void IntPointToFloat(const std::vector<Point2i> i, std::vector<Point2f> &f);
    ~OpticalFlow();


private:
    bool Display;
    Mat FrameRGB;
    Mat FrameGray;
    Mat Y;
    Mat FrameGrayPrev;
    Mat Visualization; 
    Mat FrameBin;

    double LumenMean;

    //VideoCapture Cap; 
    std::string DisplayName;
    std::vector<std::vector<Point2f>> TrackPoints;
    std::vector<std::vector<Point2f>> TrackPointsShadows;
    //Some variable about video write
    int fps;
    int Width;
    int Height;
    //VideoWriter vw;
    //VideoWriter vg;

    uint8_t FrameId;
    bool isFindFeature;
    uint8_t DetectInterval;

    //Parameters
    int MaxCorners;
    double QualityLevel;
    double MinDistance;
    int BlockSize;
    double k;
    bool UseHarris;
    
    TermCriteria termcrit;

    uint8_t TrackLen;
    
    Point2f Displacement;

    std::vector<Point2i> Shadow;
    std::vector<Point2i> Normal;

    int PatchSize; //used to detection shadow, Radius

    double isNiceThreshold1;
};

}




#endif


