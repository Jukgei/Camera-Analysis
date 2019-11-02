#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>
#include <deque>
#include <string>
#include "OpticalFlow.hpp"
using namespace cv;

void MycalcHist(std::string s, Mat images, MatND &dst){
    if(s == "Y"){
        int histBinNum = 255;
        float range[] = {0, 255};
        const float * histRange = { range };
        Mat buf;
        cvtColor(images, buf, CV_BGR2YCrCb);
        std::vector<Mat> channals;
        split(buf, channals);
        Mat Y = channals.at(0);
        calcHist(&Y,
                 1,0,Mat(),
                 dst,
                 1, &histBinNum,
                 &histRange, true, true);
    }
    else{
        int histBinNum = 255;
        float range[] = {0, 255};
        const float * histRange = { range };
        Mat buf;
        cvtColor(images, buf, CV_BGR2GRAY);
        //std::vector<Mat> channals;
        //split(buf, channals);
        //Mat Y = channals.at(0);
        calcHist(&buf,
                 1,0,Mat(),
                 dst,
                 1, &histBinNum,
                 &histRange, true, true);
    }
}

void DrawHist(const MatND &src, Mat &dst, int hpt, double MaxValue){
    for(int i = 0; i < 256; i++){
        float BinValue = src.at<float>(i);
        int RealValue = saturate_cast<int>(256 *(1 - BinValue/MaxValue) ); //( BinValue * hpt/ MaxValue  );
        //line(dst,Point(i, 255), Point(i, 256 - RealValue), Scalar(255));
        rectangle(dst, Point(i,256-1),Point((i+1)-1,RealValue),Scalar(255));
    }
}

int main(int argc, char** argv)
{
    if(argc < 2){
        std::cout<<"Please the video path."<<std::endl;
        return 0;
    }
    VideoCapture Cap(argv[1]);
    if(!Cap.isOpened()){
        std::cout<<"Open file false"<<std::endl;
        return 0;
    }
    std::string s = argv[2];
    std::deque<Mat> FrameRGB;
    while(true){
        Mat buf;
        Cap>> buf;
        if(buf.empty())
            break;
        else
            FrameRGB.push_back(buf);
    }
    bool broadcast = false;
    
    //Point3f Xp(0.5, 0.6 ,2.1);

    //Mat X{Mat_<Point3f>(Xp)};
    //
    //Mat nX;
    //X.convertTo(nX,CV_64F);
    //Mat nnX = nX.reshape(1,3).clone();
    //std::cout<<"Norm is"<<norm(nnX,NORM_L2)<<std::endl;
    
    namedWindow("Video",1); 
    //namedWindow("Hist",1);//1 means auto windows size
    //namedWindow("Gray",1);
    //namedWindow("Y", 1);
    namedWindow("Threshold",1);
    //namedWindow("SEG",1);
    std::deque<Mat>::iterator iter = FrameRGB.begin();
    auxiliary::OpticalFlow myOpti(true);
    do{
       imshow("Video", *iter);  
       //MatND histogram;
       //MycalcHist(s, *iter,histogram);
       //Mat histogramImage(256,256,CV_8U,Scalar(0));
       //int hpt = saturate_cast<int>(0.9 * 256);
       //double MaxValue = 0;
       //double MinValue = 0; 
       //minMaxLoc(histogram,&MinValue, &MaxValue, 0, 0);
       //DrawHist(histogram, histogramImage, hpt, MaxValue);
       //imshow("Hist", histogramImage);
       char key = waitKey(30);
       //Mat Gray;
       //cvtColor(*iter,Gray, CV_BGR2GRAY);
       //imshow("Gray",Gray);

       //Y channals;
       //Mat Y;
       //Mat YCrCb;
       //std::vector<Mat> channals;
       //cvtColor(*iter, YCrCb, CV_BGR2YCrCb);
       //Scalar YCrCbMean = mean(YCrCb);
       //split(YCrCb ,channals);
       //double LumenMean = YCrCbMean[0];
       //Y = channals[0];
       //CvFont font; 
       //cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,1.0,1.0,0,2,8); 
       //std::string Slumen = std::to_string(LumenMean);
       //char *c = new char[Slumen.size()];
       //strcpy(c, Slumen.data());
       //IplImage  tmp = IplImage(Y);
       //CvArr * arr = (CvArr* )&tmp;
       //cvPutText(arr, c,  cvPoint(10,10), &font, CV_RGB(0,255,0));
       //imshow("Y", Y); 
       
       //Mat bw;
       ////adaptiveThreshold(Gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 599, 0);
       //threshold(Y, bw, 0, 255, CV_THRESH_OTSU); //very good!
       //imshow("Threshold", bw);



       
       //Segamatation
       //Mat seg;
       //pyrMeanShiftFiltering(*iter, seg, 15, 10, 1, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5, 1));
       //imshow("SEG",seg);
       //front
       if(key == 'l' && broadcast == false){ 
           iter++;
           if(iter == FrameRGB.end()){
               iter = FrameRGB.begin();
               std::cout<<"Start"<<std::endl;
           }
           myOpti.GetImage(*iter);
           if(myOpti.ReturnTrackPointsSize() > 0)
               myOpti.OpticalTracking();
           if(myOpti.ReturnisFindFeature())
               myOpti.FindFeaturePoints();
           myOpti.Update();
       }

       //back
       else if(key == 'h' && broadcast == false){
           iter--;
           if(iter == FrameRGB.begin()-1){
               iter = FrameRGB.end()-1;
               std::cout<<"End"<<std::endl;
           }
           myOpti.GetImage(*iter);
           if(myOpti.ReturnTrackPointsSize() > 0)
               myOpti.OpticalTracking();
           if(myOpti.ReturnisFindFeature())
               myOpti.FindFeaturePoints();
           myOpti.Update();
           
       }
       else if(key == ' ')
           broadcast = !broadcast;
       else if(key == 'q')
           break;
       else{

       }
       if(broadcast){
           iter++;
           myOpti.GetImage(*iter);
           if(myOpti.ReturnTrackPointsSize() > 0)
               myOpti.OpticalTracking();
           if(myOpti.ReturnisFindFeature())
               myOpti.FindFeaturePoints();
           myOpti.Update();
       }
        
    }while(true);
    return 0;
}

