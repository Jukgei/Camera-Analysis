#include "OpticalFlow.hpp"

auxiliary::OpticalFlow::OpticalFlow(bool Display){
   //this->Cap = VideoCapture a(0);
    //VideoCapture buff(0);
    //this->Cap = buff;
    this->Display = Display;
    FrameId = 0;
    DisplayName = "OpticalFlow";
    //if(!Cap.isOpened()){
    //    std::cout<<"Cannot open camera."<<std::endl;
    //    return ;
    //}
    //fps = Cap.get(CAP_PROP_FPS); //get fps
    //std::cout<<"FPS:"<<fps<<std::endl;
    //if(fps <= 0 )
    //    fps = 25;
    //vw.open("opticalflowvideo/opticalflow.avi",
    //        CV_FOURCC('M','J','P','G'),
    //        fps,
    //        Size((int)Cap.get(CAP_PROP_FRAME_WIDTH)/2,
    //             (int)Cap.get(CAP_PROP_FRAME_HEIGHT)/2)
    //        );

    //vg.open("opticalflowvideo/opticalflowRGB.avi",
    //        CV_FOURCC('M','J','P','G'),
    //        fps,
    //        Size((int)Cap.get(CAP_PROP_FRAME_WIDTH)/2,
    //             (int)Cap.get(CAP_PROP_FRAME_HEIGHT)/2)
    //       );

    //if(!vw.isOpened()){
    //    std::cout<<"Video write error!"<<std::endl;
    //}
    //if(!vg.isOpened()){
    //    std::cout<<"RGB write error!"<<std::endl;
    //}

    //Parameter
    MaxCorners = 40;
    QualityLevel = 0.1;
    MinDistance = 8.0;
    BlockSize = 3;
    k = 0.04;
    UseHarris = true;

    FrameId = 0;
    isFindFeature = true;
    DetectInterval = 1;
    
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS,10,0.03);
    TrackLen = 5;

    Displacement.x = 0;
    Displacement.y = 0;

    isNiceThreshold1 = 1;
    
    PatchSize = 10;
}

auxiliary::OpticalFlow::~OpticalFlow(){
    //Cap.release();
}

bool auxiliary::OpticalFlow::GetImage(Mat &buf){
    Mat FrameYCrCb;
    std::vector<Mat> channels;
    buf.copyTo(FrameRGB);
    //resize(buf, FrameRGB, Size(buf.cols/2,buf.rows/2),0,0,INTER_LINEAR);
    Width = FrameRGB.cols;
    Height = FrameRGB.rows;
    cvtColor(FrameRGB,FrameGray,CV_BGR2GRAY);
    threshold(FrameGray, FrameBin, 0, 255, CV_THRESH_OTSU);
    cvtColor(FrameRGB,FrameYCrCb,CV_BGR2YCrCb);
    Scalar YCrCbMean = mean(FrameYCrCb);
    split(FrameYCrCb, channels);
    Y = channels.at(0);     //cv_8UC1
    //std::cout<<"Y type is:" <<Y.type()<<std::endl;
    LumenMean = YCrCbMean[0];
    //std::cout<<"LumenMean is:" <<LumenMean << std::endl;
    //vg.write(FrameRGB);
    if(Display)
        FrameRGB.copyTo(Visualization);
    return true;
}


std::string auxiliary::OpticalFlow::ReturnDisplayName(){
    return this->DisplayName;
}

Point2f auxiliary::OpticalFlow::OpticalTracking(){
    Mat img0 = FrameGrayPrev;
    Mat img1 = FrameGray;
    std::vector<Point2f> p0;
    std::vector<Point2f> p1;
    std::vector<Point2f> p0r;
    std::vector<uchar>  status;
    std::vector<float> err;
    std::vector<Point2f> d;
    std::vector<bool> isNice;
    std::vector<std::vector<Point2f>> NewTrackPoints;
    Size winSize(15,15);
    //TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS,10,0.03);

    for(uint32_t i = 0; i < TrackPoints.size(); i++){
        std::vector<Point2f> temp;
        temp = TrackPoints[i];
        //printf("(%f,%f)",temp[temp.size()-1].x,temp[temp.size()-1].y);
        //std::cout<<"temp"<<temp[temp.size()-1].x<<','<<temp[temp.size()-1].y<<' '<<std::endl;
        p0.push_back(temp[temp.size()-1]);
    }
    
    calcOpticalFlowPyrLK(img0,img1,
                         p0,p1,status,
                         err, winSize,
                         2,termcrit);

    calcOpticalFlowPyrLK(img1,img0,
                         p1,p0r,status,
                         err,winSize,
                         2,termcrit);
    PointVectorErr(p0,p0r,d); 
    
    uint32_t num = 0;
    Point2f SumErr(0.0,0.0);
    
    JudgmentPoint(d,isNice);
    std::vector<Point2f> Prev;
    std::vector<Point2f> Now;
    int recall = 0;
    for(uint32_t i = 0; i < p1.size() && num < 100; i++){
        if( !isNice[i] ){
            recall++;
            continue;
        }

        std::vector<Point2f> temp = TrackPoints[i];
        Prev.push_back(temp[temp.size()-1]);
        Now.push_back(p1[i]);
        SumErr = (p1[i]  - temp[temp.size()-1] ) + SumErr; 
        num++;
        temp.push_back(p1[i]);
        if(temp.size() > TrackLen){
            std::vector<Point2f>::iterator k = temp.begin();
            temp.erase(k); //Delet the first element
        }
        NewTrackPoints.push_back(temp);
        //if(Display)
        //    circle(Visualization,p1[i],2, Scalar(0,255,0), -1);
    }
    std::cout<<"TrackPoints Prev num is"<<TrackPoints.size()<<std::endl;
    std::cout<<"Recall failure num is"<<recall<<std::endl;
    Mat mask;
    Mat resMatrix;
    Displacement.x = SumErr.x/num;
    Displacement.y = SumErr.y/num; 
    printf("MeanErr:(%f, %f)\n", Displacement.x, Displacement.y);
    double Threshold1 = sqrt(Displacement.x * Displacement.x +
                            Displacement.y * Displacement.y) * 0.5; 
    Threshold1 = Threshold1 > 1 ? Threshold1 : 1; 
    isNiceThreshold1 = Threshold1;
    std::vector<bool> out;
    if(Prev.size() > 0 && Now.size() > 0){
        resMatrix = findHomography(Prev, Now, RANSAC, Threshold1, mask, 300, 0.995);
        std::cout<<"Prev point size is:" << Prev.size()<< std::endl;
        //std::cout<< "Mask cols:"<< mask.cols<<"Mask rows:"<<mask.rows<<std::endl;  //cols =1, rows = Prev.size()
        mask = mask.reshape(1,1).clone();
        
        for(uint8_t i = 0; i < mask.cols; i++){
            if(mask.at<uchar>(0,i) != 0)
                out.push_back(true);
            else
                out.push_back(false);
        }
        for(uint8_t i = 0; i < out.size(); i++){
            std::cout<<out[i]<<' ';
        }
        
        std::cout<<'\n';
        //std::cout<<mask<<std::endl;
        //std::cout<<"mask TYPE"<<mask.type()<<std::endl;
    }
    
    
    //std::cout<< "RANSACErr("<<resMatrix.at<double>(1,3)<<','<< resMatrix.at<double>(2,3)<<')' << std::endl;
    //printf("RANSACErr:(%lf, %lf)\n", resMatrix.at<double>(0,2), resMatrix.at<double>(1,2));
    std::cout<<"resMatrix="<< resMatrix<<std::endl;
    //std::cout<<"resMatrixType Type is"<< resMatrix.type()<<std::endl;  //64FC1

    
    //if(std::isnan(Displacement.x) || std::isnan(Displacement.y))
    //    printf("num = %d\n SumErr(%f,%f)",num,SumErr.x,SumErr.y);
    //std::cout<<"("<<Displacement.x<<','<<Displacement.y<<')'<<std::endl;
    //Now Publish 
    TrackPoints = NewTrackPoints;
    //std::cout<<"TrackPoints Num :"<<TrackPoints.size()<<std::endl;
    std::cout<<"Track Num is"<<TrackPoints.size()<<std::endl;
    if(Display){
        std::vector<Point2i> DrawPointSet;
        for(uint32_t i = 0 ; i < TrackPoints.size(); i++){
            std::vector<Point2f> temp = TrackPoints[i];
            for(uint32_t i = 0; i < temp.size(); i++){
                //std::cout<<'('<<IntPoint.x<<","<<IntPoint.y<<')'<<' '<<std::endl;
                DrawPointSet.push_back(Point2i(temp[i]));
            }
            if(out[i]){
                circle(Visualization,DrawPointSet[DrawPointSet.size()-1],2, Scalar(0,255,0), -1);
                polylines(Visualization,DrawPointSet,false,Scalar(0,255,0));
                DrawPointSet.clear();
            }
            else{
                circle(Visualization,DrawPointSet[DrawPointSet.size()-1],2, Scalar(0,0,255), -1);
                polylines(Visualization,DrawPointSet,false,Scalar(0,0,255));
                DrawPointSet.clear();
            }
            //check the point good or poor
            //if(temp.size() >= 2 && resMatrix.cols == 3 && resMatrix.rows == 3){
            //    Point3f Xp(temp[temp.size()-2].x, temp[temp.size()-2].y,1);

            //    Point3f Yp(temp[temp.size()-1].x, temp[temp.size()-1].y,1);
            //    Mat X{Mat_<Point3f>(Xp)};
            //    Mat Y{Mat_<Point3f>(Yp)};
            //    
            //    Mat nX;
            //    Mat nY;
            //    X.convertTo(nX,CV_64F);
            //    Y.convertTo(nY,CV_64F);
            //    Mat nnX = nX.reshape(1,3).clone();
            //    Mat nnY = nY.reshape(1,3).clone();
            //    if(norm(resMatrix * nnX - nnY,NORM_L2) < Threshold1){
            //        circle(Visualization,DrawPointSet[DrawPointSet.size()-1],2, Scalar(0,255,0), -1);
            //        polylines(Visualization,DrawPointSet,false,Scalar(0,255,0));
            //        DrawPointSet.clear();
            //    }
            //    else{
            //        circle(Visualization,DrawPointSet[DrawPointSet.size()-1],2, Scalar(0,0,255), -1);
            //        polylines(Visualization,DrawPointSet,false,Scalar(0,0,255));
            //        DrawPointSet.clear();
            //    }

            //}
            
        }
        imshow(ReturnDisplayName(),Visualization);
    }
   
//////////////////////////Bad Point
//    
//    p0.clear();
//    p1.clear();
//    p0r.clear();
//    status.clear();
//    err.clear();
//    d.clear();
//    isNice.clear();
//    NewTrackPoints.clear();
//    //Size winSize(15,15);
//    for(uint32_t i = 0; i < TrackPointsShadows.size(); i++){
//        std::vector<Point2f> temp;
//        temp = TrackPointsShadows[i];
//        //printf("(%f,%f)",temp[temp.size()-1].x,temp[temp.size()-1].y);
//        //std::cout<<"temp"<<temp[temp.size()-1].x<<','<<temp[temp.size()-1].y<<' '<<std::endl;
//        p0.push_back(temp[temp.size()-1]);
//    }
//    //for(uint32_t i = 0; i < p0.size(); i++)
//    //    printf("(%f,%f)\n",p0[i].x,p0[i].y);
//    //for(uint32_t i = 0; i < p0.size();i++)
//    //    std::cout<<p0[i]<<std::endl;
//    calcOpticalFlowPyrLK(img0,img1,
//                         p0,p1,status,
//                         err, winSize,
//                         2,termcrit);
//
//    calcOpticalFlowPyrLK(img1,img0,
//                         p1,p0r,status,
//                         err,winSize,
//                         2,termcrit);
//    PointVectorErr(p0,p0r,d); 
//    
//    //for(uint32_t i = 0; i < status.size(); i++) //    printf("%d\n",status[i]);
//    //for(uint32_t i = 0; i < p1.size(); i++)
//    //    printf("(%f,%f)\n",p1[i].x,p1[i].y);
//    //for(uint32_t i = 0; i < d.size(); i++)
//    //    printf("(%f,%f)\n",d[i].x,d[i].y);
//    //for(uint32_t i = 0; i < p0.size(); i++){
//    //    //std::cout<<'('<<p0[i].x<<','<<p0[i].y<<')'<<
//    //    //    ' '<<'('<<p0r[i].x<<','<<p0r[i].y<<')'<<std::endl;
//    //    printf("p0:(%f,%f),p0r:(%f,%f)\n",p0[i].x,p0[i].y,p0r[i].x,p0r[i].y);
//    //}
//
//    //num = 0;
//    //SumErr.x = 0;
//    //SumErr.y = 0;
//    JudgmentPoint(d,isNice);
//    for(uint32_t i = 0; i < p1.size(); i++){
//        if( !isNice[i] )
//            continue;
//
//        std::vector<Point2f> temp = TrackPointsShadows[i];
//        //SumErr = (p1[i]  - temp[temp.size()-1] ) + SumErr; 
//        //num++;
//        temp.push_back(p1[i]);
//        if(temp.size() > TrackLen){
//            std::vector<Point2f>::iterator k = temp.begin();
//            temp.erase(k); //Delet the first element
//        }
//        NewTrackPoints.push_back(temp);
//        if(Display)
//            circle(Visualization,p1[i],2, Scalar(0,0,255), -1);
//    }
//    //Displacement.x = SumErr.x/num;
//    //Displacement.y = SumErr.y/num; 
//    //if(std::isnan(Displacement.x) || std::isnan(Displacement.y))
//    //    printf("num = %d\n SumErr(%f,%f)",num,SumErr.x,SumErr.y);
//    //std::cout<<"("<<Displacement.x<<','<<Displacement.y<<')'<<std::endl;
//    //Now Publish 
//    TrackPointsShadows = NewTrackPoints;
//    //std::cout<<"TrackPoints Num :"<<TrackPoints.size()<<std::endl;
    //if(Display){
    //    std::vector<Point2i> DrawPointSet;
    //    for(uint32_t i = 0 ; i < TrackPointsShadows.size(); i++){
    //        std::vector<Point2f> temp = TrackPointsShadows[i];
    //        for(uint32_t i = 0; i < temp.size(); i++){
    //            //std::cout<<'('<<IntPoint.x<<","<<IntPoint.y<<')'<<' '<<std::endl;
    //            DrawPointSet.push_back(Point2i(temp[i]));
    //        }
    //        polylines(Visualization,DrawPointSet,false,Scalar(0,0,255));
    //        DrawPointSet.clear();
    //    }
    //}
    
    //vw.write(Visualization);
    return Displacement;
}

void auxiliary::OpticalFlow::JudgmentPoint(const std::vector<Point2f> err, std::vector<bool>  &isNice ){
    for(uint32_t i = 0; i < err.size(); i++){
        if(err[i].x < isNiceThreshold1 && err[i].y < isNiceThreshold1)
            isNice.push_back(true);
        else
            isNice.push_back(false);
    }
}

void auxiliary::OpticalFlow::PointVectorErr(const std::vector<Point2f> a,const std::vector<Point2f> b,std::vector<Point2f> &d){
    for(uint32_t i = 0; i < a.size(); i++){
        Point2f temp = a[i] - b[i];
        if(temp.x < 0)
            temp.x = -temp.x;
        if(temp.y < 0)
            temp.y = -temp.y;
        d.push_back(temp);
    }
}

void auxiliary::OpticalFlow::FindFeaturePoints(){
    Mat mask = Mat::zeros(FrameRGB.rows,FrameRGB.cols,CV_8UC1);
    mask = 255; //Bug

    for( uint32_t i = 0; i < TrackPoints.size(); i++   ) {
        std::vector<Point2f> point = TrackPoints[i];
        Point2i temp = point[point.size()-1];
        circle(mask,temp,5,0,-1);
    }

    //for( uint32_t i = 0; i < TrackPointsShadows.size(); i++   ) {
    //    std::vector<Point2f> point = TrackPointsShadows[i];
    //    Point2i temp = point[point.size()-1];
    //    circle(mask,temp,5,0,-1);
    //}

    std::vector<Point2i> p; 
    goodFeaturesToTrack(FrameGray,
                        p,
                        MaxCorners,
                        QualityLevel,
                        MinDistance,
                        mask,
                        BlockSize,
                        UseHarris,
                        k);

    //DetectShadow(p);

    if(!p.empty()){
        //std::vector<Point2f> temp;
        //IntPointToFloat(p,temp);
        for(uint32_t i = 0; i < p.size(); i++ ){
            std::vector<Point2f> buf;
            buf.push_back(Point2f(p[i]));
            TrackPoints.push_back(buf);
            buf.clear();
        }
    }
    
    //if(!Shadow.empty()){
    //    for(uint32_t i = 0; i < Shadow.size(); i++){
    //        std::vector<Point2f> buf;
    //        buf.push_back(Point2f(Shadow[i]));
    //        TrackPointsShadows.push_back(buf);
    //        buf.clear();
    //    }
    //}

}

void auxiliary::OpticalFlow::DetectShadow(const std::vector<Point2i> featurepoint){
    Shadow.clear();
    Normal.clear();
    
    for(uint32_t i = 0; i < featurepoint.size(); i++){
        if(featurepoint[i].x - PatchSize < 0         || 
           featurepoint[i].x + PatchSize > Width - 1 ||
           featurepoint[i].y - PatchSize < 0         ||
           featurepoint[i].y + PatchSize > Height + 1)
            Shadow.push_back(featurepoint[i]); //or throw directly
        else{
            int StartRow = featurepoint[i].y - PatchSize;
            int EndRow = featurepoint[i].y + PatchSize;
            int StartCol = featurepoint[i].x - PatchSize;
            int EndCol = featurepoint[i].x + PatchSize;
            uint32_t sum = 0;
            for(int j = StartRow; j <= EndRow; j++){
                uchar * pdata = FrameBin.ptr<uchar>(j);
                uchar * pdat  = Y.ptr<uchar>(j);
                for( int k = StartCol; k <= EndCol; k++ )
                    sum += *pdata++;
            }
            if(sum > 0.75 * 255.0 * 2 * PatchSize * 2 * PatchSize )
                Normal.push_back(featurepoint[i]);
            else{
                if(Y.at<uchar>(featurepoint[i]) > LumenMean * 0.45 &&
                   Y.at<uchar>(featurepoint[i]) < LumenMean * 1.2)
                    Shadow.push_back(featurepoint[i]);
                else
                    Normal.push_back(featurepoint[i]);
                
            }
        }   
        //if(Y.at<uchar>(featurepoint[i]) < LumenMean * 1)
        //    Shadow.push_back(featurepoint[i]);
        //else
        //    Normal.push_back(featurepoint[i]);
    }
}

void auxiliary::OpticalFlow::Update(){
    FrameId = (FrameId + 1) % DetectInterval;
    if(FrameId == 0)
        isFindFeature = true;
    else
        isFindFeature = false;
    
    FrameGray.copyTo(FrameGrayPrev);
    
    
} 

int auxiliary::OpticalFlow::ReturnTrackPointsSize(){
    //return TrackPoints.size()<TrackPointsShadows.size()?TrackPoints.size():TrackPointsShadows.size() ;
    return TrackPoints.size();
}

bool auxiliary::OpticalFlow::ReturnisFindFeature(){
    return isFindFeature;
}

bool auxiliary::OpticalFlow::ReturnDisplay(){
    return Display;
}

//void auxiliary::OpticalFlow::IntPointToFloat( const std::vector<Point2i> i, std::vector<Point2f> &f ){
//    for(uint32_t j = 0; j < i.size(); j++ ){
//        f.push_back(Point2f(i[j]));
//    }
//}
//
//
