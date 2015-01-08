#include <pcl/io/pcd_io.h>
#include <ctime>
#include "gnuplot-iostream.h"
#include <pcl_ros/filters/filter.h>
#include <limits>
#include <fstream>
#include <sstream>


//global variables declared in ni_vision.cpp
extern sensor_msgs::PointCloud2ConstPtr cloud_;
extern boost::mutex m;
extern cv::Mat cvm_image_camera;



//modification of function SelRecognition_1 in func_recognition.hpp
//takes the shift between the point cloud and the rgb image as an additional argument (bestShift_x, bestShift_y)
void SelRecognition_Shift (int nCandID, int nImgScale, int nDsWidth, std::vector<int> vnProtoPtsCnt, cv::Mat cvm_rgb_org, cv::Mat &cvm_cand_tmp,
                           std::vector<std::vector<int> > mnProtoPtsIdx, std::vector<int> &vnIdxTmp, int bestShift_x, int bestShift_y)
{
    int pts_cnt = 0;
    int xx, yy;
    if (nImgScale > 1) {
        int idx;
        for (int j = 0; j < vnProtoPtsCnt[nCandID]; j++) {
            GetPixelPos(mnProtoPtsIdx[nCandID][j], nDsWidth, xx, yy);
            xx = xx*nImgScale; yy = yy*nImgScale;
            xx += bestShift_y;
            yy += bestShift_x;

            for (int mm = 0; mm < nImgScale; mm++) {
                if (yy-mm < 0) continue;
                for (int nn = 0; nn < nImgScale; nn++) {
                    if (xx-nn < 0) continue;
                    idx = (yy-mm) * nDsWidth*nImgScale + xx-nn;
                    vnIdxTmp[pts_cnt++] = idx;
                }
            }
        }
    }
    else {vnIdxTmp = mnProtoPtsIdx[nCandID]; pts_cnt = (int)vnProtoPtsCnt[nCandID];}

    for (int i = 0; i < pts_cnt; i++) {
        cvm_cand_tmp.data[vnIdxTmp[i]*3] = cvm_rgb_org.data[vnIdxTmp[i]*3];
        cvm_cand_tmp.data[vnIdxTmp[i]*3+1] = cvm_rgb_org.data[vnIdxTmp[i]*3+1];
        cvm_cand_tmp.data[vnIdxTmp[i]*3+2] = cvm_rgb_org.data[vnIdxTmp[i]*3+2];
    }
}



/* calculate the shift (x,y) between rgb and depth image
 * in:
 * index: indices of the points left in cloud_mod (wrt the original point cloud)
 * cvm_rgb_org: the high resolution rgb image
 * nImgScale: ratio of original and downsampled image
 * nDsWidth: width of downsampled image
 * fn: name of the file the data is written to
 * out:
 * beforeMatch:
 * bestShift_x, bestShift_y: the calculated shift in both directions
 * cloud_mod: point cloud with shifted rgb values
 * beforeMatch, afterMatch: colored image before and after shift was applied, for comparison
 */
void matchLrHr(pcl::PointCloud<pcl::PointXYZRGB> & cloud_mod, const std::vector<int> & index,  const cv::Mat & cvm_rgb_org, int nImgScale,
               int nDsWidth, int & bestShift_x, int & bestShift_y, cv::Mat & beforeMatch, cv::Mat & afterMatch, std::string fn = "")
{
    //get pixel positions of points in cloud
    vector< vector<int> > coords;
    int xx, yy;
    vector <int> pair(2);
    for (int i = 0; i < index.size(); i++)
    {
        GetPixelPos(index[i], nDsWidth, yy, xx);
        xx = xx*nImgScale; yy = yy*nImgScale;
        pair[0] = xx;
        pair[1] = yy;
        coords.push_back(pair);
    }


    //find the optimal shift (in x and y direction)
    double accu;
    double max = 0;
    int maxShift_x = 100;
    int maxShift_y = 100;
    bestShift_x = bestShift_y = 0;

    cv::Vec3b rgb;
    for (int shift_x = -maxShift_x; shift_x <= maxShift_x; shift_x ++)
    {
        for (int shift_y = -maxShift_y; shift_y <= maxShift_y; shift_y ++)
        {
            accu = 0;
            for (int i = 0; i < coords.size(); i++)
            {
                xx = coords[i][0] + shift_x;
                yy = coords[i][1] + shift_y;
                if (xx >= 0 && xx < cvm_rgb_org.rows && yy >= 0 && yy < cvm_rgb_org.cols)
                {
                     rgb = cvm_rgb_org.at<cv::Vec3b>(xx,yy);
                     accu += norm(rgb);
                }
             }
            if (accu > max)
            {
                max = accu;
                bestShift_x = shift_x;
                bestShift_y = shift_y;
            }
         }
    }

    // compute shifted low resolution image and shift RGB values in point cloud
    cv::Mat before_lowRes = cv::Mat::zeros(cvm_rgb_org.rows / nImgScale, cvm_rgb_org.cols / nImgScale, cvm_rgb_org.type());
    cv::Mat after_lowRes = cv::Mat::zeros(cvm_rgb_org.rows / nImgScale, cvm_rgb_org.cols / nImgScale, cvm_rgb_org.type());

    for (int i = 0; i < coords.size(); i++)
    {
        before_lowRes.at<cv::Vec3b>(coords[i][0] / nImgScale, coords[i][1] /nImgScale) = cvm_rgb_org.at<cv::Vec3b>(coords[i][0], coords[i][1]);

        rgb = cvm_rgb_org.at<cv::Vec3b>(coords[i][0] + bestShift_x, coords[i][1] + bestShift_y);
        after_lowRes.at<cv::Vec3b>(coords[i][0] / nImgScale, coords[i][1] / nImgScale) = rgb;
        cloud_mod.points[i].r = rgb[2];
        cloud_mod.points[i].g = rgb[1];
        cloud_mod.points[i].b = rgb[0];
    }

    resize(before_lowRes, beforeMatch, beforeMatch.size());
    resize(after_lowRes, afterMatch, afterMatch.size());
    stringstream s;
    s << "calculated shift (x,y): " << bestShift_x << ", " << bestShift_y;
    cv::putText(afterMatch, s.str(), cv::Point(100,100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 1, 8, false);

    //write point cloud to .pcd file
    if (fn != "")
    {
        if (!cloud_mod.empty())
           pcl::io::savePCDFileASCII(fn, cloud_mod);
        else
            cout << "empty point cloud" << endl;
    }
}





/* select relevant parts of point cloud and write data to pcd file
 * in:
 * cloud: the original point cloud
 * width, height, depth: size of the bounding box (in meters)
 * cThreshRel: in the range [0,1]; relative (to max) threshold for rgb values
 * out:
 * cloud_mod: the modified point cloud (only the relevant parts are left)
 * index: indices of the points left in cloud_mod (wrt the original point cloud)
 * fn: name of the file the data is written to
 */
void RecPcl(const pcl::PointCloud<pcl::PointXYZRGB> & cloud, float width, float height, float depth, float cThreshRel,
            pcl::PointCloud<pcl::PointXYZRGB> & cloud_mod, std::vector<int> & index, std::string fn = "")
{
    //apply RGB threshold
    pcl::PointCloud<pcl::PointXYZRGB> cloud_thresh;
    pcl::copyPointCloud (cloud, cloud_thresh);
    pcl::PointCloud<pcl::PointXYZRGB>::iterator it;

    //find range of the rgb values
    double min, max;
    it = cloud_thresh.points.begin();
    min = max = it->rgb;
    for (it = cloud_thresh.points.begin(); it < cloud_thresh.points.end(); it++)
    {
        if(it->rgb > max)
            max = it->rgb;
        else if(it->rgb < min)
            min = it->rgb;
    }

    //define bounding box
    vector<pcl::PointXYZ> box(2);
    box[0] = pcl::PointXYZ(- width / 2, - height / 2, 0);
    box[1] = pcl::PointXYZ(width / 2, height / 2, depth);

    //set unwanted values to nan
    const float nan = std::numeric_limits<float>::quiet_NaN();
    double threshold = cThreshRel * max;
    for (it = cloud_thresh.points.begin(); it < cloud_thresh.points.end(); it++)
    {
        if(it->rgb < threshold || it->x < box[0].x || it->x > box[1].x || it->y < box[0].y || it->y > box[1].y || it->z < box[0].z || it->z > box[1].z)
            it->x = nan;
    }

    //remove nan
    pcl::removeNaNFromPointCloud(cloud_thresh, cloud_mod, index);


    //write point cloud to .pcd file
    if (fn != "")
    {
        if (!cloud_mod.empty())
           pcl::io::savePCDFileASCII(fn, cloud_mod);
        else
            cout << "empty point cloud" << endl;
    }
}



/* reconstruct high resolution image from modified point cloud
 * in:
 * size: size of the modified point cloud
 * index: the indices of the points in the modified cloud wrt the original cloud
 * nImgScale: ratio of original and downsampled image
 * nDsWidth: width of downsampled image
 * cvm_rgb_org: the original rgb image
 * fn: name of the file the data is written to
 * out:
 * HrImg: the reconstructed high resolution RGB image
 */
void HR_from_Pcl(int size, const std::vector<int> & index,
                 int nImgScale, int nDsWidth, const cv::Mat & cvm_rgb_org, cv::Mat & HrImg, int bestShift_x, int bestShift_y, std::string fn)
{
    std::vector<int> vnIdxTmp(size*nImgScale*nImgScale, 0);
    vector< vector<int> > Idx;
    Idx.push_back(index);
    vector<int> Cnt;
    Cnt.push_back(size);
    SelRecognition_Shift(0, nImgScale, nDsWidth, Cnt, cvm_rgb_org, HrImg, Idx, vnIdxTmp, bestShift_x, bestShift_y);
    cv::imwrite(fn, HrImg);
}






/* choose the right segments for smoothHighRes
 * in:
 * HrImg: the high resolution image before smoothing
 * mnGSegmPts: graph based segmentation result
 * NoErode: number of erode operations to be performed on binary template
 * NoDilate: number of dilate operations to be performed on binary template
 * ShareThresh: threshold for selection of surfaces from graph based segmentation
 * out:
 * fits: fits[i] is true if segment i is chosen, false if not
 * binTemp: binary template after morphological operations were applied
 */
void chooseGbSegs(cv::Mat & HrImg, std::vector< std::vector<CvPoint> > & mnGSegmPts, int NoErode, int NoDilate,  float ShareThresh,
                  vector<bool> & fits, cv::Mat & binTemp)
{
    //number of segments
    int Segnum = mnGSegmPts.size();

    //mapping point -> segment
    cv::Mat cvm_gbsegm(HrImg.size(), CV_32SC1);
    for (int seg = 0; seg < Segnum; seg++)
        for (int p = 0; p < mnGSegmPts[seg].size(); p++)
            cvm_gbsegm.at<int>(mnGSegmPts[seg][p].y, mnGSegmPts[seg][p].x) = seg;

    //binary template
    cv::Mat binary = cv::Mat::zeros(HrImg.rows, HrImg.cols, CV_8UC1);
    cv::cvtColor(HrImg, binary, CV_RGB2GRAY);
    cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY);
    if (NoDilate > 0)
        cv::dilate(binary, binary, cv::Mat(), cv::Point(-1,-1), NoDilate);
    if (NoErode > 0)
        cv::erode(binary, binary, cv::Mat(), cv::Point(-1,-1), NoErode);

    cvtColor(binary, binTemp, CV_GRAY2BGR);

    //choose surfaces
    std::vector <int> share(Segnum);
    for (int seg = 0; seg < Segnum; seg++)
        share[seg] = 0;

    for(int i=0; i<HrImg.rows; i++)
        for(int j=0; j<HrImg.cols; j++)
            if(binary.at<uchar>(i,j) != 0)
            {
                int seg = cvm_gbsegm.at<int>(i,j);
                share[seg] ++;
            }

    for (int seg = 0; seg < Segnum; seg++)
        if (float(share[seg]) / mnGSegmPts[seg].size() > ShareThresh)
            fits[seg] = true;
        else
            fits[seg] = false;

}



/* "smooth" high resolution image using graph based segmentation
 * in:
 * cvm_rgb_org: the original high resolution rgb image
 * HrImg: the high resolution image before smoothing
 * NoErode: number of erode operations to be performed on binary template
 * NoDilate: number of dilate operations to be performed on binary template
 * ShareThresh: threshold for selection of surfaces from graph based segmentation
 * GSegmSigma, GSegmGrThrs, GSegmMinSize: parameters for graph based segmentation
 * name of the file the data is written to
 * out:
 * binTemp: binary template after morphological operations were applied
 * HrSmooth: final result, "smooth" high resolution RGB image
 */
void smoothHighRes(const cv::Mat & cvm_rgb_org, cv::Mat & HrImg, int NoErode, int NoDilate, float ShareThresh, double GSegmSigma, int GSegmGrThrs,
                   int GSegmMinSize, cv::Mat & binTemp, cv::Mat & HrSmooth, std::string fn = "")
{
    //use all values in HrImg that are "not black" (a threshold is applied)
    //double min, max;
    //cv::minMaxLoc(HrImg, &min, &max);
    //cv::threshold(HrImg, HrSmooth, 0.1 * max, 0, cv::THRESH_TOZERO);

    //graph-based segmentation
    std::vector< std::vector<CvPoint> > mnGSegmPts;
    GbSegmentation(cvm_rgb_org, GSegmSigma, GSegmGrThrs, GSegmMinSize, mnGSegmPts);

    //find the right surfaces
    //cv::Mat HrSmooth = cv::Mat::zeros(HrImg.rows, HrImg.cols, HrImg.type());
    vector<bool> fits(mnGSegmPts.size());
    chooseGbSegs(HrImg, mnGSegmPts, NoErode, NoDilate, ShareThresh, fits, binTemp);

    //combine surfaces to high resolution image
    for(int seg = 0; seg < fits.size(); seg++)
    {
        if (fits[seg])
        {
            for (int i = 0; i < mnGSegmPts[seg].size(); i++)
                HrSmooth.at<cv::Vec3b>(mnGSegmPts[seg][i].y, mnGSegmPts[seg][i].x) = cvm_rgb_org.at<cv::Vec3b>(mnGSegmPts[seg][i].y, mnGSegmPts[seg][i].x);
        }
    }

    if (fn != "")
        cv::imwrite(fn, HrSmooth);
}





// return the sum of two struct timespec variables
struct timespec addTimeSpec(struct timespec a, struct timespec b)
{
  struct timespec sum;
  sum.tv_sec = a.tv_sec + b.tv_sec + (a.tv_nsec + b.tv_nsec) / 1000000000;
  sum.tv_nsec = (a.tv_nsec + b.tv_nsec) % 1000000000;
  return sum;
}

// return a < b for two struct timespec variables a and b
bool compareTimeSpec( struct timespec a, struct timespec b )
{
    if (a.tv_sec < b.tv_sec )
        return true;
    else if (a.tv_sec == b.tv_sec && a.tv_nsec < b.tv_nsec)
            return true;
    else
        return false;
}



/* object registration main function
 * returns true when finished
 *
 * in:
 * size_org: size of the high resolution RGB image
 * nImgScale: ratio of original and downsampled image
 * nDsWidth: width of downsampled image
 * maxnum: number of snapshots to be recorded
 * delayS: delay time in seconds
 * width, height, depth: size of the bounding box (in meters)
 * cThreshRel: in the range [0,1]; relative (to max) threshold for rgb values
 * NoErode: number of erode operations to be performed on binary template
 * NoDilate: number of dilate operations to be performed on binary template
 * ShareThresh: threshold for selection of surfaces from graph based segmentation
 * GSegmSigma, GSegmGrThrs, GSegmMinSize: parameters for graph based segmentation
 */
bool Registration( cv::Size size_org, int nImgScale, int nDsWidth,
                   int maxnum, int delayS, float width, float height, float depth, float cThreshRel, int NoErode, int NoDilate, float ShareThresh,
                   double GSegmSigma, int GSegmGrThrs, int GSegmMinSize)
{
    //window to display results
    cv::namedWindow("Object Registration", cv::WINDOW_NORMAL);

    //variables to store point cloud and RGB image
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cv::Mat cvm_rgb_org = cv::Mat::zeros(size_org, CV_8UC3);

    //directory where the data is stored
    static const std::string sPclDir = "Registration_data";
    mkdir (sPclDir.data(), 0777);

    //time control
    struct timespec delay;
    delay.tv_sec = delayS;
    delay.tv_nsec = 0;
    struct timespec t_curTime;
    struct timespec t_next;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_curTime);


    for (int count = 0; count < maxnum; count++)
    {
        t_next = addTimeSpec(t_curTime, delay);

        //signal to the user so he knows when the snapshot is taken
        cout << "click! " << t_curTime.tv_sec << " s,  " << t_curTime.tv_nsec << " ns" << endl;

        //grab point cloud
        m.lock ();
        pcl::fromROSMsg (*cloud_, cloud);
        m.unlock ();

        //grab RGB image
        cvm_image_camera.copyTo(cvm_rgb_org);

        //filenames
        string ext = static_cast<ostringstream*>( &(ostringstream() << (count + 1)))->str();
        string sPcl_fn = sPclDir + "/" + "PointCloud_" + ext + ".pcd";
        string sHrPcl_fn = sPclDir + "/" + "HrPcl_" + ext + ".jpg";
        string sSmoothPcl_fn = sPclDir + "/" + "SmoothPcl_" + ext + ".jpg";

        //image buffer for results
        cv::Mat results = cv::Mat::zeros(2 * cvm_rgb_org.rows, 2 * cvm_rgb_org.cols, cvm_rgb_org.type());

        //select relevant parts of point cloud
        pcl::PointCloud<pcl::PointXYZRGB> cloud_mod;
        std::vector<int> index;
        RecPcl(cloud, width, height, depth, cThreshRel, cloud_mod, index);

        //match point cloud to high resolution image and store the result
        int bestShift_x, bestShift_y;
        cv::Mat beforeMatch = results(cv::Rect(0,0,cvm_rgb_org.cols, cvm_rgb_org.rows));
        cv::Mat afterMatch = results(cv::Rect(cvm_rgb_org.cols,0,cvm_rgb_org.cols, cvm_rgb_org.rows));
        matchLrHr(cloud_mod, index, cvm_rgb_org, nImgScale, nDsWidth, bestShift_x, bestShift_y, beforeMatch, afterMatch, sPcl_fn);

        //reconstruct high resolution image from point cloud
        cv::Mat HrPcl = cv::Mat::zeros(size_org, CV_8UC3);
        HR_from_Pcl(cloud_mod.size(), index, nImgScale, nDsWidth, cvm_rgb_org, HrPcl, bestShift_x, bestShift_y, sHrPcl_fn);

        //smooth high resolution image from point cloud
        cv::Mat binTemp = results(cv::Rect(0,cvm_rgb_org.rows,cvm_rgb_org.cols, cvm_rgb_org.rows));
        cv::Mat final = results(cv::Rect(cvm_rgb_org.cols,cvm_rgb_org.rows,cvm_rgb_org.cols, cvm_rgb_org.rows));
        smoothHighRes(cvm_rgb_org, HrPcl, NoErode, NoDilate, ShareThresh, GSegmSigma, GSegmGrThrs, GSegmMinSize, binTemp, final, sSmoothPcl_fn);

        //draw the results
        cv::imshow("Object Registration", results);
        cv::waitKey(30);

        //time delay
        while ( compareTimeSpec( t_curTime, t_next ))
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_curTime);
    }

    return true;
}







