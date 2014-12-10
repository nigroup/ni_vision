#include <pcl/io/pcd_io.h>
#include <ctime>
#include "gnuplot-iostream.h"
#include <pcl_ros/filters/filter.h>
#include <limits>
#include <fstream>


//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/image_viewer.h>



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

    if (fn != "")
    {
        //write point cloud data to .pcd file
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
 * nDsWidth: size of the original point cloud
 * cvm_rgb_org: the original rgb image
 * fn: name of the file the data is written to
 * out:
 * HrImg: the reconstructed high resolution RGB image
 */
void HR_from_Pcl(int size, const std::vector<int> & index,
                 int nImgScale, int nDsWidth, const cv::Mat & cvm_rgb_org, cv::Mat HrImg, std::string fn)
{
    std::vector<int> vnIdxTmp(size*nImgScale*nImgScale, 0);
    vector< vector<int> > Idx;
    Idx.push_back(index);
    vector<int> Cnt;
    Cnt.push_back(size);
    SelRecognition_1(0, nImgScale, nDsWidth, Cnt, cvm_rgb_org, HrImg, Idx, vnIdxTmp);
    cv::imwrite(fn, HrImg);
}



/* find surfaces that fit into the bounding box and write them to file
 * in:
 * cloud: the point cloud
 * mnPtsIdx: mapping surface -> points
 * nCnt: number of segments
 * vnPtsCnt: number of points for each surface
 * width, height, depth: size of the bounding box (in meters)
 * fn: name of the file the data is written to
 * out:
 * fits: flags indicating which surfaces were stored
 */
void RecSeg(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
            const vector< vector<int> > & mnPtsIdx, int nCnt, const vector<int> & vnPtsCnt,
            float width, float height, float depth, vector<bool> & fits, std::string fn)
{
    //define bounding box
    vector<pcl::PointXYZ> box(2);
    box[0] = pcl::PointXYZ(- width / 2, - height / 2, 0);
    box[1] = pcl::PointXYZ(width / 2, height / 2, depth);

    //find surfaces that fit into the bounding box
    pcl::PointXYZRGB point;
    for (int surfIdx = 0; surfIdx < nCnt; surfIdx++)
    {
        fits[surfIdx] = true;
        for (int i = 0; i < vnPtsCnt[surfIdx]; i++)
        {
            point = cloud.points[mnPtsIdx[surfIdx][i]];
            if (point.x != point.x || point.x < box[0].x || point.x > box[1].x || point.y < box[0].y || point.y > box[1].y)
            {
                fits[surfIdx] = false;
                break;
            }
        }
    }

    //write surfaces to file
    ofstream fout(fn.c_str());
    for (int surfIdx = 0; surfIdx < nCnt; surfIdx++)
    {
        if (fits[surfIdx])
        {
            for (int i = 0; i < vnPtsCnt[surfIdx]; i++)
            {
                point = cloud.points[mnPtsIdx[surfIdx][i]];
                fout << surfIdx << " " << point.x << " " << point.y << " " << point.z << " " << point.rgb << "\n";
            }
        }
    }
    fout.close();
}



/* reconstruct high resolution image from segmentation result
 * in:
 * mnPtsIdx: mapping surface -> points
 * nCnt: number of segments
 * vnPtsCnt: number of points for each surface
 * fits: flags indicating which surfaces to use
 * nImgScale: ratio of original and downsampled image
 * nDsWidth: size of the original point cloud
 * cvm_rgb_org: the original rgb image
 * fn: name of the file the data is written to
 * out:
 * HrImg: the reconstructed high resolution RGB image
 */
void HR_from_Seg(const vector< vector<int> > & mnPtsIdx, int nCnt, const vector<int> & vnPtsCnt, vector<bool> & fits,
                 int nImgScale, int nDsWidth, const cv::Mat & cvm_rgb_org, cv::Mat & HrImg, std::string fn)
{
    for (int surfIdx = 0; surfIdx < nCnt; surfIdx++)
    {
        if (fits[surfIdx])
        {

            std::vector<int> vnIdxTmp(vnPtsCnt[surfIdx]*nImgScale*nImgScale, 0);
            SelRecognition_1 (surfIdx, nImgScale, nDsWidth, vnPtsCnt, cvm_rgb_org, HrImg, mnPtsIdx, vnIdxTmp);
        }
    }
    cv::imwrite(fn, HrImg);
}



void chooseGbSegs2(cv::Mat & HrImg, std::vector< std::vector<CvPoint> > & mnGSegmPts, vector<bool> & fits)
{
    cv::Mat binary = cv::Mat::zeros(HrImg.rows, HrImg.cols, CV_8UC1);
    cv::cvtColor(HrImg, binary, CV_RGB2GRAY);
    cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY);
    cv::erode(binary, binary, cv::Mat(), cv::Point(-1,-1), 5);

    cv::namedWindow("binary", cv::WINDOW_NORMAL);
    cv::imshow("binary", binary);

    for (int seg = 0; seg<fits.size(); seg++)
        fits[seg] = false;

    for(int i=0; i<HrImg.rows; i++)
    {
        for(int j=0; j<HrImg.cols; j++)
        {
            if(binary.at<uchar>(i,j) != 0)
            {
                for (int seg = 0; seg < mnGSegmPts.size(); seg++)
                {
                    for (int p = 0; p < mnGSegmPts[seg].size(); p++)
                    {
                        if(mnGSegmPts[seg][p].y == i && mnGSegmPts[seg][p].x == j)
                        {
                          fits[seg] = true;
                            break;
                        }
                    }
                }
             }
         }
      }

}


void chooseGbSegs(cv::Mat & HrImg, std::vector< std::vector<CvPoint> > & mnGSegmPts, vector<bool> & fits)
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
    cv::dilate(binary, binary, cv::Mat(), cv::Point(-1,-1), 5);
    cv::erode(binary, binary, cv::Mat(), cv::Point(-1,-1), 5);

    cv::namedWindow("binary", cv::WINDOW_NORMAL);
    cv::imshow("binary", binary);

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

    float shareThresh = 0.3;
    for (int seg = 0; seg < Segnum; seg++)
        if (float(share[seg]) / mnGSegmPts[seg].size() > shareThresh)
            fits[seg] = true;
        else
            fits[seg] = false;

}



void smoothHighRes(const cv::Mat & cvm_rgb_org, cv::Mat & HrImg, std::string fn = "")
{

    //graph-based segmentation
    nGSegmSigma = 1;
    nGSegmGrThrs = 0;
    nGSegmMinSize = 500;
    std::vector< std::vector<CvPoint> > mnGSegmPts;
    GbSegmentation(cvm_rgb_org, nGSegmSigma, nGSegmGrThrs, nGSegmMinSize, mnGSegmPts);

    //find the right surfaces
    cv::Mat HrSmooth = cv::Mat::zeros(HrImg.rows, HrImg.cols, HrImg.type());
    vector<bool> fits(mnGSegmPts.size());
    chooseGbSegs(HrImg, mnGSegmPts, fits);

    //combine surfaces to high resolution image
    for(int seg = 0; seg < fits.size(); seg++)
    {
        if (fits[seg])
        {
            for (int i = 0; i < mnGSegmPts[seg].size(); i++)
                HrSmooth.at<cv::Vec3b>(mnGSegmPts[seg][i].y, mnGSegmPts[seg][i].x) = cvm_rgb_org.at<cv::Vec3b>(mnGSegmPts[seg][i].y, mnGSegmPts[seg][i].x);
        }
    }

    //cv::namedWindow("temp", cv::WINDOW_NORMAL);
    //cv::imshow("temp", HrSmooth);
    if (fn != "")
        cv::imwrite(fn, HrSmooth);


}




/* object registration
 * returns false while the process is not yet finished
 * returns true if all point clouds have been stored
 *
 * maxnum: number of snapshots to be recorded
 * delayS: delay time in seconds
 * width, height, depth: size of the bounding box (in meters)
 * cThreshRel: in the range [0,1]; relative (to max) threshold for rgb values
 * nImgScale: ratio of original and downsampled image
 * nDsWidth: size of the original point cloud
 * cvm_rgb_org: the original rgb image
 * size_org: size of the original RGB image
 * mnPtsIdx: mapping surface -> points
 * nCnt: number of segments
 * vnPtsCnt: number of points for each surface
 */
bool Registration(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                  int maxnum, int delayS, float width, float height, float depth, float cThreshRel,
                  int nImgScale, int nDsWidth, const cv::Mat & cvm_rgb_org, cv::Size size_org,
                  const vector< vector<int> > & mnPtsIdx, int nCnt, const vector<int> & vnPtsCnt)
{

    //directory where the data is stored
    static const std::string sPclDir = "Pcl_data";
    mkdir (sPclDir.data(), 0777);

    //time control
    static clock_t curTime;
    static clock_t lastTime;
    static int count = 0;

    clock_t delay = delayS * CLOCKS_PER_SEC; //delay time in clock units
    curTime = clock();

    if (curTime - lastTime > delay)
    {
        lastTime = curTime;

        //signal to the user so he knows when the snapshot is taken
        cout << "click! " << double(curTime) / CLOCKS_PER_SEC << endl;

        //filenames
        string ext = static_cast<ostringstream*>( &(ostringstream() << (count + 1)))->str();
        string sPcl_fn = sPclDir + "/" + "PointCloud_" + ext + ".pcd";
        string sSurf_fn = sPclDir + "/" + "Surfaces_" + ext + ".txt";
        string sHrSeg_fn = sPclDir + "/" + "HrSeg_" + ext + ".jpg";
        string sHrPcl_fn = sPclDir + "/" + "HrPcl_" + ext + ".jpg";
        string sSmoothPcl_fn = sPclDir + "/" + "SmoothPcl_" + ext + ".jpg";
        string sSmoothSeg_fn = sPclDir + "/" + "SmoothSeg_" + ext + ".jpg";

        //draw original image and bounding box (width and height)
        cv::Mat boxImg = cvm_rgb_org.clone();
        cv::Point pt1(0, boxImg.cols);
        cv::Point pt2(boxImg.rows, 0);
        pcl::PointXYZRGB point;
        for (int i = 0; i < cloud.size(); i++)
        {
            point = cloud.points[i];
            if (point.x >= -width/2 && point.x <=width/2 && point.y >= -height/2  && point.y <= height/2)
            {
                int xx, yy;
                GetPixelPos(i, nDsWidth, xx, yy);
                xx *= nImgScale;
                yy *= nImgScale;

                if(pt1.x < xx)
                    pt1.x = xx;
                if(pt1.y > yy)
                    pt1.y = yy;
                if(pt2.x > xx)
                    pt2.x = xx;
                if(pt2.y < yy)
                    pt2.y = yy;
             }
        }
        cv::Scalar color(255, 255, 0);
        cv::rectangle(boxImg, pt1, pt2, color, 3);
        cv::namedWindow("boxImg", cv::WINDOW_NORMAL);
        cv::imshow("boxImg", boxImg);

        //edit and store point cloud
        pcl::PointCloud<pcl::PointXYZRGB> cloud_mod;
        std::vector<int> index;
        RecPcl(cloud, width, height, depth, cThreshRel, cloud_mod, index, sPcl_fn);

        //reconstruct high resolution image from point cloud
        cv::Mat HrPcl = cv::Mat::zeros(size_org, CV_8UC3);
        HR_from_Pcl(cloud_mod.size(), index, nImgScale, nDsWidth, cvm_rgb_org, HrPcl, sHrPcl_fn);

        //smooth high resolution image from point cloud
        smoothHighRes(cvm_rgb_org, HrPcl, sSmoothPcl_fn);

        //store surfaces that fit into the bounding box
        vector<bool> fits (nCnt);
        RecSeg(cloud, mnPtsIdx, nCnt, vnPtsCnt, width, height, depth, fits, sSurf_fn);

        //reconstruct high resolution image from surfaces
        cv::Mat HrSeg = cv::Mat::zeros(size_org, CV_8UC3);
        HR_from_Seg(mnPtsIdx, nCnt, vnPtsCnt, fits, nImgScale, nDsWidth, cvm_rgb_org, HrSeg, sHrSeg_fn);

        //smooth high resolution image from surfaces
        smoothHighRes(cvm_rgb_org, HrSeg, sSmoothSeg_fn);

        //count how many point clouds have been stored in this round
        //return true when finished
        count++;
        if (count == maxnum)
        {
            count = 0;
            return true;
        }
        else
        {
            return false;
        }

    }
    return false;
}








//other functions tested ////////////////////////////////////////////////////////////////////////////////////


//reconstruct the high resolution RGB image from several samples
bool Registration2(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                   int maxnum, int delayS, float width, float height, float depth, float cThreshRel,
                   int nImgScale, int nDsWidth, const cv::Mat & cvm_rgb_org, cv::Size size_org)
{


    //directory where the data is stored
    static const std::string sPclDir = "Pcl_data";
    mkdir (sPclDir.data(), 0777);

    //time control
    static clock_t curTime;
    static clock_t lastTime;
    static int count = 0;

    clock_t delay = delayS * CLOCKS_PER_SEC; //delay time in clock units
    curTime = clock();

    static bool go = false;
    static int sample = 0;
    static cv::Mat HrImgMult = cv::Mat::zeros(size_org, CV_8UC3);

    if (go || curTime - lastTime > delay)
    {
        if (!go)
        {
            lastTime = curTime;
            go = true;
        }

        //filenames
        string ext = static_cast<ostringstream*>( &(ostringstream() << (count + 1)))->str();
        string sHrPcl_fn = sPclDir + "/" + "HrPcl_" + ext + ".jpg";
        string sHrPclMult_fn = sPclDir + "/" + "HrPclMult_" + ext + ".jpg";


            //edit point cloud
            pcl::PointCloud<pcl::PointXYZRGB> cloud_mod;
            std::vector<int> index;
            RecPcl(cloud, width, height, depth, cThreshRel, cloud_mod, index);

            //get HR image
            cv::Mat HrImgSample= cv::Mat::zeros(size_org, CV_8UC3);
            HR_from_Pcl(cloud_mod.size(), index, nImgScale, nDsWidth, cvm_rgb_org, HrImgSample, sHrPcl_fn);

            //merge sample images
            max(HrImgSample, HrImgMult, HrImgMult);

        sample++;
        cout << sample << endl;
        if(sample == 10)
        {
            sample = 0;
            go = false;
            cv::imwrite(sHrPclMult_fn, HrImgMult);
        }

        sleep(2);

        //count how many point clouds have been stored in this round
        //return true when finished
        if (!go) count++;
        if (count == maxnum)
        {
            count = 0;
            return true;
        }
        else
        {
              return false;
        }


    }
return false;

}




/*
//show PointCloud;  seems to be a problem with threading
pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ptrCloud(&cloud);
pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
viewer.showCloud(ptrCloud);
while (!viewer.wasStopped())
{
}
*/

/*
//try pcl_visualizer class; does this work in a thread?
static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ptrCloud(&cloud);
viewer->addPointCloud<pcl::PointXYZRGB> (ptrCloud, "sample cloud");
viewer->initCameraParameters ();
if (!viewer->wasStopped ())
{
  viewer->spinOnce (100);
  boost::this_thread::sleep (boost::posix_time::microseconds (100000));
}
viewer->removePointCloud();   */

/*
//try to convert point cloud to 2D image; works in thread; viewer can also be declared static
static pcl::visualization::ImageViewer viewer("View Clouds");
viewer.showRGBImage(cloud);   */


//try sending data to gnuplot
//Gnuplot gp;
//gp << "splot \"" << sPcl_fn << "\" using 1:2:3:4 with dots palette\n";






