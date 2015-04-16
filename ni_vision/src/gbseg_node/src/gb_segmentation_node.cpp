// ros and OpenCV related includes
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// message definitions
#include "gbseg_node/params_GbSeg.h"
#include "gbseg_node/point.h"
#include "gbseg_node/segment.h"
#include "gbseg_node/segmentation.h"

// function that converts between image formats
// and calls the graph-based segmentation function
#include "func_segmentation_gb.hpp"


// default parameters
static double GSegmSigma_def_ = 1.0;
static int GSegmGrThrs_def_ = 100;
static int GSegmMinSize_def_ = 100;
static bool show_def_ = true;




class GbSegmentationNode
{
public:

    // constructor
    GbSegmentationNode(ros::NodeHandle &nh, int argc, char **argv)
        : it_(nh), name_in_img_("/camera/rgb/image_color"), name_in_param_("params_GbSeg"),
          name_out_result_("result_GbSeg"), name_out_result_image_("resultImage_GbSeg")
    {
        // subscribe to image and parameter topic
        img_sub_ = it_.subscribe(name_in_img_, 1, &GbSegmentationNode::imageCallback, this);
        param_sub_ = nh.subscribe(name_in_param_, 1, &GbSegmentationNode::parameterCallback, this);

        // advertise topics to publish results
        result_pub_ = nh.advertise<gbseg_node::segmentation>(name_out_result_, 1, true);
        result_image_pub_ = it_.advertise(name_out_result_image_, 1, true);

        // set parameters
        if ( !parseCmd(argc, argv, "-GSegmSigma", GSegmSigma_) )
            GSegmSigma_ = GSegmSigma_def_;
        if ( !parseCmd(argc, argv, "-GSegmGrThrs", GSegmGrThrs_) )
            GSegmGrThrs_ = GSegmGrThrs_def_;
        if ( !parseCmd(argc, argv, "-GSegmMinSize", GSegmMinSize_) )
            GSegmMinSize_ = GSegmMinSize_def_;
        if ( !parseCmd(argc, argv, "-show", show_) )
            show_ = show_def_;

        // inform user
        std::cout << "starting gb_segmentation node with parameters" << std::endl;
        std::cout << "GSegmSigma = " << GSegmSigma_ << ", GSegmGrThrs = " << GSegmGrThrs_
                  << ", GSegmMinSize = " << GSegmMinSize_  << std::endl;


        // start thread for display window
        cv::startWindowThread();
    }

protected:

    // auxiliary functions

    // parse command line arguments
    template <typename T>
    int parseCmd(int argc, char** argv, const char* key, T & val)
    {
        for (int i = 1; i < argc; i++)
        {
            if ((strcmp (argv[i], key) == 0) && (++i < argc))
            {
                val = atof(argv[i]);
                return i-1;
            }
        }
        return 0;
    }

    // convert segmentation result into labeled image (labels correspond to segments)
    void GSegmPts2labeled(std::vector< std::vector < CvPoint > > & mnGSegmPts, cv::Mat & img_labeled)
    {
        for (int i = 0; i < mnGSegmPts.size(); i++)
        {
            for (int j = 0; j < mnGSegmPts[i].size(); j++)
                img_labeled.at<int>(mnGSegmPts[i][j].y, mnGSegmPts[i][j].x) = i;
        }
    }

    // convert segmentation result into rgb image (for visualization)
    void GSegmPts2rgb(std::vector< std::vector < CvPoint > > & mnGSegmPts, cv::Mat & img_show)
    {
        for (int i = 0; i < mnGSegmPts.size(); i++)
        {
            cv::Vec3b color(rand()%255+1, rand()%255+1, rand()%255+1);
            for (int j = 0; j < mnGSegmPts[i].size(); j++)
                img_show.at<cv::Vec3b>(mnGSegmPts[i][j].y, mnGSegmPts[i][j].x) = color;
        }
    }



    // callback functions

    // image callback
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        // grab image from sensor message and perform graph-based segmentation
        cv::Mat img_in_ =  cv_bridge::toCvShare(msg, "bgr8")->image;
        std::vector< std::vector< CvPoint > > mnGSegmPts_;
        GbSegmentation(img_in_, GSegmSigma_, GSegmGrThrs_, GSegmMinSize_, mnGSegmPts_);

        // convert results into labeled image
        cv::Mat img_labeled_ = cv::Mat::zeros(img_in_.size(), CV_32SC1);
        GSegmPts2labeled(mnGSegmPts_, img_labeled_);

        // visualization
        if (show_)
        {
            cv::Mat img_show_ = cv::Mat::zeros(img_in_.size(), CV_8UC3);
            GSegmPts2rgb(mnGSegmPts_, img_show_);
            cv::namedWindow("Graph-Based Image Segmentation", cv::WINDOW_NORMAL);
            cv::imshow("Graph-Based Image Segmentation", img_show_);
        }

        // publish the results in vector < vector <point> > format
        gbseg_node::segmentation segmentation_msg;
        segmentation_msg.width = img_in_.cols;
        segmentation_msg.height = img_in_.rows;
        for (int i = 0; i < mnGSegmPts_.size(); i++)
        {
            gbseg_node::segment segment_msg;
            gbseg_node::point point_msg;
            for (int j = 0; j < mnGSegmPts_[i].size(); j++)
            {
                point_msg.x = mnGSegmPts_[i][j].x;
                point_msg.y = mnGSegmPts_[i][j].y;
                segment_msg.points.push_back(point_msg);
            }
            segmentation_msg.segments.push_back(segment_msg);
        }
        result_pub_.publish(segmentation_msg);

        // publish the results as a labeled image
        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "32SC1", img_labeled_).toImageMsg();
        result_image_pub_.publish(image_msg);

    }

    // parameter callback
    void parameterCallback(const gbseg_node::params_GbSeg::ConstPtr& msg)
    {
        // reset parameters
        GSegmSigma_ = msg->GSegmSigma;
        GSegmGrThrs_ = msg->GSegmGrThrs;
        GSegmMinSize_ = msg->GSegmMinSize;
        show_ = bool(msg->show);

        // inform user
        std::cout << "resetting parameters..." << std::endl;
        std::cout << "GSegmSigma = " << GSegmSigma_ << ", GSegmGrThrs = " << GSegmGrThrs_
                  << ", GSegmMinSize = " << GSegmMinSize_ << std::endl;

        // close display window
        if (!show_)
            cv::destroyWindow("Graph-Based Image Segmentation");
    }


    // members ...

    // ... ROS related
    std::string name_in_img_;
    std::string name_in_param_;
    std::string name_out_result_;
    std::string name_out_result_image_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber img_sub_;
    image_transport::Publisher result_image_pub_;
    ros::Publisher result_pub_;
    ros::Subscriber param_sub_;

    // ... parameters
    double GSegmSigma_;
    int GSegmGrThrs_;
    int GSegmMinSize_;
    bool show_;

};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "gb_segmentation");
    ros::NodeHandle nh;
    GbSegmentationNode gb_segmentation_node(nh, argc, argv);
    ros::spin();
    return 0;
}

