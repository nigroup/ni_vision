#include "ros/ros.h"

#include "gbseg_node/point.h"
#include "gbseg_node/segment.h"
#include "gbseg_node/segmentation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


// this code shows how the messages published by the gb_segmentation_node
// can be read in a source code file


// this function reads the messages in vector < vector <point> > format
// and converts them to an rgb image where the colors correspond to the segments
void segmentationCallback(const gbseg_node::segmentation::ConstPtr& msg)
{
    cv::Mat img_out = cv::Mat::zeros(cv::Size(msg->width, msg->height), CV_8UC3);
    for (int i = 0; i < msg->segments.size(); i++)
    {
        cv::Vec3b color(rand()%255+1, rand()%255+1, rand()%255+1);
        for (int j = 0; j < msg->segments[i].points.size(); j++)
        {
            img_out.at<cv::Vec3b>(msg->segments[i].points[j].y, msg->segments[i].points[j].x) = color;
        }
    }

    cv::namedWindow("GbSeg results from segment[] format", cv::WINDOW_NORMAL);
    cv::imshow("GbSeg results from segment[] format", img_out);
    cv::waitKey(1);
}


// this function reads the messages in image format
// the labels are converted to colors for visualization purposes
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat img_in = cv_bridge::toCvShare(msg, "32SC1")->image;
    cv::Mat img_out = cv::Mat::zeros(img_in.size(), CV_8UC3);

    double min, max;
    cv::minMaxLoc(img_in, &min, &max);
    std::vector<cv::Vec3b> colors;
    for (int k = 0; k < max; k++)
        colors.push_back(cv::Vec3b(rand()%255+1, rand()%255+1, rand()%255+1));

    for (int i = 0; i < img_in.rows; i++)
        for (int j = 0; j < img_in.cols; j++)
            img_out.at<cv::Vec3b>(i,j) = colors[img_in.at<int>(i,j)];

    cv::namedWindow("GbSeg results from image format", cv::WINDOW_NORMAL);
    cv::imshow("GbSeg results from image format", img_out);
    cv::waitKey(1);
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "result_GbSeg_subscriber");
  ros::NodeHandle nh;
  ros::Subscriber seg_sub = nh.subscribe("result_GbSeg", 1, segmentationCallback);
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber img_sub = it.subscribe("resultImage_GbSeg", 1, imageCallback);
  cv::startWindowThread();
  ros::spin();

  return 0;
}
