// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/pcl/typedefs_fwd.h"  // point cloud typedef
#include "elm/io/readnyudepthv2labeled.h"

using namespace cv;
using namespace elm;

namespace ni {

/**
 * @brief Node for Depth Segmentation into surfaces
 */
class NYUV2DepthNode
{
public:
    /**
     * @brief Constructor
     *
     * Register subscriptions.
     *
     * @param nh node handle
     */
    NYUV2DepthNode(ros::NodeHandle &nh)
        : it_(nh),
          name_out_bgr_("/camera/rgb/image_color"),
          name_out_cloud_("/camera/depth_registered/points"),
          name_out_label_("/ni/nyud2/label"),
          count_(0)
    {
        pub_bgr_    = it_.advertise(name_out_bgr_, 1);
        pub_img_label_  = it_.advertise(name_out_label_, 1);
        pub_cloud_ = nh.advertise<CloudXYZ>(name_out_cloud_, 1);

        std::string fname("/media/win/Users/woodstock/dev/data/nyu_depth_v2_labeled.mat");
        reader_.ReadHeader(fname);
    }

    void next()
    {
        Mat bgr, labels, depth;

        reader_.Next(bgr, depth, labels);

        labels.convertTo(labels, CV_16UC1);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr msg_cloud (new pcl::PointCloud<pcl::PointXYZRGB>(static_cast<uint32_t>(depth.cols), static_cast<uint32_t>(depth.rows)));
        msg_cloud->header.frame_id = "some_tf_frame";
        msg_cloud->height = depth.rows,
        msg_cloud->width =  depth.cols;

        // Depth Intrinsic Parameters from NYU Depth v2 toolbox, camera_params.m
        double fx_d = 5.8262448167737955e+02;
        double fy_d = 5.8269103270988637e+02;
        double cx_d = 3.1304475870804731e+02;
        double cy_d = 2.3844389626620386e+02;

        //ELM_COUT_VAR(depth);

        for (int r=0, i=0; r<depth.rows; r++) {

            for (int c=0; c<depth.cols; c++, i++) {

                double z = depth.at<float>(r, c);
                double xd = (static_cast<double>(c)-cx_d)*z/fx_d;
                double yd = (static_cast<double>(r)-cy_d)*z/fy_d;

                cv::Vec3b pixel = bgr.at<cv::Vec3b>(r, c);
                pcl::PointXYZRGB point(pixel[2], pixel[1], pixel[0]);
                point.x = static_cast<float>(xd);
                point.y = static_cast<float>(yd);
                point.z = static_cast<float>(z);
                msg_cloud->points[i] = point;
            }
        }

        // convert to publish map image
        // mimic timestamp of processed point cloud
        std_msgs::Header header;
        //header.stamp = ros::Time().fromNSec(msg->header.stamp*1e3);

        // in color
        sensor_msgs::ImagePtr img_msg_bgr = cv_bridge::CvImage(
                    header,
                    sensor_msgs::image_encodings::BGR8,
                    bgr).toImageMsg();

        pub_bgr_.publish(img_msg_bgr);

        pub_cloud_.publish(msg_cloud);

        sensor_msgs::ImagePtr img_msg_gray = cv_bridge::CvImage(
                    header,
                    sensor_msgs::image_encodings::MONO16,
                    labels).toImageMsg();

        pub_img_label_.publish(img_msg_gray);

//        std::stringstream s;
//        s<<"/home/kashefy/nyudv2/labels/" << count_ << ".png";

//        labels.convertTo(labels, CV_8UC1);
//        cv::imwrite(s.str(), labels);

        ELM_COUT_VAR(count_);
        count_++;
    }

    // members
    // ROS topic members
    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers
    image_transport::Publisher pub_bgr_;    ///< color image publisher
    ros::Publisher pub_cloud_;
    image_transport::Publisher pub_img_label_;    ///< label image publisher

    // IO/topic names
    std::string name_out_bgr_;     ///< publishing topic name
    std::string name_out_cloud_;   ///< Signal name and publishing topic name
    std::string name_out_label_;   ///< publishing topic name

    elm::ReadNYUDepthV2Labeled reader_;

    int count_;
};

} // namespace ni

int main(int argc, char** argv)
{
    /**
     * The ros::init() function needs to see argc and argv so that it can perform
     * any ROS arguments and name remapping that were provided at the command line. For programmatic
     * remappings you can use a different version of init() which takes remappings
     * directly, but for most command-line programs, passing argc and argv is the easiest
     * way to do it.  The third argument to init() is the name of the node.
     *
     * You must call one of the versions of ros::init() before using any other
     * part of the ROS system.
     */
    ros::init(argc, argv, "nyu_v2_depth");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh;
    ni::NYUV2DepthNode nyu_v2_depth_node(nh);

    ros::Rate loop_rate(0.5);
    while (ros::ok()) {

        nyu_v2_depth_node.next();
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}

