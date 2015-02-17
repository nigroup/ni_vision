// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "elm/core/pcl/typedefs_fwd.h"

/**
 * @brief The DepthMap node
 * @todo Is the mutex crucial?
 */
class DepthMapNode
{
public:
    /**
     * @brief Constructor
     *
     * Register subscriptions.
     *
     * @param nh node handle
     */
    DepthMapNode(ros::NodeHandle &nh)
        : it_(nh)
    {
        /**
         * The subscribe() call is how you tell ROS that you want to receive messages
         * on a given topic.  This invokes a call to the ROS
         * master node, which keeps a registry of who is publishing and who
         * is subscribing.  Messages are passed to a callback function, here
         * called chatterCallback.  subscribe() returns a Subscriber object that you
         * must hold on to until you want to unsubscribe.  When all copies of the Subscriber
         * object go out of scope, this callback will automatically be unsubscribed from
         * this topic.
         *
         * The second parameter to the subscribe() function is the size of the message
         * queue.  If messages are arriving faster than they are being processed, this
         * is the number of messages that will be buffered up before beginning to throw
         * away the oldest ones.
         */
        cloud_sub_ = nh.subscribe<elm::CloudXYZ>("/camera/depth_registered/points", 30, &DepthMapNode::callback, this);

        img_pub_ = it_.advertise("/image_converter/output_video", 1);


    }

protected:
    /**
     * @brief Point cloud callback
     * @param msg point cloud message
     */
    void callback(const elm::CloudXYZ::ConstPtr& msg)
    {
        mtx_.lock ();

        printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);

//        BOOST_FOREACH (const pcl::PointXYZ& pt, msg->points) {

//            printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
//        }

        cloud_ = msg;

        cv::Mat image(50, 50, CV_8UC1);
        randn(image, 0, 200);
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();

        img_pub_.publish(img_msg);

        mtx_.unlock ();
    }

    // members
    ros::Subscriber cloud_sub_;     ///< point cloud subscriber

    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers
    image_transport::Publisher img_pub_;    ///< depth map image publisher

    boost::mutex mtx_;                      ///< mutex object for thread safety

    boost::shared_ptr<const elm::CloudXYZ > cloud_;  ///< most recent point cloud
};

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
    ros::init(argc, argv, "depth_map");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh;
    DepthMapNode depth_map_node(nh);

    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();

    return 0;
}

