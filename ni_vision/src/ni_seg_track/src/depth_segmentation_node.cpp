// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "elm/core/cv/mat_utils.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"            // Signal class: Stimulus -> layer activation -> response
#include "elm/core/pcl/typedefs_fwd.h"  // point cloud typedef

#include "ni/layers/depthmap.h"
#include "ni/layers/layerfactoryni.h"

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
        : it_(nh),
          name_in_("/camera/depth_registered/points"),
          name_out_("/ni/seg_track/depth_segmentation")
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
        cloud_sub_ = nh.subscribe<elm::CloudXYZ>(name_in_, 30, &DepthMapNode::callback, this);

        {
            // Instantiate DepthMap layer
            elm::LayerConfig cfg;
            elm::LayerIONames io;
            io.Input(ni::DepthMap::KEY_INPUT_STIMULUS, name_in_);
            io.Output(ni::DepthMap::KEY_OUTPUT_RESPONSE, "depth_map");
            layers_.push_back(ni::LayerFactoryNI::CreateShared("DepthMap", cfg, io));
        }
        {
            // Instantiate Depth Gradient layer
            elm::LayerConfig cfg;
            elm::LayerIONames io;
            io.Input(ni::DepthMap::KEY_INPUT_STIMULUS, "depth_map");
            io.Output(ni::DepthMap::KEY_OUTPUT_RESPONSE, "depth_grad");
            layers_.push_back(ni::LayerFactoryNI::CreateShared("DepthGrad", cfg, io));
        }

        img_pub_ = it_.advertise(name_out_, 1);
    }

protected:
    /**
     * @brief Point cloud callback
     * @param msg point cloud message
     */
    void callback(const elm::CloudXYZ::ConstPtr& msg)
    {
        mtx_.lock ();
        {
            sig_.Clear();

            cloud_.reset(new elm::CloudXYZ(*msg)); // TODO: avoid copy

            sig_.Append(name_in_, cloud_);

            layer_->Activate(sig_);
            layer_->Response(sig_);

            // get calculated depth map
            cv::Mat1f img = sig_.MostRecentMat1f(name_out_);

            // convert in preparation to publish depth map image
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(
                        std_msgs::Header(),
                        sensor_msgs::image_encodings::TYPE_32FC1,
                        img).toImageMsg();

            img_pub_.publish(img_msg);
        }
        mtx_.unlock (); // release mutex
    }

    typedef ni::LayerFactoryNI::LayerShared LayerShared;
    typedef std::vector<LayerShared > VecLayerShared;

    // members
    ros::Subscriber cloud_sub_;     ///< point cloud subscriber

    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers
    image_transport::Publisher img_pub_;    ///< depth map image publisher

    boost::mutex mtx_;                      ///< mutex object for thread safety

    std::string name_in_;   ///< Signal name and subscribing topic name
    std::string name_out_;   ///< Signal name and publishing topic name

    elm::CloudXYZPtr cloud_;  ///< most recent point cloud

    VecLayerShared layers_; ///< layer pipeline (ordered list of layer instances)

    elm::Signal sig_;
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

