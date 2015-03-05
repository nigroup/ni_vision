// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>          // applyColorMap()

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"            // Signal class: Stimulus -> layer activation -> response
#include "elm/core/pcl/typedefs_fwd.h"  // point cloud typedef

#include "ni/layers/depthmap.h"
#include "ni/layers/layerfactoryni.h"

using namespace cv;
using namespace elm;

namespace ni {

/**
 * @brief Node for Surface tracking
 */
class SurfaceTrackingNode
{
public:
    ~SurfaceTrackingNode()
    {
        if(sync_ptr_ != NULL) {

            delete sync_ptr_;
        }

        if(cloud_sub_ptr_ != NULL) {

            delete cloud_sub_ptr_;
        }

        if(img_sub_ptr_ != NULL) {

            delete img_sub_ptr_;
        }
    }

    /**
     * @brief Constructor
     *
     * Register subscriptions.
     *
     * @param nh node handle
     */
    SurfaceTrackingNode(ros::NodeHandle &nh)
        : it_(nh),
          name_in_cld_("/camera/depth_registered/points"),
          name_in_img_("/camera/rgb/image_color"),
          name_in_seg_("/ni/depth_segmentation/depth_segmentation"),
          name_out_("/ni/depth_segmentation/surfaces/image")
    {

        using namespace message_filters; // Subscriber, sync_policies

        cloud_sub_ptr_ = new Subscriber<CloudXYZ>(nh, name_in_cld_, 1);
        img_sub_ptr_ = new Subscriber<sensor_msgs::Image>(nh, name_in_img_, 1);

        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        int queue_size = 10;
        sync_ptr_ = new Synchronizer<MySyncPolicy>(MySyncPolicy(queue_size),
                                                   *cloud_sub_ptr_,
                                                   *img_sub_ptr_);

        sync_ptr_->registerCallback(
                    boost::bind(
                        &SurfaceTrackingNode::callback, this, _1, _2)
                    );

        { // 0
            // Instantiate DepthMap layer
            LayerConfig cfg;
            LayerIONames io;
            io.Input(DepthMap::KEY_INPUT_STIMULUS, name_in_cld_);
            io.Output(DepthMap::KEY_OUTPUT_RESPONSE, "depth_map");
            layers_.push_back(LayerFactoryNI::CreateShared("DepthMap", cfg, io));
        }

        img_pub_ = it_.advertise(name_out_, 1);
    }

protected:


    void callback(const CloudXYZ::ConstPtr& cld, const sensor_msgs::ImageConstPtr& img)
    {
        // Solve all of perception here...
        mtx_.lock ();
        {
            sig_.Clear();

            cloud_.reset(new CloudXYZ(*cld)); // TODO: avoid copy

            sig_.Append(name_in_cld_, cloud_);

            for(size_t i=0; i<layers_.size(); i++) {

                layers_[i]->Activate(sig_);
                layers_[i]->Response(sig_);
            }

            // get calculated depth map

            //imshow("depth_map", ConvertTo8U(sig_.MostRecentMat1f("depth_map")));

            Mat1f img = Mat1f::zeros(50, 50);

            img.setTo(0.f, isnan(img));
            Mat mask_not_assigned = img <= 0.f;

            Mat img_color;

            applyColorMap(ConvertTo8U(img),
                              img_color,
                              COLORMAP_HSV);

            img_color.setTo(Scalar(0), mask_not_assigned);

            imshow("img_color", img_color);
            waitKey(1);

            // convert in preparation to publish depth map image
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(
                        std_msgs::Header(),
                        sensor_msgs::image_encodings::BGR8,
                        img_color).toImageMsg();

            img_pub_.publish(img_msg);
        }
        mtx_.unlock (); // release mutex
    }

    typedef LayerFactoryNI::LayerShared LayerShared;
    typedef std::vector<LayerShared > VecLayerShared;
    typedef message_filters::sync_policies::ApproximateTime<CloudXYZ, sensor_msgs::Image> MySyncPolicy;

    // members
    //MySyncPolicy policy_;
    MySyncPolicy *policy_ptr_;
    message_filters::Synchronizer<MySyncPolicy> *sync_ptr_;
    //message_filters::Synchronizer<MySyncPolicy> sync_;

    message_filters::Subscriber<CloudXYZ > *cloud_sub_ptr_;     ///< point cloud subscriber
    message_filters::Subscriber<sensor_msgs::Image > *img_sub_ptr_;

    message_filters::Subscriber<CloudXYZ > cloud_sub_;     ///< point cloud subscriber
    message_filters::Subscriber<sensor_msgs::Image > img_sub_;

    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers
    image_transport::Subscriber img_sub_seg_;   ///< segmentation map image subsriber
    image_transport::Subscriber img_sub_img_;   ///< raw image subsriber
    image_transport::Publisher img_pub_;    ///< surface map image publisher

    boost::mutex mtx_;                      ///< mutex object for thread safety

    std::string name_in_cld_;    ///< Signal name and subscribing topic name
    std::string name_in_img_;    ///< Signal name and subscribing topic name
    std::string name_in_seg_;    ///< Signal name and subscribing topic name
    std::string name_out_;      ///< Signal name and publishing topic name

    CloudXYZPtr cloud_;  ///< most recent point cloud

    Mat1f img_seg_;
    Mat1f img_;

    VecLayerShared layers_; ///< layer pipeline (ordered list of layer instances)

    Signal sig_;
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
    ros::init(argc, argv, "surface_tracking");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh;
    ni::SurfaceTrackingNode surface_tracking_node(nh);

    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();

    return 0;
}

