// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

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

#include "elm/layers/medianblur.h"

#include "ni/layers/depthmap.h"
#include "ni/layers/depthgradient.h"
#include "ni/layers/depthgradientrectify.h"
#include "ni/layers/depthsegmentation.h"
#include "ni/layers/mapareafilter.h"
#include "ni/layers/layerfactoryni.h"


/**
 * @brief The DepthMap node
 * @todo Is the mutex crucial?
 */
class DepthSegmentationNode
{
public:
    /**
     * @brief Constructor
     *
     * Register subscriptions.
     *
     * @param nh node handle
     */
    DepthSegmentationNode(ros::NodeHandle &nh)
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
        cloud_sub_ = nh.subscribe<elm::CloudXYZ>(name_in_, 1, &DepthSegmentationNode::callback, this);

        { // 0
            // Instantiate DepthMap layer
            elm::LayerConfig cfg;
            elm::LayerIONames io;
            io.Input(ni::DepthMap::KEY_INPUT_STIMULUS, name_in_);
            io.Output(ni::DepthMap::KEY_OUTPUT_RESPONSE, "depth_map");
            //io.Output(ni::DepthMap::KEY_OUTPUT_RESPONSE, name_out_);
            layers_.push_back(ni::LayerFactoryNI::CreateShared("DepthMap", cfg, io));
        }
        { // 1
            // Instantiate Depth Gradient layer
            elm::LayerConfig cfg;

            elm::PTree p;
            p.put(ni::DepthGradient::PARAM_GRAD_WEIGHT, 0.5f);
            cfg.Params(p);

            elm::LayerIONames io;
            io.Input(ni::DepthGradient::KEY_INPUT_STIMULUS, "depth_map");
            io.Output(ni::DepthGradient::KEY_OUTPUT_GRAD_X, "depth_grad_x");
            io.Output(ni::DepthGradient::KEY_OUTPUT_GRAD_Y, "depth_grad_y");
            //io.Output(ni::DepthGradient::KEY_OUTPUT_GRAD_Y, name_out_);
            layers_.push_back(ni::LayerFactoryNI::CreateShared("DepthGradient", cfg, io));
        }
        { // 2
            // Instantiate MedianBlur layer
            // applied on vertical gradient component
            elm::LayerConfig cfg;

            elm::PTree p;
            p.put(elm::MedianBlur::PARAM_APERTURE_SIZE, 5);
            cfg.Params(p);

            elm::LayerIONames io;
            io.Input(elm::MedianBlur::KEY_INPUT_STIMULUS, "depth_grad_y");
            io.Output(elm::MedianBlur::KEY_OUTPUT_RESPONSE, "depth_grad_y_smooth");
            //io.Output(elm::MedianBlur::KEY_OUTPUT_RESPONSE, name_out_);
            layers_.push_back(ni::LayerFactoryNI::CreateShared("MedianBlur", cfg, io));
        }
        { // 3
            // Instantiate layer for rectifying smoothed gradient
            // apply thresholds on raw gradient and have them reflect on smoothed component
            elm::LayerConfig cfg;

            elm::PTree p;
            p.put(ni::DepthGradientRectify::PARAM_MAX_GRAD, 0.04f);
            cfg.Params(p);

            elm::LayerIONames io;
            io.Input(ni::DepthGradientRectify::KEY_INPUT_GRAD_X, "depth_grad_x");
            io.Input(ni::DepthGradientRectify::KEY_INPUT_GRAD_Y, "depth_grad_y");
            io.Input(ni::DepthGradientRectify::KEY_INPUT_GRAD_SMOOTH, "depth_grad_y_smooth");
            io.Output(ni::DepthGradientRectify::KEY_OUTPUT_RESPONSE, "depth_grad_y_smooth_r");
            //io.Output(elm::MedianBlur::KEY_OUTPUT_RESPONSE, name_out_);
            layers_.push_back(ni::LayerFactoryNI::CreateShared("DepthGradientRectify", cfg, io));
        }
        {
            // Instantiate Depth segmentation layer
            // applied on smoothed vertical gradient component
            elm::LayerConfig cfg;

            elm::PTree params;
            params.put(ni::DepthSegmentation::PARAM_MAX_GRAD, 0.003f);
            cfg.Params(params);

            elm::LayerIONames io;
            io.Input(ni::DepthSegmentation::KEY_INPUT_STIMULUS, "depth_grad_y_smooth_r");
            io.Output(ni::DepthSegmentation::KEY_OUTPUT_RESPONSE, "depth_seg_raw");
            //io.Output(ni::DepthSegmentation::KEY_OUTPUT_RESPONSE, name_out_);
            layers_.push_back(ni::LayerFactoryNI::CreateShared("DepthSegmentation", cfg, io));
        }
        {
            // Instantiate Map Area Filter layer for smoothing surfaces
            // by merging small-sized surfaces together
            // then merging them with largest neighbor
            elm::LayerConfig cfg;

            elm::PTree params;
            params.put(ni::MapAreaFilter::PARAM_TAU_SIZE, 200);
            cfg.Params(params);

            elm::LayerIONames io;
            io.Input(ni::MapAreaFilter::KEY_INPUT_STIMULUS, "depth_seg_raw");
            io.Output(ni::MapAreaFilter::KEY_OUTPUT_RESPONSE, name_out_);
            layers_.push_back(ni::LayerFactoryNI::CreateShared("MapAreaFilter", cfg, io));
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

            for(size_t i=0; i<layers_.size(); i++) {

                layers_[i]->Activate(sig_);
                layers_[i]->Response(sig_);
            }

            // get calculated depth map
            cv::Mat1f img = sig_.MostRecentMat1f(name_out_);

            double min_val, max_val;
            cv::minMaxIdx(img, &min_val, &max_val);

            cv::imshow("img", elm::ConvertTo8U(img));
            cv::waitKey(1);

            img.setTo(0.f, elm::isnan(img));
            cv::Mat mask_not_assigned = img <= 0.f;

            cv::Mat img_color;

            cv::applyColorMap(elm::ConvertTo8U(img),
                              img_color,
                              cv::COLORMAP_HSV);

            img_color.setTo(cv::Scalar(0), mask_not_assigned);

            cv::imshow("img_color", img_color);

            // convert in preparation to publish depth map image
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(
                        std_msgs::Header(),
                        sensor_msgs::image_encodings::BGR8,
                        img_color).toImageMsg();

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
    DepthSegmentationNode depth_map_node(nh);

    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();

    return 0;
}

