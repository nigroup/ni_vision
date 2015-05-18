// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <ni_depth_segmentation/NodeKVPConfig.h>

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
#include "ni/layers/depthgradientsmoothing.h"
#include "ni/layers/depthsegmentation.h"
#include "ni/layers/mapareafilter.h"
#include "ni/layers/layerfactoryni.h"
#include "ni/legacy/timer.h"


using namespace cv;
using namespace elm;

namespace ni {

/**
 * @brief Node for Depth Segmentation into surfaces
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
          name_out_map_color_("/ni/depth_segmentation/depth_segmentation/map_image_color"),
          name_out_map_gray_("/ni/depth_segmentation/depth_segmentation/map_image_gray")
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
        cloud_sub_ = nh.subscribe<CloudXYZ>(name_in_, 1, &DepthSegmentationNode::callback, this);

        // Set up a dynamic reconfigure server.
        // This should be done before reading parameter server values.
        {
            dynamic_reconfigure::Server<ni_depth_segmentation::NodeKVPConfig>::CallbackType cb;
            cb = boost::bind(&DepthSegmentationNode::configCallback, this, _1, _2);
            dr_srv_.setCallback(cb);
        }

        initLayers(nh);

        img_pub_bgr_    = it_.advertise(name_out_map_color_, 1);
        img_pub_gray_   = it_.advertise(name_out_map_gray_, 1);
    }

protected:

    void initLayers(ros::NodeHandle &nh)
    {
        { // 0
            // Instantiate DepthMap layer
            LayerConfig cfg;
            LayerIONames io;
            io.Input(DepthMap::KEY_INPUT_STIMULUS, name_in_);
            io.Output(DepthMap::KEY_OUTPUT_RESPONSE, "depth_map");
            layers_.push_back(LayerFactoryNI::CreateShared("DepthMap", cfg, io));
        }
        { // 1
            // Instantiate Depth Gradient layer
            LayerConfig cfg;

            PTree p;
            p.put(DepthGradient::PARAM_GRAD_WEIGHT, 0.5f);
            cfg.Params(p);

            LayerIONames io;
            io.Input(DepthGradient::KEY_INPUT_STIMULUS, "depth_map");
            io.Output(DepthGradient::KEY_OUTPUT_GRAD_X, "depth_grad_x");
            io.Output(DepthGradient::KEY_OUTPUT_GRAD_Y, "depth_grad_y");
            layers_.push_back(LayerFactoryNI::CreateShared("DepthGradient", cfg, io));
        }
//        { // 2
//            // Instantiate MedianBlur layer
//            // applied on vertical gradient component
//            LayerConfig cfg;

//            PTree p;
//            p.put(MedianBlur::PARAM_APERTURE_SIZE, 5);
//            cfg.Params(p);

//            LayerIONames io;
//            io.Input(MedianBlur::KEY_INPUT_STIMULUS, "depth_grad_y");
//            io.Output(MedianBlur::KEY_OUTPUT_RESPONSE, "depth_grad_y_smooth");
//            layers_.push_back(LayerFactoryNI::CreateShared("MedianBlur", cfg, io));
//        }
        { // 2
            // Instantiate Depth gradient smoothing layer
            // applied on vertical gradient component
            LayerConfig cfg;

            PTree p;
            p.put(DepthGradientSmoothing::PARAM_APERTURE_SIZE, 5);
            p.put(DepthGradientSmoothing::PARAM_BAND_1, 5);
            p.put(DepthGradientSmoothing::PARAM_BAND_2, 14);
            p.put(DepthGradientSmoothing::PARAM_FILTER_MODE, 2);
            p.put(DepthGradientSmoothing::PARAM_MAX, 0.04f);
            p.put(DepthGradientSmoothing::PARAM_SMOOTH_CENTER, 128);
            p.put(DepthGradientSmoothing::PARAM_SMOOTH_FACTOR, 3);
            p.put(DepthGradientSmoothing::PARAM_SMOOTH_MODE, 2);
            cfg.Params(p);

            LayerIONames io;
            io.Input(DepthGradientSmoothing::KEY_INPUT_STIMULUS, "depth_grad_y");
            io.Output(DepthGradientSmoothing::KEY_OUTPUT_RESPONSE, "depth_grad_y_smooth");
            layers_.push_back(LayerFactoryNI::CreateShared("DepthGradientSmoothing", cfg, io));
        }
        { // 3
            // Instantiate layer for rectifying smoothed gradient
            // apply thresholds on raw gradient and have them reflect on smoothed component
            LayerConfig cfg;

            PTree p;
            p.put(DepthGradientRectify::PARAM_MAX_GRAD, 0.04f); // paper = 0.04
            cfg.Params(p);

            LayerIONames io;
            io.Input(DepthGradientRectify::KEY_INPUT_GRAD_X, "depth_grad_x");
            io.Input(DepthGradientRectify::KEY_INPUT_GRAD_Y, "depth_grad_y");
            io.Input(DepthGradientRectify::KEY_INPUT_GRAD_SMOOTH, "depth_grad_y_smooth");
            io.Output(DepthGradientRectify::KEY_OUTPUT_RESPONSE, "depth_grad_y_smooth_r");
            layers_.push_back(LayerFactoryNI::CreateShared("DepthGradientRectify", cfg, io));
        }
        {   // 4
            // Instantiate Depth segmentation layer
            // applied on smoothed vertical gradient component
            double tmp;
            double tmp_default = static_cast<double>(DepthSegmentation::DEFAULT_MAX_GRAD);
            nh.param<double>(DepthSegmentation::PARAM_MAX_GRAD, tmp, tmp_default);
            float max_grad = static_cast<float>(tmp);

            LayerConfig cfg;

            PTree params;
            params.put(DepthSegmentation::PARAM_MAX_GRAD, max_grad); // paper = 0.003
            cfg.Params(params);

            LayerIONames io;
            io.Input(DepthSegmentation::KEY_INPUT_STIMULUS, "depth_grad_y_smooth_r");
            io.Output(DepthSegmentation::KEY_OUTPUT_RESPONSE, "depth_seg_raw");
            layers_.push_back(LayerFactoryNI::CreateShared("DepthSegmentation", cfg, io));
        }
        {
            // Instantiate Map Area Filter layer for smoothing surfaces
            // by merging small-sized surfaces together
            // then merging them with largest neighbor
            LayerConfig cfg;

            PTree params;
            params.put(MapAreaFilter::PARAM_TAU_SIZE, 200); // @todo adapt this threshold to higher resolution input
            cfg.Params(params);

            LayerIONames io;
            io.Input(MapAreaFilter::KEY_INPUT_STIMULUS, "depth_seg_raw");
            io.Output(MapAreaFilter::KEY_OUTPUT_RESPONSE, name_out_map_gray_);
            layers_.push_back(LayerFactoryNI::CreateShared("MapAreaFilter", cfg, io));
        }
    }

    /**
     * @brief Point cloud callback
     * @param msg point cloud message
     */
    void callback(const CloudXYZ::ConstPtr& msg)
    {
        mtx_.lock ();
        {
            struct timespec t_total_start, t_total_end;
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_start);

            cloud_.reset(new CloudXYZ(*msg)); // TODO: avoid copy

            sig_.Clear();
            sig_.Append(name_in_, cloud_);

            for(size_t i=0; i<layers_.size(); i++) {

                layers_[i]->Activate(sig_);
                layers_[i]->Response(sig_);
            }

            // get calculated depth map
            //Mat1f img = sig_.MostRecentMat1f("depth_seg_raw");
            Mat1f img = sig_.MostRecentMat1f(name_out_map_gray_);
            //Mat1f img(50, 50, 0.f);

//            double min_val, max_val;
//            minMaxIdx(sig_.MostRecentMat1f("depth_grad_y_smooth"), &min_val, &max_val);
//            ELM_COUT_VAR("s"<<min_val << " " << max_val);
//            minMaxIdx(sig_.MostRecentMat1f("depth_grad_y_smooth_r"), &min_val, &max_val);
//            ELM_COUT_VAR("r"<<min_val << " " << max_val);

//            imshow("s", ConvertTo8U(sig_.MostRecentMat1f("depth_grad_y_smooth")));
//            imshow("r", ConvertTo8U(sig_.MostRecentMat1f("depth_grad_y_smooth_r")));

            img.setTo(0.f, isnan(img));
            Mat mask_not_assigned = img <= 0.f;

            Mat img_color;

            Mat1b img_gray = ConvertTo8U(img);

            applyColorMap(img_gray,
                          img_color,
                          COLORMAP_HSV);

            img_color.setTo(Scalar(0), mask_not_assigned);

            //imshow("img_color", img_color);
            //imshow("img_gray", img_gray);
            //waitKey(1);

            // convert to publish map image
            // mimic timestamp of processed point cloud
            std_msgs::Header header;
            header.stamp = ros::Time().fromNSec(msg->header.stamp*1e3);

            // in color
            sensor_msgs::ImagePtr img_msg_color = cv_bridge::CvImage(
                        header,
                        sensor_msgs::image_encodings::BGR8,
                        img_color).toImageMsg();

            img_pub_bgr_.publish(img_msg_color);

            // in grayscale
            sensor_msgs::ImagePtr img_msg_gray = cv_bridge::CvImage(
                        header,
                        sensor_msgs::image_encodings::MONO8,
                        img_gray).toImageMsg();

            img_pub_gray_.publish(img_msg_gray);

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_end);
            double nTimeTotal = double(timespecDiff(&t_total_end, &t_total_start)/1e9);

            ROS_INFO("running at %f Hz", 1./nTimeTotal);
        }
        mtx_.unlock (); // release mutex
    }

    void configCallback(ni_depth_segmentation::NodeKVPConfig &config, uint32_t level)
    {
        ROS_INFO("KVP change:");
        // Set class variables to new values. They should match what is input at the dynamic reconfigure GUI.
        std::string key = config.key;
        double a = config.a;

        ROS_INFO("\t%s:=%f", key.c_str(), a);

        if(layers_.size() > 0) {

            if(key == DepthSegmentation::PARAM_MAX_GRAD) {

                // 4
                ///<@todo reconfigure layer without indexing
                // Instantiate Depth segmentation layer
                // applied on smoothed vertical gradient component
                LayerConfig cfg;

                PTree params;

                float new_value = static_cast<float>(a);
                params.put(DepthSegmentation::PARAM_MAX_GRAD, new_value);
                cfg.Params(params);

                layers_[4]->Reconfigure(cfg);
            }
        }
    }

    typedef LayerFactoryNI::LayerShared LayerShared;
    typedef std::vector<LayerShared > VecLayerShared;

    // members
    // ROS topic members
    ros::Subscriber cloud_sub_;     ///< point cloud subscriber

    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers
    image_transport::Publisher img_pub_bgr_;    ///< segmentation map image publisher colored
    image_transport::Publisher img_pub_gray_;    ///< segmentation map image publisher monochrome

    dynamic_reconfigure::Server<ni_depth_segmentation::NodeKVPConfig> dr_srv_;

    boost::mutex mtx_;                      ///< mutex object for thread safety

    // IO/topic names
    std::string name_in_;   ///< Signal name and subscribing topic name
    std::string name_out_map_color_;   ///< publishing topic name
    std::string name_out_map_gray_;   ///< Signal name and publishing topic name

    // processing members
    CloudXYZPtr cloud_;  ///< most recent point cloud

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
    ros::init(argc, argv, "depth_segmentation");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh("~");
    ni::DepthSegmentationNode depth_segmentation_node(nh);

    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();

    return 0;
}

