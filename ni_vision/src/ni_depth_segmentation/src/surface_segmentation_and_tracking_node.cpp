// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_reconfigure/server.h>
#include <ni_depth_segmentation/NodeKVPConfig.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>          // applyColorMap()

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/graph/graphattr.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"            // Signal class: Stimulus -> layer activation -> response
#include "elm/core/pcl/typedefs_fwd.h"  // point cloud typedef

#include "ni/core/color_utils.h"
#include "ni/core/surface.h"
#include "ni/layers/depthmap.h"
#include "ni/layers/depthgradient.h"
#include "ni/layers/depthgradientrectify.h"
#include "ni/layers/depthgradientsmoothing.h"
#include "ni/layers/depthsegmentation.h"
#include "ni/layers/mapareafilter.h"
#include "ni/layers/surfacetracking.h"
#include "ni/layers/layerfactoryni.h"
#include "ni/legacy/timer.h"

/** A post from ROS Answers suggested using image_transport::SubscriberFilter
 *  source: http://answers.ros.org/question/9705/synchronizer-and-image_transportsubscriber/
 *
 *  @todo get rid of the #ifdef and choose one
 */
#define USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER 1

using namespace cv;
using namespace elm;

namespace ni {

/**
 * @brief Node for Surface segmentation using depth information and tracking surfaces over time
 */
class SurfaceSegAndTrackingNode
{
public:
    ~SurfaceSegAndTrackingNode()
    {
        if(sync_ptr_ != NULL) {

            delete sync_ptr_;
        }
    }

    /**
     * @brief Constructor
     *
     * Register subscriptions and publications.
     *
     * @param nh node handle reference
     * @todo determine ideal queue size for subscribers and sync policy
     */
    SurfaceSegAndTrackingNode(ros::NodeHandle &nh)
        : it_(nh),
          name_in_cld_("/camera/depth_registered/points"),
          name_in_img_("/camera/rgb/image_color"),
          name_out_("/ni/depth_segmentation/surfaces/image"),
#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
          img_sub_(it_, name_in_img_, 10),
#else // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
          img_sub_(nh, name_in_img_, 10),
#endif // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
          cloud_sub_(nh, name_in_cld_, 10)
    {

        using namespace message_filters; // Subscriber, sync_policies

        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        int queue_size = 10;
        sync_ptr_ = new Synchronizer<MySyncPolicy>(MySyncPolicy(queue_size),
                                                   cloud_sub_,
                                                   img_sub_);

        sync_ptr_->registerCallback(
                    boost::bind(
                        &SurfaceSegAndTrackingNode::callback, this, _1, _2)
                    );

        // Set up a dynamic reconfigure server.
        // This should be done before reading parameter server values.
        {
            dynamic_reconfigure::Server<ni_depth_segmentation::NodeKVPConfig>::CallbackType cb;
            cb = boost::bind(&SurfaceSegAndTrackingNode::configCallback, this, _1, _2);
            dr_srv_.setCallback(cb);
        }

        initLayers(nh);

        // publishers
        img_pub_bgr_ = it_.advertise(name_out_+"_color", 1);
        img_pub_ = it_.advertise(name_out_, 1);
    }

protected:

    void initLayers(ros::NodeHandle &nh)
    {
        { // 0
            // Instantiate DepthMap layer
            LayerConfig cfg;
            LayerIONames io;
            io.Input(DepthMap::KEY_INPUT_STIMULUS, name_in_cld_);
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
            p.put(DepthGradientSmoothing::PARAM_SMOOTH_MODE, 2 );
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
        {
            // 4
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
            io.Output(MapAreaFilter::KEY_OUTPUT_RESPONSE, "map_gray_");
            layers_.push_back(LayerFactoryNI::CreateShared("MapAreaFilter", cfg, io));
        }
        {
            LayerConfig cfg;

            PTree p;
            p.put(SurfaceTracking::PARAM_HIST_BINS,     8);
            p.put(SurfaceTracking::PARAM_WEIGHT_COLOR,  0.4f);
            p.put(SurfaceTracking::PARAM_WEIGHT_POS,    0.1f);
            p.put(SurfaceTracking::PARAM_WEIGHT_SIZE,   0.5f);
            p.put(SurfaceTracking::PARAM_MAX_COLOR,     0.3f);
            p.put(SurfaceTracking::PARAM_MAX_POS,       0.15f);
            p.put(SurfaceTracking::PARAM_MAX_SIZE,      0.3f);
            p.put(SurfaceTracking::PARAM_MAX_DIST,      1.6f);

            cfg.Params(p);

            LayerIONames io;
            io.Input(SurfaceTracking::KEY_INPUT_BGR_IMAGE, name_in_img_);
            io.Input(SurfaceTracking::KEY_INPUT_CLOUD, name_in_cld_);
            io.Input(SurfaceTracking::KEY_INPUT_MAP, "map_gray_");
            io.Output(SurfaceTracking::KEY_OUTPUT_RESPONSE, name_out_);
            layers_.push_back(LayerFactoryNI::CreateShared("SurfaceTracking", cfg, io));
        }
    }

    void callback(const CloudXYZ::ConstPtr& cld,
                  const sensor_msgs::ImageConstPtr& img)
    {
        namespace img_enc=sensor_msgs::image_encodings;

        // Solve all of perception here...
        mtx_.lock ();
        {
            struct timespec t_total_start, t_total_end;
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_start);

            cloud_.reset(new CloudXYZ(*cld)); ///< @todo avoid copy

            try {
                cv_bridge::CvImageConstPtr cv_img_ptr;
                cv_img_ptr = cv_bridge::toCvShare(img, img_enc::BGR8);

                ni::normalizeColors(cv_img_ptr->image, img_normalized_colors_);
            }
            catch (cv_bridge::Exception& e) {

                ROS_ERROR("cv_bridge exception for img message: %s", e.what());
                return;
            }

            // External inputs from messages extracted...
            // Now to processing...

            sig_.Clear();
            sig_.Append(name_in_cld_, cloud_);
            sig_.Append(name_in_img_, static_cast<Mat1f>(img_normalized_colors_));

            for(size_t i=0; i<layers_.size(); i++) {

                layers_[i]->Activate(sig_);
                layers_[i]->Response(sig_);
            }

            Mat1f img = sig_.MostRecentMat1f(name_out_);
            //img(0) = 0.f;
            //img(1) = 20.f;

            Mat mask_not_assigned = img <= 0.f;

            Mat img_color;
            Mat1b img_gray = img;//ConvertTo8U(img);
            applyColorMap(img_gray,
                          img_color,
                          COLORMAP_HSV);

            img_color.setTo(Scalar(0), mask_not_assigned);

            //imshow("img_color", img_color);
            //waitKey(1);

            // convert to publish map image
            // mimic timestamp of processed point cloud
            std_msgs::Header header;
            header.stamp = ros::Time().fromNSec(cld->header.stamp*1e3);

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

            img_pub_.publish(img_msg_gray);

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

    typedef sensor_msgs::Image msg_Img;
    typedef message_filters::sync_policies::ApproximateTime<CloudXYZ, msg_Img> MySyncPolicy;

#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
    typedef image_transport::SubscriberFilter ImageSubscriber;
#else // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
    typedef message_filters::Subscriber< sensor_msgs::Image > ImageSubscriber;
#endif // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER

    // members
    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers

    std::string name_in_cld_;    ///< Signal name and subscribing topic name
    std::string name_in_img_;    ///< Signal name and subscribing topic name
    std::string name_out_;       ///< Signal name and publishing topic name

    message_filters::Synchronizer<MySyncPolicy> *sync_ptr_;

    ImageSubscriber img_sub_; ///< synchronized image subscriber (RGB)
    message_filters::Subscriber<CloudXYZ > cloud_sub_;         ///< synchronized point cloud subscriber

    image_transport::Publisher img_pub_bgr_;    ///< surface map image publisher
    image_transport::Publisher img_pub_;        ///< surface map image publisher

    dynamic_reconfigure::Server<ni_depth_segmentation::NodeKVPConfig> dr_srv_;

    boost::mutex mtx_;                      ///< mutex object for thread safety

    CloudXYZPtr cloud_;  ///< most recent point cloud

    cv::Mat3f img_normalized_colors_;

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
    ros::init(argc, argv, "surface_segmentation_and_tracking");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh("~");
    ni::SurfaceSegAndTrackingNode surface_tracking_node(nh);

    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();

    return 0;
}

