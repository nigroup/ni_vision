// %Tag(FULLTEXT)%
#include <ros/ros.h>

#include <set>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/graph/graphattr.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"            // Signal class: Stimulus -> layer activation -> response
#include "elm/core/pcl/typedefs_fwd.h"  // point cloud typedef
#include "elm/core/featuredata.h"

#include "ni/core/color_utils.h"
#include "ni/core/surface.h"
#include "ni/layers/attention.h"
#include "ni/layers/recognition.h"
#include "ni/layers/layerfactoryni.h"
#include "ni/legacy/timer.h"

#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float32MultiArray.h"

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
 * @brief Node for Top-Down Attention and Object Recognition
 */
class AttentionAndRecognitionNode
{
public:
    ~AttentionAndRecognitionNode()
    {
        if(sync_ptr_ != NULL) {

            delete sync_ptr_;
        }
    }

    /**
     * @brief Constructor
     *
     * Register subscriptions.
     *
     * @param nh node handle reference
     * @todo determine ideal queue size for subscribers and sync policy
     */
    AttentionAndRecognitionNode(ros::NodeHandle &nh)
        : it_(nh),
          name_in_cld_("/camera/depth_registered/points"),
          name_in_img_("/camera/rgb/image_color"),
          name_in_seg_("/ni/depth_segmentation/surfaces/image"),
          name_out_rect_("/ni/depth_segmentation/recognition/rect"),
          name_out_histogram_("/ni/depth_segmentation/recognition/hist"),
          name_out_matchFlag_("/ni/depth_segmentation/recognition/found"),
          name_out_keypoints_("/ni/depth_segmentation/recognition/keypoints"),
          name_out_matchedKeypoints_("/ni/depth_segmentation/recognition/matchedKeypoints"),
          name_out_examinedIndex_("/ni/depth_segmentation/recognition/examinedIndex"),
#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
          img_sub_(it_, name_in_img_, 1),
          img_sub_seg_(it_, name_in_seg_, 1),
#else // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
          img_sub_(nh, name_in_img_, 1),
          img_sub_seg_(nh, name_in_seg_, 1),
#endif // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
          cloud_sub_(nh, name_in_cld_, 1)
    {

        using namespace message_filters; // Subscriber, sync_policies

        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        int queue_size = 60;
        sync_ptr_ = new Synchronizer<MySyncPolicy>(MySyncPolicy(queue_size),
                                                   cloud_sub_,
                                                   img_sub_,
                                                   img_sub_seg_);

        sync_ptr_->registerCallback(
                    boost::bind(
                        &AttentionAndRecognitionNode::callback, this, _1, _2, _3)
                    );

        initLayers(nh);
        recog_pub_matchFlag_ = nh.advertise<std_msgs::Bool>(name_out_matchFlag_, 1);
        recog_pub_rect_ = nh.advertise<std_msgs::Float32MultiArray>(name_out_rect_, 1);
        recog_pub_keypoints_ = nh.advertise<std_msgs::Float32MultiArray>(name_out_keypoints_, 1);
        recog_pub_matchedKeypoints_ = nh.advertise<std_msgs::Float32MultiArray>(name_out_matchedKeypoints_, 1);
        recog_pub_examinedIndex_ = nh.advertise<std_msgs::Float32>(name_out_examinedIndex_, 1);
    }

protected:
    void initLayers(ros::NodeHandle &nh)
    {

        boost::filesystem::path path_color("/home/fritjof/Video/models/Lib8B/simplelib_3dch_DanKlorix.yaml");
        boost::filesystem::path path_sift("/home/fritjof/Video/models/Lib8B/lib_sift_DanKlorix_0015.yaml");

        { // 0
            // Instantiate top-down attention
            LayerConfig cfg;

            PTree p;
            p.put(Attention::PARAM_HIST_BINS,   8);
            p.put(Attention::PARAM_SIZE_MAX,  400);
            p.put(Attention::PARAM_SIZE_MIN,  100);
            p.put(Attention::PARAM_PTS_MIN,   200);

            std::string tmp;
//            nh.getParam(Attention::PARAM_PATH_COLOR, tmp);
//            boost::filesystem::path path_color(tmp);

//            nh.getParam(Attention::PARAM_PATH_SIFT, tmp);
//            boost::filesystem::path path_sift(tmp);

            p.put<boost::filesystem::path>(Attention::PARAM_PATH_COLOR, path_color);
            p.put<boost::filesystem::path>(Attention::PARAM_PATH_SIFT,  path_sift);

            cfg.Params(p);

            LayerIONames io;
            io.Input(Attention::KEY_INPUT_BGR_IMAGE, name_in_img_);
            io.Input(Attention::KEY_INPUT_CLOUD, name_in_cld_);
            io.Input(Attention::KEY_INPUT_MAP, name_in_seg_);
            io.Output(Attention::KEY_OUTPUT_HISTOGRAM, name_out_histogram_);
            io.Output(Attention::KEY_OUTPUT_RECT, name_out_rect_);
            io.Output(Attention::KEY_OUTPUT_INDEX, name_out_examinedIndex_);
            layers_.push_back(LayerFactoryNI::CreateShared("Attention", cfg, io));
        }
        { // 1
            // Instantiate top-down attention
            LayerConfig cfg;

            PTree p;
            std::string tmp;
//            nh.getParam(Attention::PARAM_PATH_COLOR, tmp);
//            boost::filesystem::path path_color(tmp);

//            nh.getParam(Attention::PARAM_PATH_SIFT, tmp);
//            boost::filesystem::path path_sift(tmp);

            p.put<boost::filesystem::path>(Recognition::PARAM_PATH_COLOR, path_color);
            p.put<boost::filesystem::path>(Recognition::PARAM_PATH_SIFT,  path_sift);

            cfg.Params(p);

            LayerIONames io;
            io.Input(Recognition::KEY_INPUT_BGR_IMAGE, name_in_img_);
            io.Input(Recognition::KEY_INPUT_HISTOGRAM, name_out_histogram_);
            io.Input(Recognition::KEY_INPUT_RECT, name_out_rect_);
            io.Output(Recognition::KEY_OUTPUT_MATCH_FLAG, name_out_matchFlag_);
            io.Output(Recognition::KEY_OUTPUT_KEYPOINTS, name_out_keypoints_);
            io.Output(Recognition::KEY_OUTPUT_MATCHED_KEYPOINTS, name_out_matchedKeypoints_);
            layers_.push_back(LayerFactoryNI::CreateShared("Recognition", cfg, io));
        }
    }

    void callback(const CloudXYZ::ConstPtr& cld,
                  const sensor_msgs::ImageConstPtr& img,
                  const sensor_msgs::ImageConstPtr& img_seg)
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
                img_normalized_colors_ = cv_img_ptr->image;
                //ni::normalizeColors(cv_img_ptr->image, img_normalized_colors_);
                //img_normalized_colors_.convertTo(img_normalized_colors_8bit_, CV_8UC3);
                //cv::imshow("x2", img_normalized_colors_8bit_);
            }
            catch (cv_bridge::Exception& e) {

                ROS_ERROR("cv_bridge exception for img message: %s", e.what());
                return;
            }

            try {
                cv_bridge::CvImageConstPtr cv_img_ptr;
                cv_img_ptr = cv_bridge::toCvShare(img_seg, img_enc::MONO8);
                cv_img_ptr->image.convertTo(img_seg_, CV_32FC1);
            }
            catch (cv_bridge::Exception& e) {

                ROS_ERROR("cv_bridge exception for img_seg message: %s", e.what());
                return;
            }

            // External inputs from messages extracted...
            // Now to processing...

            sig_.Clear();
            sig_.Append(name_in_cld_, cloud_);
            sig_.Append(name_in_img_, static_cast<Mat1f>(img_normalized_colors_));

            sig_.Append(name_in_seg_, img_seg_);

            for(size_t i=0; i<layers_.size(); i++) {

                layers_[i]->Activate(sig_);
                layers_[i]->Response(sig_);
            }

            Mat1f rect_ = sig_.MostRecentMat1f(name_out_rect_);
            bool matchFlag = sig_.MostRecent(name_out_matchFlag_).get<int>() > 0;
            Mat1f keypoints = sig_.MostRecentMat1f(name_out_keypoints_);
            Mat1f matchedKeypoints = sig_.MostRecentMat1f(name_out_matchedKeypoints_);
            float examinedIndex = sig_.MostRecent(name_out_examinedIndex_).get<float>();


            // mimic timestamp of processed data
            std_msgs::Header header;
            header.stamp = ros::Time().fromNSec(cld->header.stamp*1e3);

            std_msgs::Bool msg;
            msg.data = matchFlag;
            // todo add header
            recog_pub_matchFlag_.publish(msg);

            std_msgs::Float32MultiArray msg2;
            msg2.data = rect_;
            // todo add header
            recog_pub_rect_.publish(msg2);

            if(!keypoints.empty()) {

                std_msgs::Float32MultiArray msg3;
                msg3.data = keypoints.reshape(1, keypoints.total());
                recog_pub_keypoints_.publish(msg3);

                std_msgs::Float32MultiArray msg4;
                msg4.data = matchedKeypoints;
                recog_pub_matchedKeypoints_.publish(msg4);
            }
//            std::cout << rect_ << std::endl;
//            std::cout << matchFlag_ << std::endl;


            std_msgs::Float32 msg5;
            msg5.data = examinedIndex;
            recog_pub_examinedIndex_.publish(msg5);
            printf("Index in node is %f", msg5.data);


            clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_end);
            double nTimeTotal = double(timespecDiff(&t_total_end, &t_total_start)/1e9);

            ROS_INFO("running at %f Hz", 1./nTimeTotal);
        }
        mtx_.unlock (); // release mutex
    }

    typedef LayerFactoryNI::LayerShared LayerShared;
    typedef std::vector<LayerShared > VecLayerShared;

    typedef sensor_msgs::Image msg_Img;
    typedef message_filters::sync_policies::ApproximateTime<CloudXYZ, msg_Img, msg_Img> MySyncPolicy;

#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
    typedef image_transport::SubscriberFilter ImageSubscriber;
#else // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
    typedef message_filters::Subscriber< sensor_msgs::Image > ImageSubscriber;
#endif // USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER

    // members
    image_transport::ImageTransport it_;    ///< faciliatate image publishers and subscribers

    std::string name_in_cld_;    ///< Signal name and subscribing topic name
    std::string name_in_img_;    ///< Signal name and subscribing topic name
    std::string name_in_seg_;    ///< Signal name and subscribing topic name
    std::string name_out_rect_;
    std::string name_out_matchFlag_;
    std::string name_out_histogram_;
    std::string name_out_keypoints_;
    std::string name_out_matchedKeypoints_;
    std::string name_out_examinedIndex_;


    message_filters::Synchronizer<MySyncPolicy> *sync_ptr_;

    ImageSubscriber img_sub_; ///< synchronized image subscriber (RGB)
    ImageSubscriber img_sub_seg_; ///< synchronized image subscriber (Segmentation map)
    message_filters::Subscriber<CloudXYZ > cloud_sub_;         ///< synchronized point cloud subscriber

    ros::Publisher recog_pub_rect_;
    ros::Publisher recog_pub_matchFlag_;
    ros::Publisher recog_pub_keypoints_;
    ros::Publisher recog_pub_matchedKeypoints_;
    ros::Publisher recog_pub_examinedIndex_;

    boost::mutex mtx_;                      ///< mutex object for thread safety

    CloudXYZPtr cloud_;  ///< most recent point cloud

    cv::Mat1f img_seg_;
    cv::Mat3f img_normalized_colors_;
    cv::Mat img_normalized_colors_8bit_;

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
    ros::init(argc, argv, "attentionAndRecognition");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh("~");
    ni::AttentionAndRecognitionNode attentionAndRecognition_node(nh);

    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();

    return 0;
}

