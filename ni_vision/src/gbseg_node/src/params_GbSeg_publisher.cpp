#include "ros/ros.h"
#include "gbseg_node/params_GbSeg.h"

// this code shows how the parameters of the gb_segmentation_node
// can be set from within a source code file

int main(int argc, char **argv)
{
    ros::init(argc, argv, "params_GbSeg_publisher");
    ros::NodeHandle nh;
    ros::Publisher params_pub = nh.advertise<gbseg_node::params_GbSeg>("params_GbSeg", 1000, true);


    gbseg_node::params_GbSeg msg;
    msg.GSegmSigma = 2.0;
    msg.GSegmGrThrs = 500;
    msg.GSegmMinSize = 500;
    msg.show = true;


    std::cout << "publishing ... ";
    params_pub.publish(msg);
    std::cout << "... done" << std::endl;
    ros::spin();

    return 0;
}
