ros node for graph-based segmentation


running the node: 
    -> rosrun gbseg_node gb_segmentation_node

    (remember to launch the camera)

running the node with specified parameters:
    -> rosrun gbseg_node gb_segmentation_node -GSegmSigma 1.0 -GSegmGrThrs 1000 -GSegmMinSize 100 -show 1
    
    replace the values with your parameters
    you can specify all parameters, only a subset or none 
    parameters which are not specified are set to default values

    GSegmSigma: smooth image with Gaussian filter 
    GSegmGrThrs: constant for threshold function 
    GSegmMinSize: minimum component size (enforced in post processing stage)
    show: if true (1), the resulting segmentation is shown in a window,
          if false (0), the result is not displayed
------------------------------------------------------------------------------

resetting the parameters:
    -> rostopic pub -1 /params_GbSeg gbseg_node/params_GbSeg '{GSegmSigma: 1.0, GSegmGrThrs: 1000, GSegmMinSize: 100, show: 1}'

    replace the values with your parameters


the code of the node params_GbSeg_publisher shows how the parameters can be reset from within a source code file
    -> rosrun gbseg_node params_GbSeg_publisher

------------------------------------------------------------------------------

gb_segmentation_node publishes the results in two different formats:

    the messages published to the topic "result_GbSeg" are of type gbseg_node::segmentation 

    a 'segmentation' message contains a vector of 'segment' messages, called 'segments'
    a 'segment' message contains a vector of 'point' messages, called 'points'
    a 'point' message has two fields, 'x' and 'y', WHERE X CORRESPONDS TO COLUMNS AND Y CORRESPONDS TO ROWS

    the messages published to the topic "resultImage_GbSeg" are of type sensor_msgs::ImagePtr 
    and can be converted into cv::Mat using the appropriate functions from the cv_bridge module

    the values of the image pixels are integers, starting from 0, indicating which segment the pixel belongs to
    (in order to display the image using the cv::imshow function, those values need to be transformed to the range [0,1] (grayscale)
    or replaced with rgb values)

the code of the node result_GbSeg_subscriber shows how the results published by gb_segmentation_node can be used in a source code file
    -> rosrun gbseg_node result_GbSeg_subscriber




