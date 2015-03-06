#include "gbsegmentation/segment-image.h"
#include "gbsegmentation/pnmfile.h"


/* Convert from opencv image to image format for gb seg
 */
image<rgb>* convertIplToNativeImage (cv::Mat cvm_input)
{
    int width = cvm_input.cols; int height = cvm_input.rows;
    image<rgb> *im = new image<rgb>(width,height);
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            cv::Vec3b s = cvm_input.at<cv::Vec3b>(j,i);
            rgb curr;
            curr.r = s.val[0];
            curr.g = s.val[1];
            curr.b = s.val[2];
            im->data[i + j*width] = curr;
        }
    }
    return im;
}


/* Call the graph-based segmentation function
 *
 * Input:
 * cv_rgb - input image
 * sigma, gsegk, minblob - paramter for gb seg
 *
 * output - set of pixels for segments
 */
void GbSegmentation (cv::Mat cv_rgb, double sigma, int gsegk, int minblob, std::vector<std::vector<CvPoint> > &output)
{
    image<rgb> *converted = convertIplToNativeImage(cv_rgb);
    segment_image(converted, sigma, gsegk, minblob, NULL, output);
    delete converted;
}
