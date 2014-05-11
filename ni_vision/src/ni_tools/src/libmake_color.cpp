/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, NI Group, TU-Berlin.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 * * $Id: makelib.cpp 2013-11-30 00:00:00 Fritjof Wolf and JongHan Park
 *
 */

/* This program computes the color histogram for a specified image. It provides a GUI, so that the user can
 * decide which part of the image should be used for the histogram computation. Then it computes the histogramm,
 * stores it and shows the extracted image.
 */

#include <iostream>
#include <dirent.h>

// ROS core
#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>

#include "opencv2/imgproc/imgproc_c.h"


#include "siftfast/siftfast.h"

#include "terminal_tools/parse.h"



// Global parameters

int threshold_slider = 40;       // Slider for the threshold of the binary image
int diameter_slider = 10;        // Slider for the diameter of the line which can be drawn in the binary image
IplImage* original_image = NULL;
IplImage* binary_image = NULL;
CvRect box = cvRect(-1,-1,0,0);
bool drawing_box = false;        // shows if the user currently draws a rectangle in the original image
bool drawing_white = false;      // shows if the user currently draws white in the binary image
bool drawing_black = false;      // shows if the user currently draws black in the binary image
int startbox_x;
int startbox_y;

std::string sPath;
std::string sInputPath;
std::string sOutputPath;
std::string sOutputFileName;

int nFocusX = 0, nFocusY = 0, nFocusWidth = 0, nFocusHeight = 0;
int nHistBinNr = 0;

int cCvCurrentkey = -1;

Keypoint keypts;
cv::Mat histo;




/* This method unpacks the color channels from a float number into three unsigned chars
 *
 * Input:
 * rgb - color rgb value of a pixel as a 32 bit float
 *
 * Output (per reference):
 * r,g,b - single R,G,B channels as unsigned 8 bit int
 */
void unpack_rgb(float rgb, uint8_t& r, uint8_t& g, uint8_t& b) {

        uint32_t rgbval;
        memcpy(&rgbval, &rgb, sizeof(float));

        b = (uint8_t)((rgbval >> 16) & 0x000000ff);
        g = (uint8_t)((rgbval >> 8) & 0x000000ff);
        r = (uint8_t)((rgbval) & 0x000000ff);

}




/* Define our callback which we will install for mouse events.
 */
void my_mouse_callback_binary(
int event, int x, int y, int flags, void* param
);
void my_mouse_callback_original(
int event, int x, int y, int flags, void* param
);



/* Draws the bounding box that the user draws in the image.
 *
 * Input:
 * img - image in which the rectangle is drawn
 */
void draw_box( IplImage* img) {

    cvRectangle (img,
    cvPoint(box.x,box.y),
    cvPoint(box.x+box.width,box.y+box.height),
    cvScalar(0xff,0x00,0x00)
    /* blue */
    );

}




/* Method that updates the image after the user changed the threshold for the binary image
 *
 * Input:
 * threshold - integer which represents the threshold used to compute the binary image
 */
void changeImage(int threshold) {

    IplImage *cv_gray = NULL, *cv_bold = NULL, *output = NULL;
    IplImage* image = cvCloneImage(original_image);

    cv_gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
    cv_bold = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
    output = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
    cvZero(cv_gray);
    cvZero(output);
    cvZero(cv_bold);

    cvCvtColor(image, cv_gray, CV_BGR2GRAY);

    cvSetImageROI(cv_gray, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    cvSetImageROI(output, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));

    // computes the new binary image with the new threshold
    cvThreshold(cv_gray, output, threshold, 255, CV_THRESH_BINARY );

    binary_image = cvCloneImage(output);
    cvShowImage("Processed", output);
    cvReleaseImage(&cv_gray);
    cvReleaseImage(&cv_bold);
    cvReleaseImage(&output);

}




/* Initialize the parameters of the program with the input from the console. */
void parameter_init(int argc, char** argv) {

    terminal_tools::parse_argument (argc, argv, "-dir", sPath);
    terminal_tools::parse_argument (argc, argv, "-obj", sOutputFileName);

    terminal_tools::parse_argument (argc, argv, "-bin", nHistBinNr);
    if(nHistBinNr == 0) nHistBinNr = 8;


    printf ("\n\n");
    printf ("===============================================================\n");
    printf ("** Parameter setting **\n");
    printf ("===============================================================\n");
    printf ("Feature Mode: .............................. Normalized RGB Color Histogram\n");
    printf ("===============================================================\n\n\n");

}




/* Calculates the histogram of an object using the normalized RGB space. For every pixel that is in the index vector it
 * extracts the R,G,B channel and normalizes these channels with respect to their sum. Then every normalized value r,g,b
 * is assigned into one of the bins.
 *
 * Input:
 * cvm_input - image from which the color histogram should be computed
 * index - vector of ints, which can vary in length; it contains the indices of the pixel which are used to compute the
 *         color histogram
 * bin_base - int which represents the number of bins in the model (default 8)
 *
 * Output (by reference):
 * vnOut - vector of length bin_base³, which contains the normalized histogram
 */
void Calc3DColorHistogram(const cv::Mat& cvm_input, const std::vector<int>& index, int bin_base, std::vector<float> &vnOut) {

    if (index.size()){
        int bin_r, bin_g, bin_b;
        float r, g, b;

        float bin_width;

        // max has to be slightly greater than 1, since otherwise there is a problem if R, G or B is equal to 255
        float max = 1.0001, sum;
        bin_width = (float)max/bin_base;


        uint8_t R_, G_, B_;
        for(size_t i = 0; i < index.size(); i++) {

            B_ = cvm_input.data[index[i]*3];
            G_ = cvm_input.data[index[i]*3+1];
            R_ = cvm_input.data[index[i]*3+2];

            sum = (int)R_ + (int)G_ + (int)B_;


            if (sum) {r = (float)R_/sum; g = (float)G_/sum; b = (float)B_/sum;}
            else {r = 1/3; g = 1/3; b = 1/3;}
            bin_width = (float)max/bin_base;
            bin_r = (int)(r/bin_width);
            bin_g = (int)(g/bin_width);
            bin_b = (int)(b/bin_width);

            // The specific combination of the bins for the different channels is incremented in the histogram
            vnOut[bin_r*bin_base*bin_base + bin_g*bin_base + bin_b]++;
        }

        // Normalization, so that the sum over all entries of the histogram is one
        for (int i = 0; i < bin_base*bin_base*bin_base; i++) vnOut[i] = vnOut[i]/index.size();
    }

}




/* Extracts the region of the image where the object is located and returns the width, height and
 * the upper left corner of the bounding box around that image.
 *
 * Input:
 * input - original image
 *
 * Output (by reference):
 * x_min, y_min - ints, coordinates of the upper left corner of the bounding box
 * width, height - ints, width and height of the bounding box
 */
void ExtractObject(IplImage* input, int &x_min, int &y_min, int &width, int &height) {

    IplImage *cv_gray = NULL, *cv_bold = NULL;
    cv_gray = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
    cv_bold = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
    cvZero(cv_gray);
    cvZero(cv_bold);

    cvCvtColor(input, cv_gray, CV_BGR2GRAY);

    cvSetImageROI(cv_gray, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    cvSetImageROI(binary_image, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));

    cvResetImageROI(binary_image);
    cvResetImageROI(cv_gray);


    cvSetImageROI(binary_image, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    cvSetImageROI(cv_bold, cvRect(nFocusX-1, nFocusY-1, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX, nFocusY-1, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX+1, nFocusY-1, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX-1, nFocusY, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX+1, nFocusY, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX-1, nFocusY+1, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX, nFocusY+1, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvSetImageROI(cv_bold, cvRect(nFocusX+1, nFocusY+1, nFocusWidth, nFocusHeight));
    cvAdd(cv_bold, binary_image, cv_bold, NULL);
    cvResetImageROI(cv_bold);
    cvResetImageROI(binary_image);

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = NULL;
    cvFindContours(cv_bold, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

    int rx1 = 2000, ry1 = 2000, rx2 = 0, ry2 = 0;
    CvRect rect;
    for (; contours != 0; contours = contours->h_next) {
        rect = cvBoundingRect(contours, 1);
        int area = rect.width*rect.height;
        if (area < 100)
            continue;

        int x2 = rect.x + rect.width - 1, y2 = rect.y + rect.height - 1;

        if (rect.x < rx1) rx1 = rect.x;
        if (rect.y < ry1) ry1 = rect.y;
        if (x2 > rx2) rx2 = x2;
        if (y2 > ry2) ry2 = y2;
    }
    cvReleaseImage(&cv_gray);
    cvReleaseImage(&cv_bold);

    x_min = rx1; y_min = ry1; width = rx2 - rx1 + 1; height = ry2 - ry1 + 1;

}




/* Here the information of the binary image, i.e. which pixel are important for the computation of
 * the histogram, is extracted and shown. With that information the histogram is computed.
 *
 * Input:
 * input - original image
 * cv_binary - binary image, which contains the information about the pixel which should be used for histogram computation
 * x_min, y_min - coordinates of the upper left corner of the bounding box
 * width, height - width and height of the bounding box
*/
void MakeHistogram (IplImage *input, IplImage *cv_binary, int x_min, int y_min, int width, int height) {

    // Maximal number of entries in the histogram (corresponds to all possible combination of bins of the three color channels)
    int histVal = nHistBinNr*nHistBinNr*nHistBinNr;
    std::vector<float> vRgbHist(histVal,0);

    int cnt = 0;
    int size = width * height;

    // Stores the indices of the pixel for the histogram computation
    std::vector<int> index(size, 0);
    cv::Mat rgb = cv::Mat::zeros(size, 3, CV_8U);
    // stores the computed histogram
    histo = cv::Mat::zeros(1, histVal, CV_32F);
    CvScalar s;

    IplImage *cv_input_tmp = NULL, *cv_binary_tmp = NULL;
    IplImage *cv_extract = NULL;

    cv_input_tmp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
    cv_binary_tmp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    cv_extract =  cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
    cvZero(cv_input_tmp);
    cvZero(cv_binary_tmp);

    cvSetImageROI(input, cvRect(x_min, y_min, width, height));
    cvAdd(cv_input_tmp, input, cv_input_tmp,NULL);
    cvResetImageROI(input);
    cvSetImageROI(cv_binary, cvRect(x_min, y_min, width, height));
    cvAdd(cv_binary_tmp, cv_binary, cv_binary_tmp,NULL);
    cvResetImageROI(cv_binary);

    cvNamedWindow("Extracted", CV_WINDOW_AUTOSIZE); cvMoveWindow("Extracted", 600, 800);
    cvShowImage("Extracted", cv_input_tmp);


    // Creating the index set of all pixel which are white in the binary image
    // i.e. extract all the image information from the original imagewhich are
    // used to compute the histogramm.

    int i = 0;

    for(int y = 0; y < height; y++) {
        uchar* ptr = (uchar*) (cv_input_tmp->imageData + y * cv_input_tmp->widthStep);
        uchar* ptr2 = (uchar*) (cv_binary_tmp->imageData + y * cv_binary_tmp->widthStep);
        for(int x = 0; x < width; x++) {
            i++;
            if(ptr2[x] == 255) {
                rgb.at<unsigned char>(i,0) = ptr[3*x];
                rgb.at<unsigned char>(i,1) = ptr[3*x+1];
                rgb.at<unsigned char>(i,2) = ptr[3*x+2];
                index[cnt++] = i;
            }
        }
    }
    index.resize(cnt);


    // Create the extracted image, i.e. the image in which all pixel which contribute
    // to the histogram are colored and the rest is black.
    cv_extract = cvCloneImage(cv_input_tmp);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            s = cvGet2D(cv_binary_tmp,i,j);
            if(s.val[0] == 0) {
                s.val[0] = 0;
                s.val[1] = 0;
                s.val[2] = 0;
                cvSet2D(cv_extract,i,j,s);
            }
        }
    }

    cvShowImage("Processed", cv_binary_tmp);
    cvShowImage("Extracted", cv_extract);

    cvReleaseImage(&cv_extract);
    cvReleaseImage(&cv_input_tmp);
    cvReleaseImage(&cv_binary_tmp);


    // Computing the color histogram
    Calc3DColorHistogram(rgb, index, nHistBinNr, vRgbHist);

    for(int j = 0; j < histVal; j++) {
        histo.at<float>(0,j) = vRgbHist[j];
    }


}




/* Stores the computed histogram in a yaml-file.
*/
void Save2Disk()
{

    std::string sLibFile;

    sLibFile = sOutputPath + "simplelib_3dch_" + sOutputFileName + ".yaml";
    cv::FileStorage fs(sLibFile,cv::FileStorage::WRITE);
    fs << "TestObjectFeatureVectors" << histo;

}




/* This is the main method of the program. It reads the image from the specified path, display
 * the original and the binary image and starts the computation of the color histogram.
*/
void ScanDir() {

    DIR *dirp;
    struct dirent *dp;
    const char *dirchar = sInputPath.c_str();
    dirp = opendir(dirchar);
    std::cout << "\nDirectory Opened\n";

    std::cout << "Opening File : " << sInputPath << "\n\n";

    while (dirp) {
        if ((dp = readdir(dirp)) != NULL) {
            std::string sFile = dp->d_name;
            size_t pos = sFile.find(".");
            std::string sExt = sFile.substr(pos+1);

            // The image has to have one of the allowed formats: bmp, jpeg, jpg, tiff, tif, png
            if(sExt.compare("bmp")==0 || sExt.compare("jpeg")==0 || sExt.compare("jpg")==0 || sExt.compare("tiff")==0 || sExt.compare("tif")==0 || sExt.compare("png")==0) {
                std::string sFileName = sInputPath + sFile;
                std::cout << "Opening File : " << sFileName << "\n\n";
                const char *c = sFileName.c_str();
                IplImage *cv_input = cvLoadImage(c);

                original_image = cvCloneImage(cv_input);


                // The default size of the binary image
                nFocusX = 0;
                nFocusY = 0;
                nFocusWidth = (original_image->width);
                nFocusHeight = (original_image->height);


                // Display the binary image
                IplImage *cv_gray = NULL, *cv_bold = NULL, *output = NULL;
                IplImage* image = cvCloneImage(original_image);

                cv_gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
                cv_bold = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
                output = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
                cvZero(cv_gray);
                cvZero(output);
                cvZero(cv_bold);

                cvCvtColor(image, cv_gray, CV_BGR2GRAY);

                cvSetImageROI(cv_gray, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
                cvSetImageROI(output, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));

                // The binary image is computed from the grey-value image and the default threshold of 40
                cvThreshold(cv_gray, output, threshold_slider, 255, CV_THRESH_BINARY );


                binary_image = cvCloneImage(output);
                cvReleaseImage(&cv_gray);
                cvReleaseImage(&cv_bold);



                // Define Mousehandlers and trackbars
                cvSetMouseCallback("Current Image", my_mouse_callback_original,original_image);
                cvSetMouseCallback("Processed", my_mouse_callback_binary,output);

                cvCreateTrackbar( "Threshold", "Current Image", &threshold_slider, 255, changeImage);
                cvCreateTrackbar( "Line diameter", "Processed", &diameter_slider, 30, NULL);

                cvShowImage("Processed", output);
                cvShowImage("Current Image", original_image);

                std::cout << "Press a key to continue" << std::endl;
                cCvCurrentkey = cv::waitKey(0);

                cvDestroyWindow("Processed");
                cvNamedWindow("Processed", CV_WINDOW_AUTOSIZE); cvMoveWindow("Processed", 400, 200);


                if(!cv_input) {
                    std::cout << "Can not open the file : " << sFileName << std::endl;
                    break;
                }
                else {
                    cv::Size s;
                    int rows;
                    sOutputPath = sPath;
                    std::string sLibFileMode;
                    int x_min, y_min, width, height;
                    IplImage *cv_binary = cvCreateImage(cvGetSize(cv_input), IPL_DEPTH_8U, 1);

                    // Extract the coordinates of the bounding box around the object
                    ExtractObject(cv_input, x_min, y_min, width, height);

                    // Computing the color histogram
                    std::cout << "....Computing color histogram\n";
                    MakeHistogram(cv_input, binary_image, x_min, y_min, width, height);
                    cCvCurrentkey = cv::waitKey();
                    s = histo.size();
                    rows = s.height;

                    if (rows) {Save2Disk(); std::cout << "Library File Saved!\n\n";}
                    else std::cout << "Library file could not be saved. No feature was extracted.\n\n";
                    break;

                    cvReleaseImage(&cv_binary);
                    cvReleaseImage(&original_image);
                    cvReleaseImage(&binary_image);
                    cvReleaseImage(&output);
                }
            }
        }
        else {
            closedir(dirp);
            return;
        }
    }

}




/* This is the mouse handler for mouse events in the rgb image. If the user
   presses the left button, we start a box and add it
   to a copy of the current image. When the
   mouse is dragged (with the button down) we
   resize the box and update the binary image if the mouse button is released.

   Input:
   event - information about the mouse event that occured
   x,y - coordinates of the mouse event
*/
void my_mouse_callback_original(int event, int x, int y, int flags, void* param) {

    switch( event ) {
        // Change the size of the box
        case CV_EVENT_MOUSEMOVE: {
            if( drawing_box ) {
                if(x - startbox_x >= 0) {
                    box.width = x - startbox_x;
                }
                if(y - startbox_y >= 0) {
                    box.height = y-box.y;
                }
                if(x - startbox_x < 0) {
                    box.x = x;
                    box.width  = startbox_x - x;
                }
                if(y - startbox_y < 0) {
                    box.y = y;
                    box.height  = startbox_y - y;
                }


                nFocusX = box.x;
                nFocusY = box.y;
                nFocusWidth = box.width;
                nFocusHeight = box.height;


                // Creates a copy of the original image, because you can't remove a rectangle once it was drawn
                IplImage* image = cvCloneImage(original_image);

                draw_box(image);
                cvShowImage("Current Image", image);
            }
            break;
        }

        // Start a new box
        case CV_EVENT_LBUTTONDOWN: {
            drawing_box = true;
            box = cvRect(x, y, 0, 0);
            startbox_x = x;
            startbox_y = y;
            break;
        }

        // Update the extracted binary image
        case CV_EVENT_LBUTTONUP: {
            drawing_box = false;

            IplImage *cv_gray = NULL, *cv_bold = NULL, *output = NULL;
            IplImage* image = cvCloneImage(original_image);

            cv_gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
            cv_bold = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
            output = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
            cvZero(cv_gray);
            cvZero(output);
            cvZero(cv_bold);

            cvCvtColor(image, cv_gray, CV_BGR2GRAY);

            cvSetImageROI(cv_gray, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
            cvSetImageROI(output, cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
            cvThreshold(cv_gray, output, threshold_slider, 255, CV_THRESH_BINARY);

            binary_image = cvCloneImage(output);
            cvShowImage("Processed", binary_image);

            cvReleaseImage(&cv_gray);
            cvReleaseImage(&cv_bold);
            cvReleaseImage(&output);
            break;
        }
    }

}




/* This is the mouse handler for mouse events  in the binary image.
   It allows the user to change some regions from
   black to white an vice versa by drawing with the pressed left (resp. right)
   mouse button. The diameter of the drawn line is stored in the variable
   diameter_slider and can be changed via the trackbar at the top of the window.

   Input:
   event - information about the mouse event that occured
   x,y - coordinates of the mouse event
*/
void my_mouse_callback_binary(int event, int x, int y, int flags, void* param) {

    switch( event ) {
        case CV_EVENT_MOUSEMOVE: {
            CvScalar s;

            // Left mouse button is pressed
            if(drawing_white) {
                for(int i = -diameter_slider; i < diameter_slider; i++) {
                    for(int j = -diameter_slider; j < diameter_slider; j++) {
                        if(x+j < 0 || y+i < 0 || x+j >= nFocusWidth || y+i >= nFocusHeight) continue;

                        s = cvGet2D(binary_image,y+i,x+j);
                        s.val[0] = 255;
                        s.val[1] = 255;
                        s.val[2] = 255;
                        cvSet2D(binary_image,y+i,x+j,s);
                    }
                }

                cvShowImage("Processed", binary_image);
            }

            // Right mouse button is pressed
            if(drawing_black) {
                for(int i = -diameter_slider; i < diameter_slider; i++) {
                    for(int j = -diameter_slider; j < diameter_slider; j++) {

                      if(x+j < 0 || y+i < 0 || x+j >= nFocusWidth || y+i >= nFocusHeight) continue;

                        s = cvGet2D(binary_image,y+i,x+j);
                        s.val[0] = 0;
                        s.val[1] = 0;
                        s.val[2] = 0;
                        cvSet2D(binary_image,y+i,x+j,s);
                    }
                }

                cvShowImage("Processed", binary_image);
            }
            break;
        }

        case CV_EVENT_LBUTTONDOWN: {
            drawing_white = true;
            break;
        }

        case CV_EVENT_LBUTTONUP: {
            drawing_white = false;
            break;
        }

        case CV_EVENT_RBUTTONDOWN: {
            drawing_black = true;
            break;
        }

        case CV_EVENT_RBUTTONUP: {
            drawing_black = false;
            break;
        }

    }

}




int main (int argc, char** argv)
{

    parameter_init(argc, argv);
    if(argc<2){
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }
    printf ("Scanning Directory...\n");

    sInputPath = sPath + sOutputFileName + "/OnePos/";
    sOutputPath = sPath;


    cvNamedWindow("Current Image", CV_WINDOW_AUTOSIZE); cvMoveWindow("Current Image", 100, 100);
    cvNamedWindow("Processed", CV_WINDOW_AUTOSIZE); cvMoveWindow("Processed", 400, 200);

    ScanDir();

    cvDestroyAllWindows();
    return (0);

}
