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



#include <iostream>
#include <dirent.h>

// ROS core
#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>

#include "opencv2/imgproc/imgproc_c.h"


#include "siftfast/siftfast.h"

#include "terminal_tools/parse.h"


//#include "ni_vision/func_operations.h"


//// This method unpacks the color channels from a float number
void unpack_rgb(float rgb, uint8_t& r, uint8_t& g, uint8_t& b) {
        uint32_t rgbval;
        memcpy(&rgbval, &rgb, sizeof(float));

        //uint8_t garbage_ = (uint8_t)((rgb_val_ >> 24) & 0x000000ff);
        b = (uint8_t)((rgbval >> 16) & 0x000000ff);
        g = (uint8_t)((rgbval >> 8) & 0x000000ff);
        r = (uint8_t)((rgbval) & 0x000000ff);
}





// Global variables
int alpha_slider = 40;
int diameter_slider = 10;
int mode_slider = 0;
IplImage* original_image = NULL;
IplImage* binary_image = NULL;
CvRect box = cvRect(-1,-1,0,0);
bool drawing_box = false;
bool drawing_white = false;
bool drawing_black = false;
int startbox_x;
int startbox_y;
int a = 0;


// --------------------
// -----Parameters-----
// --------------------
std::string sPath;
std::string sInputPath;
std::string sOutputPath;
std::string sOutputFileName;
//////// for the 2D SIFT Keypoint detection ////////////////////
int nTotalKeypointCount=0;
int nSiftScales = 0;
double nSiftInitSigma = 0;
double nSiftPeakThresh = 0;

int nThresh = 0;
int nFocusX = 0, nFocusY = 0, nFocusWidth = 0, nFocusHeight = 0;
int nColorMode = 0;
int nHistBinNr = 0;

//Global Data
int timer = 0;
int cCvCurrentkey = -1;

Keypoint keypts;
cv::Mat histo;
// Define our callback which we will install for
// mouse events.
//
void my_mouse_callback_binary(
int event, int x, int y, int flags, void* param
);
void my_mouse_callback_original(
int event, int x, int y, int flags, void* param
);


void draw_box( IplImage* img, CvRect rect ) {
    cvRectangle (
    img,
    cvPoint(box.x,box.y),
    cvPoint(box.x+box.width,box.y+box.height),
    cvScalar(0xff,0x00,0x00)
    /* red */
    );
}

// Method updates the image each time the trackbar is moved
void changeImage(int value)
{
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
    cvThreshold(cv_gray, output, value, 255, CV_THRESH_BINARY );

    binary_image = cvCloneImage(output);
    cvShowImage("Processed", output);
    cvReleaseImage(&cv_gray);
    cvReleaseImage(&cv_bold);
    cvReleaseImage(&output);
}




// initialize parameters
void parameter_init(int argc, char** argv) {
    terminal_tools::parse_argument (argc, argv, "-dir", sPath);
    //terminal_tools::parse_argument (argc, argv, "-oppath", sOutputPath);
    terminal_tools::parse_argument (argc, argv, "-obj", sOutputFileName);
    terminal_tools::parse_argument (argc, argv, "-timer", timer);
    if(timer == 0) timer = 10;



    terminal_tools::parse_argument (argc, argv, "-bwthr", nThresh);
    if(nThresh == 0) nThresh = 30;
    terminal_tools::parse_argument (argc, argv, "-cmode", nColorMode);
    if(nColorMode == 0) nColorMode = 0;
    terminal_tools::parse_argument (argc, argv, "-bin", nHistBinNr);
    if(nHistBinNr == 0) nHistBinNr = 4;

    terminal_tools::parse_argument (argc, argv, "-siftsc", nSiftScales);
    if(nSiftScales == 0) nSiftScales = 3;
    terminal_tools::parse_argument (argc, argv, "-siftis", nSiftInitSigma);
    if(nSiftInitSigma == 0) nSiftInitSigma = 1.6;
    terminal_tools::parse_argument (argc, argv, "-siftpt", nSiftPeakThresh);
    if(nSiftPeakThresh == 0) nSiftPeakThresh = 0.04;


    printf ("\n\n");
    printf ("===============================================================\n");
    printf ("** Parameter setting **\n");
    printf ("===============================================================\n");
    switch (mode_slider) {
    case 1:
        printf ("Feature Mode: .............................. RGB Color Histogram\n");
        break;
    case 2:
        printf ("Feature Mode: .............................. SIFT\n");
        printf ("2D SIFT - Scales: .......................... %5d\n", nSiftScales);
        printf ("2D SIFT - Initial Sigma: ................... %10.4f\n", nSiftInitSigma);
        printf ("2D SIFT - Peak thresh: ..................... %10.4f\n", nSiftPeakThresh);
        break;
    case 0:
        printf ("Feature Mode: .............................. Hue Sift features\n");
        break;
    }
    printf ("===============================================================\n\n\n");
}



void Calc3DColorHistogram(cv::Mat cvm_input, std::vector<int> index, int bin_base, std::vector<float> &vnOut) {
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

            vnOut[bin_r*bin_base*bin_base + bin_g*bin_base + bin_b]++;


        }

        ///// normalizing //////////////
        for (int i = 0; i < bin_base*bin_base*bin_base; i++) vnOut[i] = vnOut[i]/index.size();
    }
}





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
    }//for
    cvReleaseImage(&cv_gray);
    cvReleaseImage(&cv_bold);

    x_min = rx1; y_min = ry1; width = rx2 - rx1 + 1; height = ry2 - ry1 + 1;
}


////////// libsiftfast Application //////////////////////////////////////////////////
void MakeSift(IplImage *input, int x, int y, int width, int height) {

    int res = 10;
    x = x - res; if (x < 0) x = 0;
    y = y - res; if (y < 0) y = 0;
    width = width + 2*res; if (x + width > input->width) width = input->width - x;
    height = height + 2*res; if (y + height > input->height) height = input->height - y;


    IplImage *cv_intensity = NULL, *cv_extract = NULL;
    cv_intensity = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    cv_extract = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
    cvZero(cv_intensity);
    cvZero(cv_extract);

    cvSetImageROI(input, cvRect(x, y, width, height));
    cvAdd(cv_extract, input, cv_extract);
    cvConvertImage(input, cv_intensity, 0);
    cvResetImageROI(input);



    Image img_sift = CreateImage(width, height);

    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            CvScalar s = cvGet2D(input, j + y, i + x);
            img_sift->pixels[i * img_sift->stride + j] = (s.val[0] + s.val[1] + s.val[2]) / (3.*255);
        }
    }

    SiftParameters dparm = GetSiftParameters();
    dparm.Scales = nSiftScales;
    dparm.InitSigma = nSiftInitSigma;
    dparm.PeakThresh = nSiftPeakThresh;
    SetSiftParameters(dparm);
    keypts = GetKeypoints(img_sift);


    std::vector<std::vector <float> > vTmpFeatureSet;
    int nCurrKeyCount=0;
    while(keypts) {
        int tmpsize = vTmpFeatureSet.size();
        vTmpFeatureSet.resize(tmpsize + 1, std::vector<float>(133,0));

        for(int i = 0; i < 128; i++)
            vTmpFeatureSet[tmpsize][i] = keypts->descrip[i];

        vTmpFeatureSet[tmpsize][128] = keypts->row;
        vTmpFeatureSet[tmpsize][129] = keypts->col;
        vTmpFeatureSet[tmpsize][130] = keypts->scale;
        vTmpFeatureSet[tmpsize][131] = keypts->ori;
        vTmpFeatureSet[tmpsize][132] = keypts->fpyramidscale;
        nCurrKeyCount++;

        cvRectangle(input, cvPoint(keypts->row + x -2, keypts->col + y -2), cvPoint(keypts->row + x +2, keypts->col + y +2),cvScalar(255, 255, 255), 1);
        keypts = keypts->next;
    }
    std::cout << "    Keycount : " << nCurrKeyCount << std::endl;
    DestroyAllResources();

    histo = cv::Mat::zeros(vTmpFeatureSet.size(), vTmpFeatureSet[0].size(), CV_32F);
    for(size_t i = 0; i < vTmpFeatureSet.size(); i++) {
          for(size_t j = 0; j < vTmpFeatureSet[0].size(); j++) {
              histo.at<float>(i, j) = vTmpFeatureSet[i][j];
          }
      }

    cvShowImage("Current Image", input);
    cvShowImage("Processed", cv_intensity);
    cvShowImage("Extracted", cv_extract);



    cvReleaseImage(&cv_extract);
    cvReleaseImage(&cv_intensity);
}



// !!!!!!!!!!!!!!!!!!!!!!!!!!!Creating the Histogramm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
void MakeHistogram (IplImage *input, IplImage *cv_binary, int x_min, int y_min, int width, int height) {

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

    cvNamedWindow("Extracted", CV_WINDOW_AUTOSIZE); cvMoveWindow("Extracted", 400, 1200);
    cvShowImage("Extracted", cv_input_tmp);

    ///// making object index //////
    // i.e. extract all the image information which are used to compute the histogramm
    int cnt = 0;
    int size = width * height;

    std::vector<int> index(size, 0);
    cv::Mat rgb = cv::Mat::zeros(size, 3, CV_8U);
    CvScalar s;
    int i = 0;
//            if(int(cv_binary_tmp->imageData[i]) == -1) {
//                  a = cv_input_tmp->imageData[i*3];
//                  b = cv_input_tmp->imageData[i*3+1];
//                  c = cv_input_tmp->imageData[i*3+2];
//                  rgb.at<unsigned char>(i,0) = a;
//                  rgb.at<unsigned char>(i,1) = b;
//                  rgb.at<unsigned char>(i,2) = c;
//                  counter123++;
//                  //Debugging
//                  printf("%u %u %u\n", a,b,c);

//                  // the positions of the relevant pixels are saved
//                  index[cnt] = i;
//                  cnt++;
//            }

    for(int y = 0; y < height; y++) {
        uchar* ptr = (uchar*) (cv_input_tmp->imageData + y * cv_input_tmp->widthStep);
        uchar* ptr2 = (uchar*) (cv_binary_tmp->imageData + y * cv_binary_tmp->widthStep);
        for(int x = 0; x < width; x++) {
            i++;
            if(ptr2[x] == 255) {
                rgb.at<unsigned char>(i,0) = ptr[3*x];
                rgb.at<unsigned char>(i,1) = ptr[3*x+1];
                rgb.at<unsigned char>(i,2) = ptr[3*x+2];
                index[cnt] = i;
                cnt++;
            }
        }
    }

            // only pixel that are white (first channel = 255) in the binary image are considered
            // for histogramm computing
//            if(cvGet2D(cv_binary_tmp,j,i).val[0] == 255) {
//                s = cvGet2D(cv_input_tmp,j,i);
//                // the image is searched column-wise
//                t = i * height + j;
//                rgb.at<unsigned char>(t,0) = (unsigned char)s.val[0];
//                rgb.at<unsigned char>(t,1) = (unsigned char)s.val[1];
//                rgb.at<unsigned char>(t,2) = (unsigned char)s.val[2];
//                index[cnt] = t;
//                cnt++;
//                printf("%u %u %u\n",(unsigned char)s.val[0],(unsigned char)s.val[1],(unsigned char)s.val[2]);
//            }


    index.resize(cnt);





    // Create the extracted image
//    CvScalar s;
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


    int res = 0;
    if (mode_slider == 1) cvRectangle(input, cvPoint(x_min - res, y_min - res), cvPoint(x_min+width-1 + res, y_min+height-1 + res), CV_RGB(255, 255, 255), 1);




    cvShowImage("Processed", cv_binary_tmp);
    cvShowImage("Extracted", cv_extract);

    cvReleaseImage(&cv_extract);
    cvReleaseImage(&cv_input_tmp);
    cvReleaseImage(&cv_binary_tmp);

    ///// calculate histogram //////

    int histVal = nHistBinNr*nHistBinNr*nHistBinNr+1;

//    double vRgbHist[nHistBinNr*nHistBinNr*nHistBinNr+1];
    std::vector<float> vRgbHist(histVal,0);

    for(int j = 0; j < histVal; j++) {
        vRgbHist[j] = 0;
    }

    histo = cv::Mat::zeros(8, histVal, CV_32F);

    i = 1;
    for(int j = 0; j < histVal; j++) {
        vRgbHist[j] = 0;
    }
    nColorMode = i;

    Calc3DColorHistogram(rgb, index, nHistBinNr, vRgbHist);

    for(int j = 0; j < histVal; j++) {
        histo.at<float>(i,j) = vRgbHist[j];
    }
    histo.at<float>(i,nHistBinNr*nHistBinNr*nHistBinNr) = i;



}




void Save2Disk(std::string sLibFileMode)
{

    std::string sLibFile;

    sLibFile = sOutputPath + sLibFileMode + sOutputFileName + ".yaml";
    cv::FileStorage fs(sLibFile,cv::FileStorage::WRITE);
    fs << "TestObjectFeatureVectors" << histo;

}




void ScanDir() {
    DIR *dirp;
    struct dirent *dp;
    //struct dirent *readdirp;
    const char *dirchar = sInputPath.c_str();
    dirp = opendir(dirchar);
    std::cout << "\nDirectory Opened\n";

    std::cout << "Opening File : " << sInputPath << "\n\n";

    while (dirp) {
        if ((dp = readdir(dirp)) != NULL) {
            std::string sFile = dp->d_name;
            size_t pos = sFile.find(".");
            std::string sExt = sFile.substr(pos+1);

            if(sExt.compare("bmp")==0 || sExt.compare("jpeg")==0 || sExt.compare("jpg")==0 || sExt.compare("tiff")==0 || sExt.compare("tif")==0 || sExt.compare("png")==0) {
                std::string sFileName = sInputPath + sFile;
                std::cout << "Opening File : " << sFileName << "\n\n";
                const char *c = sFileName.c_str();
                IplImage *cv_input = cvLoadImage(c);

//                original_image = cvCreateImage(cvSize(cv_input->width,cv_input->height), IPL_DEPTH_8U, 3);
                original_image = cvCloneImage(cv_input);

                printf("Anzahl der Kanäle ist %i\n", cv_input->nChannels);

                // Define default size of the binary image
                nFocusX = (original_image->width)/5;
                nFocusY = 10;
                nFocusWidth = (original_image->width)*3/5;
                nFocusHeight = (original_image->height)-20;


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
                cvThreshold(cv_gray, output, nThresh, 255, CV_THRESH_BINARY );


                binary_image = cvCloneImage(output);
                cvReleaseImage(&cv_gray);
                cvReleaseImage(&cv_bold);



                // Define Mousehandlers
                cvSetMouseCallback("Current Image", my_mouse_callback_original,original_image);
                cvSetMouseCallback("Processed", my_mouse_callback_binary,output);

                // With modeslider the user can select if sift or color histogramm should be computed
                // 0 = sift and color histogramm
                // 1 = color histogramm
                // 2 = sift
                // 3 = skip image
                // the default value is 0
                cvCreateTrackbar( "Mode", "Current Image", &mode_slider, 3, NULL);
                cvCreateTrackbar( "Threshold", "Current Image", &alpha_slider, 255, changeImage);
                cvCreateTrackbar( "Line diameter", "Processed", &diameter_slider, 30, NULL);
                cvShowImage("Processed", output);
                cvShowImage("Current Image", original_image);

                std::cout << "Press a key to continue" << std::endl;
                cCvCurrentkey = cv::waitKey(0);

                cvDestroyWindow("Processed");
                cvNamedWindow("Processed", CV_WINDOW_AUTOSIZE); cvMoveWindow("Processed", 1600, 200);


                // new Threshold
                nThresh = alpha_slider;



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
                    ExtractObject(cv_input, x_min, y_min, width, height);

                    switch (mode_slider) {
                    case 3:
                        continue;
                    case 0:
                        std::cout << "....Computing color histogram\n";
                        MakeHistogram(cv_input, binary_image, x_min, y_min, width, height);

                        cCvCurrentkey = cv::waitKey(0);

                        s = histo.size();
                        rows = s.height;

                        std::cout << "Saving " << rows << " Features to Disk....\n";

                        if (rows) {
                            sLibFileMode = "simplelib_3dch_";

                            Save2Disk(sLibFileMode);
                            printf("Library File Saved!\n\n");
                        }
                        else std::cout <<"Library file does not saved. No feature was extracted.\n\n";


                        std::cout << "....Computing SIFT features\n";
                        MakeSift(cv_input, x_min, y_min, width, height);
                        cCvCurrentkey = cv::waitKey();
                        s = histo.size();
                        rows = s.height;

                        std::cout << "Saving " << rows << " Features to Disk....\n";
                        if (rows) {sLibFileMode = "simplelib_sift_"; Save2Disk(sLibFileMode); std::cout << "Library File Saved!\n\n";}
                        else std::cout <<"Library file does not saved. No feature was extracted.\n\n";
                        break;
                    case 1:
                        std::cout << "....Computing color histogram\n";
                        MakeHistogram(cv_input, binary_image, x_min, y_min, width, height);
                        cCvCurrentkey = cv::waitKey();
                        s = histo.size();
                        rows = s.height;

                        std::cout << "Saving " << rows << " Features to Disk....\n";
                        if (rows) {sLibFileMode = "simplelib_3dch_"; Save2Disk(sLibFileMode); std::cout << "Library File Saved!\n\n";}
                        else std::cout <<"Library file does not saved. No feature was extracted.\n\n";
                        break;
                    case 2:
                        std::cout << "....Computing SIFT features\n";
                        MakeSift(cv_input, x_min, y_min, width, height);
                        cCvCurrentkey = cv::waitKey();
                        s = histo.size();
                        rows = s.height;

                        std::cout << "Saving " << rows << " Features to Disk....\n";
                        if (rows) {sLibFileMode = "simplelib_sift_"; Save2Disk(sLibFileMode); std::cout << "Library File Saved!\n\n";}
                        else std::cout <<"Library file does not saved. No feature was extracted.\n\n";
                        break;
                    }
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


// Fritjofs Code


// This is our mouse callback. If the user
// presses the left button, we start a box and add it
// to a copy of the current image. When the
// mouse is dragged (with the button down) we
// resize the box.
//
void my_mouse_callback_original(int event, int x, int y, int flags, void* param) {
    switch( event ) {
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

                draw_box(image, box);
                cvShowImage("Current Image", image);
            }
            break;
        }

        case CV_EVENT_LBUTTONDOWN: {
            drawing_box = true;
            box = cvRect(x, y, 0, 0);
            startbox_x = x;
            startbox_y = y;
            break;
        }

        case CV_EVENT_LBUTTONUP: {
            drawing_box = false;

            // Extracted Image is updated
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
            cvThreshold(cv_gray, output, alpha_slider, 255, CV_THRESH_BINARY );

            binary_image = cvCloneImage(output);
            cvShowImage("Processed", binary_image);

            cvReleaseImage(&cv_gray);
            cvReleaseImage(&cv_bold);
            cvReleaseImage(&output);


            break;
        }
    }
}


// This is the mouse handler for mouse events on the preview picture
// of the binary image. It allows the user to change some regions from
// black to white an vice versa.
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
    /*
    std::cout << "keyboard input hook for extra time to gently enable the qt debugger" << std::endl;
    int test_;
    std::cin >> test_;
    */


    parameter_init(argc, argv);
    if(argc<2){
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }
    printf ("Scanning Directory...\n");

    sInputPath = sPath + sOutputFileName + "/OnePos/";
    sOutputPath = sPath;


    cvNamedWindow("Current Image", CV_WINDOW_AUTOSIZE); cvMoveWindow("Current Image", 100, 100);
    cvNamedWindow("Processed", CV_WINDOW_AUTOSIZE); cvMoveWindow("Processed", 1600, 200);

    ScanDir();

    cvDestroyAllWindows();
    return (0);
}
