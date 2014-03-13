//last date of modification : 31-08-12 @Sahil
/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, NI Groupe, TU-Berlin.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted with no constrains
 */

#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>
#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "terminal_tools/parse.h"


using namespace std;

string sPath;
string sInputPath;
string sOutputPath1;
string sOutputPath2;
int nCropMode = 0;

//Global Data
int nCvCurrentkey = -1;



int nFocusX = 1, nFocusY = 17, nFocusWidth = 292, nFocusHeight = 220;


// initialize parameters
void parameter_init(int argc, char** argv) {
    terminal_tools::parse_argument (argc, argv, "-dir", sPath);
    terminal_tools::parse_argument (argc, argv, "-mode", nCropMode); if (nCropMode == 0) nCropMode = 0;
    terminal_tools::parse_argument (argc, argv, "-fox", nFocusX); if(nFocusX == 0) nFocusX = 0;
    terminal_tools::parse_argument (argc, argv, "-foy", nFocusY); if(nFocusY == 0) nFocusY = 0;
    terminal_tools::parse_argument (argc, argv, "-fow", nFocusWidth); if (nFocusWidth == 0) nFocusWidth = 292;
    terminal_tools::parse_argument (argc, argv, "-foh", nFocusHeight); if (nFocusHeight == 0) nFocusHeight = 220;

    printf ("\n\n");
    printf ("===============================================================\n");
    printf ("** Parameter setting **\n");
    printf ("===============================================================\n");
    switch (nCropMode) {
    case 0: printf ("Crop Mode: .............................. Devide Image\n"); break;
    case 1: printf ("Crop Mode: .............................. Crop Image\n"); break;
    case 2: printf ("Crop Mode: .............................. Merge Image\n"); break;
    }
    printf ("===============================================================\n\n\n");
}






void CropImage(cv::Mat cvm_input, cv::Mat &cvm_out, int nFocusX, int nFocusY, int nFocusWidth, int nFocusHeight)
{
    cvm_out = cv::Scalar(0, 0, 0);
    cv::Mat tmp;
    tmp = cvm_input(cv::Rect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    tmp.copyTo(cvm_out);
    tmp.release();
}

void MergeImage(cv::Mat cvm_input, cv::Mat &cvm_out, int nFocusX, int nFocusY, int nFocusWidth, int nFocusHeight, int nOutPosX, int nOutPosY)
{
    cv::Mat tmp1, tmp2;
    tmp1 = cvm_input(cv::Rect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    tmp2 = cvm_out(cv::Rect(nOutPosX, nOutPosY, nFocusWidth, nFocusHeight));
    tmp1.copyTo(tmp2);

    tmp1.release(); tmp2.release();
}

void ScanDir() {
    sInputPath = sPath;
    switch (nCropMode) {
    case 0:
        sOutputPath1 = sPath + "part1/";
        sOutputPath2 = sPath + "part2/";
        mkdir(sOutputPath1.data(), 0777);
        mkdir(sOutputPath2.data(), 0777);
        break;
    case 1: case 2:
        sOutputPath1 = sPath + "processed/";
        mkdir(sOutputPath1.data(), 0777);
        break;
    }

    DIR *dirp;
    struct dirent *dp;
    const char *dirchar = sInputPath.c_str();
    dirp = opendir(dirchar);
    cout << "\nDirectory " << sPath << " is opened\n";

    int cnt = 0;
    while (dirp) {
        if ((dp = readdir(dirp)) != NULL) {
            string sFileName = dp->d_name;
            size_t pos = sFileName.find(".");
            string sExt = sFileName.substr(pos+1);
            string sFile = sFileName.substr(0, pos);
            if(sExt.compare("bmp")==0 || sExt.compare("jpeg")==0 || sExt.compare("jpg")==0 || sExt.compare("tiff")==0 || sExt.compare("tif")==0) {
                string sInputFile = sInputPath + sFileName;
                //cout << "Opening File : " << sInputFile << "    " << cnt << "\n";
                cv::Mat cvm_input = cv::imread(sInputFile.data(), CV_LOAD_IMAGE_COLOR);
                cv::Mat cvm_output;

                if(!&cvm_input) {
                    cout << "Can not open the file : " << sInputFile << endl;
                    break;
                }


                string sOutputFile;
                switch (nCropMode) {
                case 0:     // Divide
                    cv::imshow("Current Image", cvm_input);

                    cvm_output.create(cv::Size(nFocusWidth, nFocusHeight), cvm_input.type());

                    CropImage(cvm_input, cvm_output, 0, 0, nFocusWidth, nFocusHeight);
                    sOutputFile = sOutputPath1 + sFile + "_trk" + "." + sExt; cv::imwrite(sOutputFile.data(), cvm_output);
                    cv::imshow("Processed", cvm_output);

                    CropImage(cvm_input, cvm_output, nFocusWidth, 0, nFocusWidth, nFocusHeight);
                    sOutputFile = sOutputPath2 + sFile + "_rgb" + "." + sExt; cv::imwrite(sOutputFile.data(), cvm_output);
                    cv::imshow("Result", cvm_output);

                    cvm_output.release();

                    break;

                case 1:     // Crop
                    cv::imshow("Current Image", cvm_input);

                    cvm_output.create(cv::Size(nFocusWidth, nFocusHeight), cvm_input.type());
                    CropImage(cvm_input, cvm_output, nFocusX, nFocusY, nFocusWidth, nFocusHeight);
                    sOutputFile = sOutputPath1 + sFileName; cv::imwrite(sOutputFile.data(), cvm_output);
                    cv::imshow("Result", cvm_output);
                    cvm_output.release();

                    break;

                case 2:     // Merge
                    if (pos < 20) {
                        cv::imshow("Current Image", cvm_input);

                        string sPrefix = sFileName.substr(0, 11);

                        DIR *dirp2;
                        struct dirent *dp2;
                        const char *dirchar2 = sInputPath.c_str();
                        dirp2 = opendir(dirchar2);

                        while (dirp2) {
                            if ((dp2 = readdir(dirp2)) != NULL) {

                                string sFileName2 = dp2->d_name;
                                string sPrefix2 = sFileName2.substr(0, 11);
                                if (sPrefix != sPrefix2) continue;

                                size_t pos2 = sFileName2.find(".");
                                if (pos2 < 20) continue;


                                string sInputFile2 = sInputPath + sFileName2;

                                cv::Mat cvm_input2 = cv::imread(sInputFile2.data(), CV_LOAD_IMAGE_COLOR);

                                cvm_output.create(cv::Size(nFocusWidth, nFocusHeight*2), cvm_input.type());
                                cvm_output = cv::Scalar(0, 0, 0);

                                int nOutPosX = 0, nOutPosY = 0;
                                MergeImage(cvm_input2, cvm_output, nFocusX, nFocusY, nFocusWidth, nFocusHeight, nOutPosX, nOutPosY);
                                nOutPosX = 0, nOutPosY = nFocusHeight;
                                MergeImage(cvm_input, cvm_output, nFocusX, nFocusY, nFocusWidth, nFocusHeight, nOutPosX, nOutPosY);

//                                cvm_output.create(cv::Size(nFocusWidth*2, nFocusHeight), cvm_input.type());
//                                cvm_output = cv::Scalar(0, 0, 0);

//                                int nOutPosX = 0, nOutPosY = 0;
//                                MergeImage(cvm_input, cvm_output, nFocusX, nFocusY, nFocusWidth, nFocusHeight, nOutPosX, nOutPosY);
//                                nOutPosX = nFocusWidth, nOutPosY = 0;
//                                MergeImage(cvm_input2, cvm_output, nFocusX, nFocusY, nFocusWidth, nFocusHeight, nOutPosX, nOutPosY);
                                sOutputFile = sOutputPath1 + sPrefix2 + "." + sExt;
                                cv::imwrite(sOutputFile.data(), cvm_output);
                                cv::imshow("Result", cvm_output);

                                cvm_input2.release();
                                cvm_output.release();

                                break;
                            }
                            else {
                                closedir(dirp2);
                                return;
                            }

                        }
                    }
                    break;
                }

                cvm_input.release();
            }
        }
        else {
            closedir(dirp);
            return;
        }
        cnt++;
        nCvCurrentkey = cv::waitKey(3);
    }
    printf("%d\n", cnt);
}





int main (int argc, char** argv)
{
    parameter_init(argc, argv);
    if(argc<2){
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }
    printf ("Scanning Directory...\n");


    switch (nCropMode) {
    case 0: case 1: case 2:
        cv::namedWindow("Current Image", CV_WINDOW_AUTOSIZE); cvMoveWindow("Current Image", 180, 100);
        cv::namedWindow("Processed", CV_WINDOW_AUTOSIZE); cvMoveWindow("Processed", 100, 200);
        cv::namedWindow("Result", CV_WINDOW_AUTOSIZE); cvMoveWindow("Result", 100, 600);
        ScanDir();
        break;
    default: cout << "Mode number is not correct!\n";
    }



    cvDestroyWindow("Current Image");
    cvDestroyWindow("Processed");
    cvDestroyWindow("Result");
    return (0);
}
