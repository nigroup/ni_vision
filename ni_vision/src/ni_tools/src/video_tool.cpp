#include <ros/ros.h>
#include <sys/stat.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#include "terminal_tools/parse.h"


<<<<<<< HEAD

#define DIVIDE 2
#define MERGE 3
=======
>>>>>>> parent of 3b1d971... 4.Versuch

int nMode = 0;
bool bOutVideo = true, bXFrame = false;
std::string sDirectory;
std::string sFileName;
std::string sExt;
int nFocusX = 1, nFocusY = 17, nFocusWidth = 292, nFocusHeight = 220;
int nFrameStart = 0, nFrameEnd = 1000;


void CropImage(cv::Mat cvm_input, cv::Mat &cvm_out, int nFocusX, int nFocusY, int nFocusWidth, int nFocusHeight)
{
    cvm_out = cv::Scalar(0, 0, 0);
    cv::Mat tmp;
    tmp = cvm_input(cv::Rect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    tmp.copyTo(cvm_out);
    tmp.release();
}

// initialize parameters
void parameter_init(int argc, char** argv) {
    terminal_tools::parse_argument (argc, argv, "-path", sDirectory);
    terminal_tools::parse_argument (argc, argv, "-file", sFileName);
    terminal_tools::parse_argument (argc, argv, "-ext", sExt);
    terminal_tools::parse_argument (argc, argv, "-mode", nMode); if (nMode == 0) nMode = 0;
    terminal_tools::parse_argument (argc, argv, "-frame", bXFrame);
    terminal_tools::parse_argument (argc, argv, "-fox", nFocusX); if(nFocusX == 0) nFocusX = 0;
    terminal_tools::parse_argument (argc, argv, "-foy", nFocusY); if(nFocusY == 0) nFocusY = 0;
    terminal_tools::parse_argument (argc, argv, "-fow", nFocusWidth); if (nFocusWidth == 0) nFocusWidth = 292;
    terminal_tools::parse_argument (argc, argv, "-foh", nFocusHeight); if (nFocusHeight == 0) nFocusHeight = 220;

    terminal_tools::parse_argument (argc, argv, "-fstart", nFrameStart); if (nFrameStart == 0) nFrameStart = 0;
    terminal_tools::parse_argument (argc, argv, "-fend", nFrameEnd); if (nFrameEnd == 0) nFrameEnd = 1000;

    printf ("\n\n");
    printf              ("===============================================================\n");
    printf              ("** Parameter setting **\n");
    printf              ("===============================================================\n");
    switch (nMode) {
    case 0: printf      ("Mode: .............................. Convert Video\n"); break;
    case 1: printf      ("Mode: .............................. Crop Video\n"); break;
    case DIVIDE: printf ("Mode: .............................. Divide Video\n"); break;
    case MERGE: printf  ("Mode: .............................. Merge Videos\n"); break;
    }
    if (bXFrame) printf ("Extracting Frames: ................. Yes\n");
    else printf         ("Extracting Frames: ................. No\n");
    printf              ("===============================================================\n\n\n");
}


int main(int argc, char* argv[]) {

    parameter_init(argc, argv);

    std::string sInput = sDirectory + sFileName;
    std::string sOutputPath = sDirectory + "finished";
    std::string sOutput = sOutputPath + "/" + sFileName;
    mkdir(sOutputPath.data(), 0777);

    cv::VideoCapture cap(sInput.data()); // open the video file for reading

    if ( !cap.isOpened() )  // if not success, exit program
    {
         std::cout << "Cannot open the video file" << std::endl;
         return -1;
    }



    //// set Frame Fate and Start Timing
    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms
    double frame_rate = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    //frame_rate = 15.0;

    //// set Canvas ////////////////////
    cv::Size size_out, size_out2;
    switch (nMode) {
    case 0: size_out = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)); break;
    case 1: size_out = cv::Size(nFocusWidth, nFocusHeight); break;
    case DIVIDE: size_out = cv::Size(nFocusWidth, nFocusHeight); size_out2 = cv::Size(nFocusWidth, nFocusHeight); break;
    case MERGE: size_out = cv::Size(nFocusWidth*2, nFocusHeight); break;
    //size_out = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH)-6, cap.get(CV_CAP_PROP_FRAME_HEIGHT)-4);
    }

    std::cout << "Frame per seconds : " << frame_rate << std::endl;

    cv::namedWindow("Video Input", CV_WINDOW_AUTOSIZE); //create a window called "Video Input"
    cvMoveWindow("Video Input", 100, 0);



    //// set Video Writer ////////////////
    CvVideoWriter *writer_out = NULL, *writer_out2 = NULL;
    if (bOutVideo) {
        int fourcc = CV_FOURCC('D', 'I', 'V', 'X'); // MPEG-4 codec
        writer_out = cvCreateVideoWriter(sOutput.data(), fourcc, frame_rate, size_out, 1);
        if (nMode == DIVIDE) writer_out2 = cvCreateVideoWriter(sOutput.data(), fourcc, frame_rate, size_out, 1);
    }


    int nFrameCntr = 0, nFrameReserve = 3;
    while(1) {
        cv::Mat cvm_frame;

        bool bSuccess = cap.read(cvm_frame); // read a new frame from video

        if (!bSuccess) {//if not success, break loop
            if (nFrameCntr) std::cout << "Process is completed" << std::endl;
            else std::cout << "Cannot read the frame from video file" << std::endl;
            cvReleaseVideoWriter(&writer_out);
            break;
        }

        else {
            nFrameCntr++;
            if (nFrameCntr < nFrameStart || nFrameCntr > nFrameEnd+nFrameReserve) continue;

            cv::imshow("Video Input", cvm_frame); //show the frame in "Video Input" window


            cv::Mat cvm_out(size_out, cvm_frame.type()); cvm_out = cv::Scalar(0, 0, 0);

            IplImage *cv_out = cvCreateImage(cvSize(cvm_out.cols, cvm_out.rows), IPL_DEPTH_8U, 3); cvZero(cv_out);
            cv_out->imageData = (char *) cvm_out.data;
            //cv_out = cvCloneImage(&(IplImage)cvm_out);

            char sFile[128];
            sprintf(sFile, "frame_%06i", nFrameCntr);

            switch (nMode) {
            case 0:
                cvm_frame.copyTo(cvm_out);
                if (nFrameCntr > nFrameEnd) cvm_out = cv::Scalar(0,0,0);
                if (bOutVideo) cvWriteFrame(writer_out, cv_out);
                if (bXFrame) {std::string sOutputFile = sOutputPath + "/" + sFile + "." + sExt; cv::imwrite(sOutputFile.data(), cvm_out);}
                break;
            case 1:
                CropImage(cvm_frame, cvm_out, nFocusX, nFocusY, nFocusWidth, nFocusHeight);
                if (false) {
                    cv::Mat tmp_in, tmp_out;
                    tmp_in = cvm_frame(cv::Rect(4, 3, 317, 524));
                    tmp_out = cvm_out(cv::Rect(3, 3, 317, 524));
                    tmp_in.copyTo(tmp_out);
                    tmp_in = cvm_frame(cv::Rect(2, 529, 317, 496));
                    tmp_out = cvm_out(cv::Rect(3, 525, 317, 496));
                    tmp_in.copyTo(tmp_out);
                    tmp_in = cvm_frame(cv::Rect(321, 0, cvm_out.cols-322, cvm_out.rows-2));
                    tmp_out = cvm_out(cv::Rect(319, 0, cvm_out.cols-322, cvm_out.rows-2));
                    tmp_in.copyTo(tmp_out);
                    tmp_in.release(); tmp_out.release();
                    cv::rectangle (cvm_out, cv::Point(1, 1), cv::Point(cvm_out.cols-2, cvm_out.rows-2), cv::Scalar(63, 63, 63), 2);
                    cv::line (cvm_out, cv::Point(0, cvm_out.rows-1), cv::Point(cvm_out.cols, cvm_out.rows-1), cv::Scalar(63, 63, 63), 2);
                }
                if (false) {
                    cv::Mat tmp_in, tmp_out;
                    tmp_in = cvm_frame(cv::Rect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
                    tmp_out = cvm_out(cv::Rect(0, 0, nFocusWidth, nFocusHeight));
                    tmp_in.copyTo(tmp_out);
                    tmp_in = cvm_frame(cv::Rect(320+nFocusX, nFocusY, nFocusWidth, nFocusHeight));
                    tmp_out = cvm_out(cv::Rect(nFocusWidth, 0, nFocusWidth, nFocusHeight));
                    tmp_in.copyTo(tmp_out);
                    tmp_in.release(); tmp_out.release();
                }
                if (nFrameCntr > nFrameEnd) cvm_out = cv::Scalar(0,0,0);
                if (bOutVideo) cvWriteFrame(writer_out, cv_out);
                if (bXFrame) {std::string sOutputFile = sOutputPath + "/" + sFile + "." + sExt; cv::imwrite(sOutputFile.data(), cvm_out);}
                break;
            case DIVIDE:
                cv::Mat cvm_output2(size_out2, cvm_frame.type());
                CropImage(cvm_frame, cvm_output, 0, 0, nFocusWidth, nFocusHeight);

                CropImage(cvm_frame, cvm_output2, nFocusWidth, 0, nFocusWidth, nFocusHeight);
                sOutputFile = sOutputPath + "/" + sFile + "_trk" + "." + sExt; cv::imwrite(sOutputFile.data(), cvm_output);
                sOutputFile = sOutputPath + "/" + sFile + "_rgb" + "." + sExt; cv::imwrite(sOutputFile.data(), cvm_output);


                cv::imshow("Processed1", cvm_output); cv::imshow("Processed2", cvm_output2);
                cvm_output2.release();
                break;
            case MERGE: break;
            }


            switch (nMode) {
            case 3:         // Extract streams and divide
                {
                }break;
            }
            cvReleaseImage(&cv_out);
            cvm_out.release();
        }
        cvm_frame.release();


        if(cv::waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;

}
