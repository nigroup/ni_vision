/*
 * other functions
 */


#include <numeric>

///////////////////////// temporal //////////////////////////////////////
std::vector<std::vector<int> > mnTmpPtsIdx;
std::vector<std::vector<bool> > mbMat1;
std::vector<std::vector<bool> > mbMat2;
std::vector<std::vector<bool> > mbMat3;
std::vector<float> vnTmpGradX;
std::vector<float> vnTmpGradY;
std::vector<bool> vbTmpSmall;
/////////////////////////////////////////////////////////////////////////


void MakeNiDirectory (std::string input, std::string &output)
{
    time_t now = time(0); tm *ltm = localtime(&now);
    char s_time[128]; sprintf(s_time, "%d%02i%02i_%02i%02i%02i", ltm->tm_year + 1900, ltm->tm_mon, ltm->tm_mday, ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
    output = input + s_time; mkdir(output.data(), 0777);
}


void PrintTimes1 (std::string sTimeFile, int nCntRecCycle, std::vector<std::vector<int> > mnTimeMeas1, std::vector<std::vector<float> > mnTimeMeas2)
{
    char sText[168];
    std::ofstream finn;

    if (mnTimeMeas1.size()) {
        finn.open(sTimeFile.data(), ios::app);
        for (size_t i = 0; i < mnTimeMeas1.size(); i++) {
            if (mnTimeMeas1[i][0] > nCntRecCycle) break;
            sprintf(sText, "%3d. Cycle - frame: %6d,  ", mnTimeMeas1[i][0], mnTimeMeas1[i][1]); finn << sText;
            sprintf(sText, "%4.0f ms (SIFT %4.0f ms)", mnTimeMeas2[i][0], mnTimeMeas2[i][1]); finn << "Object found at: " << sText;
            //sprintf(sText, "%4.0f ms (SIFT %4.0f ms, width: %3d, height: %3d, ", mnTimeMeas2[i][0], mnTimeMeas2[i][1], mnTimeMeas1[i][7], mnTimeMeas1[i][8]); finn << "Object found at: " << sText;
            //sprintf(sText, "%3d Keys, %3d initial matches, %2d true positives).", mnTimeMeas1[i][4], mnTimeMeas1[i][5], mnTimeMeas1[i][6]); finn << sText << "   ";
            sprintf(sText, "%4d. of %2d candidates queue", mnTimeMeas1[i][2], mnTimeMeas1[i][3]); finn << sText << "\n";
        }
        finn.close();
    }
}


void PrintTimes2 (std::string sTimeFile, int nCntRecCycle, double nTimeRecFound_avr, double nTimeRecFound_min, double nTimeRecFound_max, std::vector<int> vnRecogRating, std::vector<std::vector<float> > mnTimeMeas2,
                  int nCntFrame_tmp, double nTimeTotal_avr, double nFrameRate_avr, double nTimeDepth_avr, double nTimeBlur_avr, double nTimeSegm_avr, double nTimeTrack_avr, double nTimeAtt_avr, double nTimeRec_avr, double nTimeSift_avr, double nTimeFlann_avr)
{
    char sText[128];
    std::ofstream finn;

    int cnt_tmp = 0;
    double stdev = 0;
    for (size_t i = 0; i < mnTimeMeas2.size(); i++) {
        if (i >= nCntRecCycle) break;
        stdev += pow(nTimeRecFound_avr - mnTimeMeas2[i][0], 2);
        cnt_tmp++;
    }
    stdev = sqrt(stdev/cnt_tmp);

    finn.open(sTimeFile.data(), ios::app);

    finn << "\n";
    sprintf(sText, "Duration of object recognition over %d recognition cycles", nCntRecCycle); finn << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeRecFound_avr); finn << "Avr. duration for object finding:    " << sText; sprintf(sText, "%5.2f", stdev); finn << " (stdev.: " << sText << ")" << "\n";
    sprintf(sText, "%7.2f ms", nTimeRecFound_min); finn << "Min. duration for object finding:    " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeRecFound_max); finn << "Max. duration for object finding:    " << sText << "\n\n";

    finn << "Attention\n";
    for (int i = 1; i < 4; i++) if (vnRecogRating[i]) {sprintf(sText, "Target object was placed at %d. place of the queue: %d times (%d%%)\n", i, vnRecogRating[i], int(100*vnRecogRating[i]/nCntRecCycle)); finn << sText;}
    finn << "\n";

    double nTimePreTotal = nTimeDepth_avr + nTimeBlur_avr;
    double nTimeAttTotal = nTimeSegm_avr + nTimeTrack_avr + nTimeAtt_avr + nTimePreTotal;
    double nTimeRecEtc = nTimeRec_avr - nTimeSift_avr - nTimeFlann_avr;
    double nTimeEtc = nTimeTotal_avr - nTimeRec_avr - nTimeAttTotal;
    sprintf(sText, "Average time for individual processes over %d frames", nCntFrame_tmp); finn << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeEtc); finn <<               "Overhead-Total:    " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimePreTotal); finn <<          "Pre-Processing:    " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeSegm_avr); finn <<          "Segmentation:      " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeTrack_avr); finn <<         "Tracking:          " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeAtt_avr); finn <<           "Selection:         " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeAttTotal); finn <<          "Attention-Total:   " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeSift_avr); finn <<          "SIFT:              " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeFlann_avr); finn <<         "FLANN:             " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeRecEtc); finn <<            "Other:             " << sText << "\n";
    sprintf(sText, "%7.2f ms", nTimeRec_avr); finn <<           "Recognition-Total: " << sText << "\n";

    sprintf(sText, "%7.2f ms", nTimeTotal_avr); finn <<         "TOTAL:             " << sText; sprintf(sText, " (%.2f Hz)", nFrameRate_avr); finn << sText << "\n\n";

    finn.close();
}


void PrintTimes3 (std::string sTimeFile_detail, std::vector<std::vector<float> > mnTimeMeas3)
{
    char sText[128];
    std::ofstream finn;

    finn.open(sTimeFile_detail.data(), ios::app);

//    std::vector<std::vector<float> > mnTemp(mnTimeMeas3[0].size(), std::vector<float> (mnTimeMeas3.size(), 0));
//    for (size_t i = 0; i < mnTimeMeas3.size(); i++) {
//        for (size_t j = 0; j < mnTimeMeas3[0].size(); j++) {
//            mnTemp[j][i] = mnTimeMeas3[i][j];
//        }
//    }

//    for (size_t i = 0; i < mnTimeMeas3.size(); i++) {
//        double sum = std::accumulate(mnTemp[i].begin(), mnTemp[i].end(), 0.0);
//        double mean = sum / mnTemp[i].size();

//        std::vector<double> diff(mnTemp[i].size());
//        std::transform(mnTemp[i].begin(), mnTemp[i].end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
//        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//        double stdev = std::sqrt(sq_sum / mnTemp[i].size());
//    }

    int start = 5;
    sprintf(sText, "=AVERAGE(B%d:B%d),", start, start + (int)mnTimeMeas3.size() -1 -1);
    finn << "Mean," << sText << "\n";
    sprintf(sText, "=STDEV(B%d:B%d),", start, start + (int)mnTimeMeas3.size() -1 -1);
    finn << "Standard dev.," << sText << "\n\n";

    std::vector<std::string> sIndex;
    sIndex.resize(26);
    sIndex[0] = "Overhead,";
    sIndex[1] = "Pre-processing,";
    sIndex[2] = "Segmentation,";
    sIndex[3] = "Tracking,";
    sIndex[4] = "Selection,";
    sIndex[5] = "Attention-Total,";
    sIndex[6] = "SIFT,";
    sIndex[7] = "FLANN,";
    sIndex[8] = "Other,";
    sIndex[9] = "Recognition-Total,";
    sIndex[10] = "Total,";
    sIndex[11] = ",";
    sIndex[12] = "No. of Keypts.,";
    sIndex[13] = "Section size (pxls),";
    sIndex[14] = "Section size,";
    sIndex[15] = ",";
    sIndex[16] = "Initialization,";
    sIndex[17] = "Depth Processing,";
    sIndex[18] = "Smoothing,";
    sIndex[19] = "Segmentation,";
    sIndex[20] = "Tracking,";
    sIndex[21] = "Selection,";
    sIndex[22] = "SIFT,";
    sIndex[23] = "FLANN,";
    sIndex[24] = "Recognition,";
    sIndex[25] = "Total,";

    finn << "Frame no.,";
    for (size_t j = 0; j < sIndex.size(); j++) finn << sIndex[j].data();

    finn << "\n";

    for (size_t i = 1; i < mnTimeMeas3.size(); i++) {
        sprintf(sText, "%3d,", (int)i); finn << sText;
        double nPreProc = mnTimeMeas3[i][1] + mnTimeMeas3[i][2];
        double nAttTotal = nPreProc + mnTimeMeas3[i][3] + mnTimeMeas3[i][4] + mnTimeMeas3[i][5];
        double nRecEtc = mnTimeMeas3[i][8] - mnTimeMeas3[i][6] - mnTimeMeas3[i][7];
        double nOverhead = mnTimeMeas3[i][9] - nAttTotal - mnTimeMeas3[i][8];

        sprintf(sText, "%7.2f,", nOverhead); finn << sText;
        sprintf(sText, "%7.2f,%7.2f,%7.2f,%7.2f,%7.2f,", nPreProc, mnTimeMeas3[i][3], mnTimeMeas3[i][4], mnTimeMeas3[i][5], nAttTotal); finn << sText;
        sprintf(sText, "%7.2f,%7.2f,%7.2f,%7.2f,", mnTimeMeas3[i][6], mnTimeMeas3[i][7], nRecEtc, mnTimeMeas3[i][8]); finn << sText;
        sprintf(sText, "%7.2f,", mnTimeMeas3[i][9]); finn << sText;
        finn << ",";
        sprintf(sText, "%5.0f,%7.0f,%4.0f x%4.0f,", mnTimeMeas3[i][10], mnTimeMeas3[i][11] * mnTimeMeas3[i][12], mnTimeMeas3[i][11], mnTimeMeas3[i][12]); finn << sText;
        finn << ",";

        for (size_t j = 0; j < mnTimeMeas3[0].size(); j++) {
            sprintf(sText, "%7.2f,", mnTimeMeas3[i][j]);
            finn << sText;
        }
        finn << "\n";
    }

    finn.close();
}


void PrintTimesFromVector (std::string file, std::vector<float> input)
{
    char sText[128];
    std::ofstream finn;

    finn.open(file.data(), ios::app);

    int start = 2;
    finn << "Duration (ms)\n";
    sprintf(sText, "%7.2f,,Mean,=AVERAGE(A%d:A%d)", input[0], start, start+(int)input.size()-1); finn << sText << "\n";
    sprintf(sText, "%7.2f,,Standard dev.,=STDEV(A%d:A%d)", input[1], start, start+(int)input.size()-1); finn << sText << "\n";
    for (size_t i = 2; i < input.size(); i++) {
        sprintf(sText, "%7.2f", input[i]); finn << sText << "\n";
    }

    finn.close();
}

void PrintTimesSiftWhole (std::string file, std::vector<std::vector<float> > input, int width, int height)
{
    char sText[128];
    std::ofstream finn;

    finn.open(file.data(), ios::app);

    int start = 2;
    finn << "SIFT (ms),FLANN (ms),No. of Kye-points,,,SIFT (ms),FLANN (ms),No. of Kye-points,Section size\n";
    sprintf(sText, "%7.2f,%7.2f,%5.0f,,Mean,=AVERAGE(A%d:A%d),=AVERAGE(B%d:B%d),=AVERAGE(C%d:C%d),%dx%d", input[0][0], input[0][1], input[0][2], start, start+(int)input.size()-1, start, start+(int)input.size()-1, start, start+(int)input.size()-1, width, height); finn << sText << "\n";
    sprintf(sText, "%7.2f,%7.2f,%5.0f,,Standard dev.,=STDEV(A%d:A%d),=STDEV(B%d:B%d),=STDEV(C%d:C%d),0", input[1][0], input[1][1], input[0][2], start, start+(int)input.size()-1, start, start+(int)input.size()-1, start, start+(int)input.size()-1); finn << sText << "\n";
    for (size_t i = 2; i < input.size(); i++) {
        sprintf(sText, "%7.2f,%7.2f,%5.0f", input[i][0], input[i][1], input[i][2]); finn << sText << "\n";
    }

    finn.close();
}




/*
void ResetTime (int nCntDepth, int nCntBlur, int nCntSegm, int nCntTrack, int nCntRec, int nCntSift, int nCntFlann, int nCntRecCycle, int nCntGbSegm, int nCntSiftWhole, int nCntFrame_tmp,
                double nTimeRecCycle, double nTimeRecFound, double nTimeDepth_acc, double nTimeBlur_acc, double nTimeTotal_acc, double nTimeSegm_acc, double nTimeTrack_acc,
                double nTimeRec_acc, double nTimeSift_acc, double nTimeFlann_acc, double nTimeRecCycle_acc, double nTimeRecFound_acc, double nTimeGbSegm_acc, double nTimeSiftWhole_acc, double nTimeFlannWhole_acc,
                double nTimeDepth_avr, double nTimeBlur_avr, double nTimeTotal_avr, double nTimeSegm_avr, double nTimeTrack_avr,
                double nTimeRec_avr, double nTimeSift_avr, double nTimeFlann_avr, double nTimeRecCycle_avr, double nTimeRecFound_avr, double nTimeRecFound_max, double nTimeRecFound_min,
                double nTimeGbSegm_avr, double nTimeSiftWhole_avr, double nTimeFlannWhole_avr)
{
    nCntDepth = 0; nCntBlur = 0;
    nCntSegm = 0; nCntTrack = 0; nCntRec = 0; nCntSift = 0; nCntFlann = 0;
    nCntRecCycle = 0; nCntGbSegm = 0; nCntSiftWhole = 0; nCntFrame_tmp = 0;

    nTimeRecCycle = 0; nTimeRecFound = 0;
    nTimeDepth_acc = 0; nTimeBlur_acc = 0; nTimeTotal_acc = 0;
    nTimeSegm_acc = 0; nTimeTrack_acc = 0; nTimeRec_acc = 0; nTimeSift_acc = 0; nTimeFlann_acc = 0;
    nTimeRecCycle_acc = 0; nTimeRecFound_acc = 0;
    nTimeGbSegm_acc = 0; nTimeSiftWhole_acc = 0; nTimeFlannWhole_acc = 0;

    nTimeDepth_avr = 0; nTimeBlur_avr = 0; nTimeTotal_avr = 0;
    nTimeSegm_avr = 0; nTimeTrack_avr = 0; nTimeRec_avr = 0; nTimeSift_avr = 0; nTimeFlann_avr = 0;
    nTimeRecCycle_avr = 0, nTimeRecFound_avr = 0, nTimeRecFound_max = 0, nTimeRecFound_min = 10000;
    nTimeGbSegm_avr = 0; nTimeSiftWhole_avr = 0; nTimeFlannWhole_avr = 0;
}*/



void AttachImgs (cv::Mat cvm_sec1, cv::Mat cvm_sec2, cv::Mat cvm_sec3, cv::Mat cvm_sec4, int nDsWidth, int nDsHeight, cv::Mat &cvm_out) {
    cv::Mat tmp;
    tmp = cvm_out(cv::Rect(0, 0, nDsWidth, nDsHeight)); cvm_sec1.copyTo(tmp);
    tmp = cvm_out(cv::Rect(nDsWidth, 0, nDsWidth, nDsHeight)); cvm_sec2.copyTo(tmp);
    tmp = cvm_out(cv::Rect(0, nDsHeight, nDsWidth, nDsHeight)); cvm_sec3.copyTo(tmp);
    tmp = cvm_out(cv::Rect(nDsWidth, nDsHeight, nDsWidth, nDsHeight)); cvm_sec4.copyTo(tmp);
    tmp.release();
}



void accCutImage(cv::Mat cvm_input, cv::Mat &cvm_out, int nDepthX, int nDepthY, int nDepthWidth, int nDepthHeight)
{
    cvm_out = cv::Scalar(0, 0, 0);
    cv::Mat tmp;
    tmp = cvm_input(cv::Rect(nDepthX, nDepthY, nDepthWidth, nDepthHeight)); tmp.copyTo(cvm_out);
}
