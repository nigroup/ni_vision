/*
 * Functions for Segmentation and tracking
 */


#include "munkres/munkres.h"

void LP(IplImage *input, IplImage *output)
{
    int lcutoff = 130;
    int hcutoff = 255;
    uchar *data_out, *data_in;

    int height_out, width_out, step_out, channels_out;
    int step_in, channels_in;

    height_out = output->height;
    width_out = output->width;
    step_out = output->widthStep;
    step_in = input->widthStep;

    channels_out = output->nChannels;
    channels_in = input->nChannels;
    data_out = (uchar *)output->imageData;
    data_in = (uchar *)input->imageData;

    /*Do not get confused.Here i have just taken the first channel of the image regardless if the image is multichannel*/
    /*first copy the first channel into the output image.*/

    for (int i = 0; i < height_out; i++)
        for (int j = 0; j < width_out; j++)
            data_out[i*step_out + j*channels_out] = data_in[i*step_in + j*channels_in];

    /*Apply the filter in these for loops*/
    for (int i = 0; i < height_out; i++)
        for (int j = 0; j < width_out; j++) {
            if(data_out[i*step_out + j*channels_out] < lcutoff || data_out[i*step_out + j*channels_out] > hcutoff)
                data_out[i*step_out + j*channels_out]=0;
        }
}



void DSegm_FlatDepthGrad_old (cv::Mat cvm_input, std::vector<int> index, int width, int height, float max, float min, int mode, int scenter, int sband, float factor, std::vector<float> &output_blur, std::vector<float> &output_ct)
{
    float scale =  (max-min), x;
    uint8_t gr;
    for(size_t i = 0; i < index.size(); i++) {
        gr = cvm_input.data[index[i]];
        if (gr) {
            x = (gr-1) * scale /254 + min;
            output_blur[index[i]] = x;
            if (mode == 1) {
                if (gr > scenter) gr = scenter + int(factor*sband);
                else gr = scenter - int(factor*sband);
            }
            x = (gr-1) * scale /254 + min;
            output_ct[index[i]] = x;
        }
    }
}

void DSegm_FlatDepthGrad (cv::Mat cvm_input, std::vector<int> index, float max, float min, int mode, int scenter, int sband1, int sband2, float factor, std::vector<float> &output_blur, std::vector<float> &output_ct)
{
    float scale =  (max-min), x;
    uint8_t gr;
    for(size_t i = 0; i < index.size(); i++) {
        gr = cvm_input.data[index[i]];
        if (gr) {
            x = (gr-1) * scale /254 + min;
            output_blur[index[i]] = x;
            switch (mode) {
            case 1:
                if (gr > scenter - sband1 && gr < scenter + sband2) gr = scenter;
                else if (gr >= scenter + sband2) {
                    if (gr + sband2 > 255) gr = 255;
                    else gr += sband2;
                }
                else {
                    if (scenter - sband1 < 1) gr = 1;
                    else gr -= sband1;
                }

                break;
            case 2:
                if (gr > scenter - sband1 && gr < scenter + sband2) gr = scenter;
                else if (gr >= scenter + sband2) {
                    if (scenter + int(factor*sband2) > 255) gr = 255;
                    else gr = scenter + int(factor*sband2);
                }
                else {
                    if (scenter - int(factor*sband1) < 1) gr = 1;
                    else gr = scenter - int(2*factor*sband1);
                }

                break;
            }
            x = (gr-1) * scale /254 + min;
            output_ct[index[i]] = x;
        }
    }
}

void DSegm_SmoothDepthGrad (std::vector<float> vInput, std::vector<int> index, cv::Size size, float max, float min, float none, int fmode, int fsize, int smode, int scenter, int sband1, int sband2, float clfac, std::vector<float> &output_blur, std::vector<float> &output_ct)
{
    cv::Mat input_gray(size, CV_8UC1, cv::Scalar(0));
    cv::Mat input_gray_blur(size, CV_8UC1, cv::Scalar(0));

    DrawDepthGrad (vInput, index, 0, max, min, none, input_gray);

    switch (fmode) {
        case 0: cv::blur(input_gray, input_gray_blur, cv::Size(fsize, fsize), cv::Point(-1,-1)); break;
        case 1: cv::GaussianBlur(input_gray, input_gray_blur, cv::Size(fsize, fsize), 0, 0); break;
        case 2: cv::medianBlur(input_gray, input_gray_blur, fsize); break;
        case 3: cv::Laplacian(input_gray, input_gray_blur, IPL_DEPTH_8U, fsize, 1, 0); break;
        //case 4: cv::filter2D(input_gray, input_gray_blur, IPL_DEPTH_8U, kernel, cv::Point(-1,-1), 0); break;
    }
    //kernel.release();
    //input_gray.copyTo(input_gray_blur);
    //input_gray_blur.copyTo(input_gray);
    //DSegm_FlatDepthGrad_old (input_gray_blur, index, size.width, size.height, max, min, smode, scenter, sband1, clfac, output_blur, output_ct);
    DSegm_FlatDepthGrad (input_gray_blur, index, max, min, smode, scenter, sband1, sband2, clfac, output_blur, output_ct);
    input_gray.release(); input_gray_blur.release();
}

void DSegm_NeighborMatrix (std::vector<int> vInputMap, std::vector<int> input_idx, int range, int width, std::vector<std::vector<bool> >& mnOut) {
    for (size_t i = 0; i < input_idx.size() ; i++) {
        int x, y;
        int Ref, Cand;
        GetPixelPos(input_idx[i], width, x, y);
        Ref = vInputMap[input_idx[i]];
        if (!Ref) continue;

        if (y + range-1) {
            Cand = vInputMap[input_idx[i] - width];
            if (Cand) {
                if (!mnOut[Ref][Cand])
                    if (Cand != Ref) {mnOut[Ref][Cand] = true; mnOut[Cand][Ref] = true;}
            }
        }
        if (x + range-1) {
            Cand = vInputMap[input_idx[i] - 1];
            if (Cand) {
                if (!mnOut[Ref][Cand])
                    if (Cand != Ref) {mnOut[Ref][Cand] = true; mnOut[Cand][Ref] = true;}
            }
        }
    }
}

void DSegm_NeighborMatrix1 (std::vector<int> vInputMap, std::vector<int> input_idx, int range, int width, std::vector<std::vector<bool> >& mnOut) {
    for (size_t i = 0; i < input_idx.size() ; i++) {
        int x, y;
        int Ref, Cand;
        GetPixelPos(input_idx[i], width, x, y);
        Ref = vInputMap[input_idx[i]];
        if (Ref < 0) continue;

        if (y + range-1) {
            Cand = vInputMap[input_idx[i] - width];
            if (Cand >= 0) {
                if (!mnOut[Ref][Cand])
                    if (Cand != Ref) {mnOut[Ref][Cand] = true; mnOut[Cand][Ref] = true;}
            }
        }
        if (x + range-1) {
            Cand = vInputMap[input_idx[i] - 1];
            if (Cand >= 0) {
                if (!mnOut[Ref][Cand])
                    if (Cand != Ref) {mnOut[Ref][Cand] = true; mnOut[Cand][Ref] = true;}
            }
        }
    }
}



void DSegm_MatchPoints (int idx_ref, int idx_cand, int nSegCnt, std::vector<int>& seg_map, std::vector<int>& seg_list) {
    if (seg_map[idx_ref]) {
        if(seg_list[seg_map[idx_ref]] != seg_list[seg_map[idx_cand]]) {
            int seg_min = min(seg_list[seg_map[idx_ref]], seg_list[seg_map[idx_cand]]);
            int seg_max = max(seg_list[seg_map[idx_ref]], seg_list[seg_map[idx_cand]]);
            for (int i = 0; i < nSegCnt; i++) {
                if (seg_list[i] == seg_max) {
                    if (seg_min) seg_list[i] = seg_min;
                    else printf("ddddddddddddddddddddddddddddddddddd\n");
                }
            }
        }
    }
    else seg_map[idx_ref] = seg_map[idx_cand];
}



void DSegm_MergeSegments(int ref, int cand, bool clear, int n, std::vector<int> &vnLB, std::vector<int> &vnSB, std::vector<bool> &vbCB) {
    if (vnLB[ref] < vnLB[cand]) {
        vnSB[ref] = vnSB[ref] + vnSB[cand];
        for (int i = 1; i < n; i++) {
            if (vnLB[i] == vnLB[cand]) {
                vnLB[i] = vnLB[ref];
                vnSB[i] = vnSB[ref];
                if (clear) vbCB[i] = true;
            }
        }
    }

    else if (vnLB[ref] > vnLB[cand]) {
        vnSB[cand] = vnSB[cand] + vnSB[ref];
        for (int i = 1; i < n; i++) {
            if (vnLB[i] == vnLB[ref]) {
                vnLB[i] = vnLB[cand];
                vnSB[i] = vnSB[cand];
                if (clear) vbCB[i] = true;
            }
        }
    }
}


void DSegmentation (std::vector<float> vnDGrad, std::vector<int> input_idx, int tau_s, float nDSegmGradDist, float nDepthGradNone,
                        int width, int nMapSize, int x_min, int x_max, int y_min, int y_max, std::vector<int> &vnLblMap, std::vector<int> &vnLblMapFinal, int &nSegCnt) {

    int seg_cnt = 1;
    int seg_cnt_final = 1;

    int x, y;
    float nNnDepthGradRef, nDSegmDistGradTmp;
    int nIdxCand = 0;
    int nSegCntTmp = 1, nSegCntTmpMax = 10000;
    std::vector<int> vnPatchMap(nMapSize, 0);
    std::vector<int> vnLblBuffTmp(nSegCntTmpMax, 0);
    std::vector<int> vnSBTmp(nSegCntTmpMax, 0);
    std::vector<float> vnDGrad_local(nMapSize, 1000);
    std::vector<float> vnIdxMissing(input_idx.size(), 0);
    int cnt_miss = 0;

    float dg_high = nDepthGradNone * 0.8;


    for(size_t i = 0 ; i < input_idx.size(); i++) {
        if (vnDGrad[input_idx[i]] > nDepthGradNone*0.1){
            vnIdxMissing[cnt_miss++] = input_idx[i];
            continue;
        }

        nNnDepthGradRef = vnDGrad[input_idx[i]];
        vnDGrad_local[input_idx[i]] = nNnDepthGradRef;
        GetPixelPos(input_idx[i], width, x, y);

        if (y > y_min) {
            nIdxCand = input_idx[i] - width;
            if (vnPatchMap[nIdxCand]) {
                if (nNnDepthGradRef > dg_high || vnDGrad_local[nIdxCand] > dg_high) nDSegmDistGradTmp = nDepthGradNone;
                else {
                    nDSegmDistGradTmp = fabs(nNnDepthGradRef - vnDGrad_local[nIdxCand]);
                    if (nDSegmDistGradTmp < nDSegmGradDist) DSegm_MatchPoints(input_idx[i], nIdxCand, nSegCntTmp, vnPatchMap, vnLblBuffTmp);
                }
            }
        }
        if (x > x_min) {
            nIdxCand = input_idx[i] - 1;
            if (vnPatchMap[nIdxCand]) {
                if (nNnDepthGradRef > dg_high || vnDGrad_local[nIdxCand] > dg_high) nDSegmDistGradTmp = nDepthGradNone;
                else {
                    nDSegmDistGradTmp = fabs(nNnDepthGradRef - vnDGrad_local[nIdxCand]);
                    if (nDSegmDistGradTmp < nDSegmGradDist) DSegm_MatchPoints(input_idx[i], nIdxCand, nSegCntTmp, vnPatchMap, vnLblBuffTmp);
                }
            }
        }
        if (!vnPatchMap[input_idx[i]]) {
            vnPatchMap[input_idx[i]] = nSegCntTmp;
            vnLblBuffTmp[nSegCntTmp] = nSegCntTmp; nSegCntTmp++;
            if (nSegCntTmp > nSegCntTmpMax -1) {nSegCntTmp = nSegCntTmpMax; printf("ffff %d %d %d\n", (int)i, input_idx[i], nSegCntTmp); break;}
        }

        if (x < x_min && x > x_max && y < y_min && y > y_max) printf("--------------- %d %d %d %d %d %d %d\n", input_idx[i], x, y, x_min, x_max, y_min, y_max);
        vnSBTmp[vnPatchMap[input_idx[i]]]++;
    }
    vnLblBuffTmp.resize(nSegCntTmp);
    vnIdxMissing.resize(cnt_miss);



    ///////////////////////////////////////////////
    /////**   Reordering Label Buffer   **/////////
    /////**   Updating Size Buffer      **/////////
    ///////////////////////////////////////////////
    std::vector<int> vnLB(nSegCntTmp, 0);
    std::vector<int> vnSB(nSegCntTmp, 0);
    for (int i = 1; i < nSegCntTmp; i++){
        if (vnLblBuffTmp[i] == i) {
            vnSB[seg_cnt] = vnSBTmp[i];
            vnLB[i] = seg_cnt++;
        }
        else {
            vnLB[i] = vnLB[vnLblBuffTmp[i]];
            vnSB[vnLB[i]] += vnSBTmp[i];
        }
    }
    //vnLB.resize(seg_cnt);
    //vnSB.resize(seg_cnt);



    //for (int i = 1; i < seg_cnt; i++) if (!vnSB[i]) printf("Error the size of the segment %d is zero\n", i, vnSB[i]);

//    for (int i = 1; i < nSegCntTmp; i++) printf("%3d %3d %6d %3d\n", i, vnLB[i], vnSB[i], nSegCntTmp);
//    printf("\n");


    if (seg_cnt < 3) {
        for (int i = 0; i < cnt_miss; i++) vnLblMap[vnIdxMissing[i]] = 1;
        seg_cnt_final = 2;
    }
    else {
        ////// Generation of temporary Labeled Segment Map
        for (size_t i = 0; i < input_idx.size(); i++)
            vnLblMap[input_idx[i]] = vnLB[vnPatchMap[input_idx[i]]];



        ///////////////////////////////////////////////
        /////**                             **/////////
        /////**      Postprocessing         **/////////
        /////**                             **/////////
        ///////////////////////////////////////////////


        std::vector<std::vector<bool> > mbNeighbor(seg_cnt, std::vector<bool>(seg_cnt, false));
        DSegm_NeighborMatrix(vnLblMap, input_idx, 1, width, mbNeighbor);


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        vnLB.assign(seg_cnt, 0);
        std::vector<bool> vbCB(seg_cnt, false);

        for (int i = 1; i < seg_cnt; i++) {
            vnLB[i] = i;
            if (vnSB[i] > tau_s) vbCB[i] = true;
        }


        for (int i = 1; i < seg_cnt; i++) {
            if (vbCB[i]) continue;

            int max_n = 0, max_size = 0;
            bool clear = false;
            for (int j = 1; j < seg_cnt; j++) {
                if (i == j) continue;
                if (mbNeighbor[i][j]) {
                    if (vbCB[j]) {
                        if (vnSB[vnLB[j]] > max_size) {
                            max_n = j;
                            max_size = vnSB[vnLB[j]];
                        }
                    }
                    else
                        DSegm_MergeSegments(i, j, clear, seg_cnt, vnLB, vnSB, vbCB);
                }
            }

            if (max_n) {
                clear = true;
                DSegm_MergeSegments(i, max_n, clear, seg_cnt, vnLB, vnSB, vbCB);
                vbCB[i] = true;
            }
        }


        for (int i = 1; i < seg_cnt; i++){
            if (vnLB[i] == i) {
                vnLB[i] = seg_cnt_final++;
            }
            else vnLB[i] = vnLB[vnLB[i]];
        }

        for (size_t i = 0; i < input_idx.size(); i++)
            vnLblMapFinal[input_idx[i]] = vnLB[vnLblMap[input_idx[i]]];
    }

    if (seg_cnt_final) nSegCnt = seg_cnt_final - 1;
    if (!seg_cnt_final) printf("----------------------Error\n");
}


void TrackingPre (int nSegCnt, int nDSegmCutSize, int nCvWidth, int nCvHeight, std::vector<float> vnX, std::vector<float> vnY, std::vector<float> vnZ, cv:: Mat cvm_rgb_ds, int nTrackClrMode, int nTrackHistoBin, int nTrackCntMem, int nTrackCntStable, int nTrackCntDisap,
                  std::vector<int> &vnTmpPtsCnt, std::vector<std::vector<int> > &mnTmpPtsIdx, std::vector<std::vector<int> > &mnTmpRect, std::vector<std::vector<int> > &mnTmpRCenter,
                  std::vector<std::vector<float> > &mnTmpCubic, std::vector<std::vector<float> > &mnTmpCCenter, std::vector<int> &vnTmpLength, std::vector<std::vector<float> > &mnTmpClrHist,
                  std::vector<int> &vnTmpMemoryCnt, std::vector<int> &vnTmpStableCnt, std::vector<int> &vnTmpDisapCnt, int &nTrkSegCnt) {
    int final_cnt_new = 0;
    for (int i = 1; i < nSegCnt; i++) {
        if (vnTmpPtsCnt[i] < nDSegmCutSize) continue;
        int rx_acc = 0, ry_acc = 0;
        int rx_min = nCvWidth, rx_max = 0, ry_min = nCvHeight, ry_max = 0;
        float cx_acc = 0, cy_acc = 0, cz_acc = 0;
        float cx_min = 99, cx_max = -99, cy_min = 99, cy_max = -99, cz_min = 99, cz_max = -99;
        for (int j = 0; j < vnTmpPtsCnt[i]; j++) {
            int rx_tmp, ry_tmp;
            GetPixelPos(mnTmpPtsIdx[i][j], nCvWidth, rx_tmp, ry_tmp);
            rx_acc += rx_tmp; ry_acc += ry_tmp;
            if (rx_tmp < rx_min) rx_min = rx_tmp;
            if (rx_tmp > rx_max) rx_max = rx_tmp;
            if (ry_tmp < ry_min) ry_min = ry_tmp;
            if (ry_tmp > ry_max) ry_max = ry_tmp;

            float cx_tmp = vnX[mnTmpPtsIdx[i][j]], cy_tmp = vnY[mnTmpPtsIdx[i][j]], cz_tmp = vnZ[mnTmpPtsIdx[i][j]];
            cx_acc += cx_tmp; cy_acc += cy_tmp; cz_acc += cz_tmp;
            if (cx_tmp < cx_min) cx_min = cx_tmp;
            if (cx_tmp > cx_max) cx_max = cx_tmp;
            if (cy_tmp < cy_min) cy_min = cy_tmp;
            if (cy_tmp > cy_max) cy_max = cy_tmp;
            if (cz_tmp < cz_min) cz_min = cz_tmp;
            if (cz_tmp > cz_max) cz_max = cz_tmp;
        }
        mnTmpPtsIdx[final_cnt_new] = mnTmpPtsIdx[i];
        vnTmpPtsCnt[final_cnt_new] = vnTmpPtsCnt[i];
        mnTmpRCenter[final_cnt_new][0] = rx_acc / vnTmpPtsCnt[i];
        mnTmpRCenter[final_cnt_new][1] = ry_acc / vnTmpPtsCnt[i];
        mnTmpCCenter[final_cnt_new][0] = cx_acc / vnTmpPtsCnt[i];
        mnTmpCCenter[final_cnt_new][1] = cy_acc / vnTmpPtsCnt[i];
        mnTmpCCenter[final_cnt_new][2] = cz_acc / vnTmpPtsCnt[i];
        mnTmpRect[final_cnt_new][0] = rx_min;
        mnTmpRect[final_cnt_new][1] = ry_min;
        mnTmpRect[final_cnt_new][2] = rx_max;
        mnTmpRect[final_cnt_new][3] = ry_max;
        mnTmpCubic[final_cnt_new][0] = cx_min;
        mnTmpCubic[final_cnt_new][1] = cy_min;
        mnTmpCubic[final_cnt_new][2] = cz_min;
        mnTmpCubic[final_cnt_new][3] = cx_max;
        mnTmpCubic[final_cnt_new][4] = cy_max;
        mnTmpCubic[final_cnt_new][5] = cz_max;
        vnTmpLength[final_cnt_new] = int(sqrt(pow((cx_max - cx_min), 2) + pow((cy_max - cy_min), 2)+ pow((cz_max - cz_min), 2)) * 1000);

        vnTmpMemoryCnt[final_cnt_new] = nTrackCntMem - nTrackCntStable;
        vnTmpStableCnt[final_cnt_new] = 0;
        vnTmpDisapCnt[final_cnt_new] = nTrackCntDisap + 10;
        Calc3DColorHistogram(cvm_rgb_ds, mnTmpPtsIdx[final_cnt_new], nTrackHistoBin, mnTmpClrHist[final_cnt_new]);

        final_cnt_new++;
    }
    nTrkSegCnt = final_cnt_new;
}


void TrackingAAA (int seg, int j_min, int cnt_old, int nObjsNrLimit, float nTrackDist, float huge, std::vector<int> &vnSegCandQtt, std::vector<int> &vnMemCandQtt, std::vector<int> &vnMemCandMin, std::vector<int> &vnMatchedSeg, std::vector<std::vector<float> > &mnDistTmp) {
    for (int j = 0; j < cnt_old; j++) {
        if (j == j_min) continue;
        if (mnDistTmp[seg][j] > nTrackDist) continue;

        mnDistTmp[seg][j] = huge;
        vnMemCandQtt[j]--;
        vnSegCandQtt[seg]--;
        if (!vnMemCandQtt[j]) vnMemCandMin[j] = nObjsNrLimit;

        //// if there is only one element in the colum left
        if (vnMemCandQtt[j] != 1) continue;
        for (int ii = 0; ii < seg; ii++) {
            if (mnDistTmp[ii][j] > nTrackDist) continue;
            vnMatchedSeg[ii] = j;
            if (vnSegCandQtt[ii] > 1) TrackingAAA (ii, j, cnt_old, nObjsNrLimit, nTrackDist, huge, vnSegCandQtt, vnMemCandQtt, vnMemCandMin, vnMatchedSeg, mnDistTmp);
        }
    }
}



void Tracking(int nTrkSegCnt, int nObjsNrLimit, double dp_dia, int nTrackMode, double nTrackDPos, double nTrackDSize, double nTrackDRgb, double nTrackPFac, double nTrackSFac, double nTrackCFac, double nTrackDist, int nTrackCntMem, int nTrackCntStable, int nTrackCntDisap, int bin,
              std::vector<int> vnTmpPtsCnt, std::vector<std::vector<int> > mnTmpPtsIdx, std::vector<std::vector<int> > mnTmpRect, std::vector<std::vector<int> > mnTmpRCenter, std::vector<std::vector<float> > mnTmpCubic, std::vector<std::vector<float> > mnTmpCCenter, std::vector<int> vnTmpLength, std::vector<std::vector<float> > mnTmpClrHist, std::vector<int> vnTmpMemoryCnt, std::vector<int> vnTmpStableCnt, std::vector<int> vnTmpDisapCnt,
              std::vector<int> &vnProtoIdx, std::vector<int> &vnProtoPtsCnt, std::vector<std::vector<int> > &mnProtoPtsIdx, std::vector<std::vector<int> > &mnProtoRect, std::vector<std::vector<int> > &mnProtoRCenter, std::vector<std::vector<float> > &mnProtoCubic, std::vector<std::vector<float> > &mnProtoCCenter, std::vector<int> &vnProtoLength, std::vector<std::vector<float> > &mnProtoClrHist,
              std::vector<int> &vnProtoFound, std::vector<int> &vnProtoMemoryCnt, std::vector<int> &vProtoStableCnt, std::vector<int> &vProtoDisapCnt, int &nProtoCnt, bool flag_mat) {


    int cnt_new = 0, cnt_old = nProtoCnt;
    std::vector<int> objs_new_idx(nTrkSegCnt, 0);

    float huge = 100;
    int nDim = max(nTrkSegCnt, cnt_old);
    std::vector<std::vector<float> > mnDistClr(nDim, std::vector<float>(nDim, huge));
    std::vector<std::vector<float> > mnDistTotal(nDim, std::vector<float>(nDim, huge));
    std::vector<std::vector<float> > mnDistTmp(nDim, std::vector<float>(nDim, huge));


    if (nTrkSegCnt && cnt_old) {
        for (int i = 0; i < nTrkSegCnt; i++) {
            //int size_ref = vnTmpPtsCnt[i];
            for (int j = 0; j < cnt_old; j++) {
                //int xc = mnTmpRCenter[i][0], yc = mnTmpRCenter[i][1];
                //float dp = sqrt(pow(xc - mnProtoRCenter[j][0], 2) + pow(yc - mnProtoRCenter[j][1], 2))/dp_dia;
                float xc = mnTmpCCenter[i][0], yc = mnTmpCCenter[i][1], zc = mnTmpCCenter[i][2];
                float dp = sqrt(pow(xc - mnProtoCCenter[j][0], 2) + pow(yc - mnProtoCCenter[j][1], 2) + pow(zc - mnProtoCCenter[j][2], 2));
                //float ds = (float)abs(vnProtoPtsCnt[j] - size_ref)/max(vnProtoPtsCnt[j], size_ref);
                float ds = (float)abs(vnProtoLength[j] - vnTmpLength[i])/max(vnProtoLength[j], vnTmpLength[i]);
                float dc = 0;
                for (int ii = 0; ii < bin; ii++) dc += fabs(mnProtoClrHist[j][ii] - mnTmpClrHist[i][ii]);
                mnDistClr[i][j] = dp;
                //printf("%2d %2d %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f\n", i, j, dp, xc, yc, zc, mnProtoCCenter[j][0], mnProtoCCenter[j][0], mnProtoCCenter[j][0]);

                if (dp < nTrackDPos && ds < nTrackDSize && dc < nTrackDRgb)
                    mnDistTotal[i][j] = nTrackPFac * dp + nTrackSFac * ds + nTrackCFac * dc;
            }
            //printf("\n");
        }

        if (flag_mat) {
            char sText[128];
            std::ofstream finn1;
            std::string filename = "matrix.txt";

            finn1.open(filename.data(), std::ios::app);

            sprintf(sText, "%dx%d\n", nTrkSegCnt, cnt_old);
            finn1<< sText;
            for (int i = 0; i < nDim; i++) {if (!i) sprintf(sText, "%11d", i); else sprintf(sText, "%8d", i); finn1<< sText;} finn1<< "\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistTotal[i][j]); else sprintf(sText, "%8.3f", mnDistTotal[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n\n";

            for (int i = 0; i < nDim; i++) {if (!i) sprintf(sText, "%11d", i); else sprintf(sText, "%8d", i); finn1<< sText;} finn1<< "\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistClr[i][j]); else sprintf(sText, "%8.3f", mnDistClr[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n\n";
            finn1.close();
        }
    }






    ////////////// Tracking with optimization ////////////////////////////////////
    float offset = 0.01;
    std::vector<int> vnSegCandQtt(nTrkSegCnt, 0);
    std::vector<int> vnSegCandMin(nTrkSegCnt, nObjsNrLimit);
    std::vector<int> vnMemCandQtt(cnt_old, 0);
    std::vector<int> vnMemCandMin(cnt_old, nObjsNrLimit);
    std::vector<int> vnMatchedSeg(nTrkSegCnt, nObjsNrLimit*2);
    std::vector<int> vnMatchedMem(cnt_old, nObjsNrLimit*2);

    mnDistTmp = mnDistTotal;

    for (int i = 0; i < nTrkSegCnt; i++) {
        float j_min = huge;
        for (int j = 0; j < cnt_old; j++) {
            if (mnDistTmp[i][j] >= nTrackDist) {mnDistTmp[i][j] = huge; continue;}

            vnSegCandQtt[i]++;
            vnMemCandQtt[j]++;

            if (mnDistTmp[i][j] > j_min) continue;
            if (mnDistTmp[i][j] == j_min) mnDistTmp[i][j] += offset;
            else {j_min = mnDistTmp[i][j]; vnSegCandMin[i] = j;}
        }
    }

    for (int j = 0; j < cnt_old; j++) {
        float i_min = huge;
        for (int i = 0; i < nTrkSegCnt; i++) {
            if (mnDistTmp[i][j] >= nTrackDist) continue;

            if (mnDistTmp[i][j] > i_min) continue;
            if (mnDistTmp[i][j] == i_min) mnDistTmp[i][j] += offset;
            else {
                i_min = mnDistTmp[i][j];
                vnMemCandMin[j] = i;
            }
        }
    }

    for (int i = 0; i < nTrkSegCnt; i++) {
        if (vnSegCandQtt[i]) {
            //// if no other initial elements in the column
            if (vnMemCandQtt[vnSegCandMin[i]] < 2) {
                if (vnMatchedSeg[i] < nObjsNrLimit) printf("Error, the segment %d is already matched %d\n", i, vnSegCandMin[i]);
                vnMatchedSeg[i] = vnSegCandMin[i];

                if (vnSegCandQtt[i] > 1) TrackingAAA (i, vnSegCandMin[i], cnt_old, nObjsNrLimit, nTrackDist, huge, vnSegCandQtt, vnMemCandQtt, vnMemCandMin, vnMatchedSeg, mnDistTmp);
            }
        }
        else {
            vnMatchedSeg[i] = nObjsNrLimit;
        }
    }




    ////////////// Optimaization //////////////////////////////////////////////
    if (!nTrackMode) {
        std::vector<int> idx_seg;
        int cnt_nn = 0;
        for (int i = 0; i < nTrkSegCnt; i++) {
            if (vnMatchedSeg[i] > nObjsNrLimit) {
                for (int j = 0; j < cnt_old; j++) {
                    if (mnDistTmp[i][j] < nTrackDist) {
                        vnMatchedMem[j] = 0;
                    }
                }
                idx_seg.resize(cnt_nn + 1);
                idx_seg[cnt_nn++] = i;
            }
        }

        if (cnt_nn) {
            std::vector<int> idx_mem;
            cnt_nn = 0;
            for (int j = 0; j < cnt_old; j++) {
                if (!vnMatchedMem[j]) {
                    idx_mem.resize(cnt_nn + 1);
                    idx_mem[cnt_nn++] = j;
                }
            }


            int munkres_huge = 100000;
            int nDimMunkres = max((int)idx_seg.size(), (int)idx_mem.size());
            MunkresMatrix<double> m_MunkresIn(nDimMunkres, nDimMunkres);
            MunkresMatrix<double> m_MunkresOut(nDimMunkres, nDimMunkres);

            for (size_t i = 0; i < idx_seg.size(); i++) {
                for (size_t j = 0; j < idx_mem.size(); j++) {
                    if(mnDistTmp[idx_seg[i]][idx_mem[j]]) m_MunkresIn(i,j) = 100/mnDistTmp[idx_seg[i]][idx_mem[j]];
                    else m_MunkresIn(i,j) = (double)munkres_huge;
                }
            }

            if (idx_mem.size() > idx_seg.size()) {
                for (int i = (int)idx_seg.size(); i < nDimMunkres; i++) {
                    for (int j = 0; j < nDimMunkres; j++) m_MunkresIn(i,j) = rand()%10 +1;
                }
            }
            if (idx_mem.size() < idx_seg.size()) {
                for (int j = (int)idx_mem.size(); j < nDimMunkres; j++) {
                    for (int i = 0; i < nDimMunkres; i++) m_MunkresIn(i,j) = rand()%10 +1;
                }

            }

            m_MunkresOut = m_MunkresIn;

            Munkres m;
            m.solve(m_MunkresOut);


            for (int i = 0; i < nDimMunkres; i++) {
                int rowcount = 0;
                for (int j = 0; j < nDimMunkres; j++) if (m_MunkresOut(i,j) == 0) rowcount++;
                if (rowcount != 1) std::cerr << "Row " << i << " has " << rowcount << " columns that have been matched." << std::endl;
            }
            for (int j = 0; j < nDimMunkres; j++) {
                int colcount = 0;
                for (int i = 0; i < nDimMunkres; i++) if (m_MunkresOut(i,j) == 0) colcount++;
                if (colcount != 1) std::cerr << "Column " << j << " has " << colcount << " rows that have been matched." << std::endl;
            }




            for (size_t i = 0; i < idx_seg.size(); i++) {
                for (size_t j = 0; j < idx_mem.size(); j++) {
                    if (m_MunkresOut(i,j) == 0) vnMatchedSeg[idx_seg[i]] = idx_mem[j];
                    else mnDistTmp[idx_seg[i]][idx_mem[j]] = huge;
                }
            }




            if (flag_mat) {
                char sText[128];
                std::ofstream finn1;
                std::string filename = "matrix.txt";

                finn1.open(filename.data(), std::ios::app);

                sprintf(sText, "%dx%d\n", nTrkSegCnt, cnt_old);
                finn1<< sText;
                for (int i = 0; i < nDim; i++) {if (!i) sprintf(sText, "%11d", i); else sprintf(sText, "%8d", i); finn1<< sText;} finn1<< "\n";
                for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.4f", i, m_MunkresOut(i,j)); else sprintf(sText, "%8.4f", m_MunkresOut(i,j)); finn1<< sText;} finn1<< "\n";} finn1<< "\n\n";
                finn1.close();
            }
        }
    }
    ////////////// End of the otimization for tracking ////////////////////////////////////



    std::vector<bool> objs_old_flag(cnt_old, false);
    for (int i = 0; i < nTrkSegCnt; i++) {
        int cand = 0, j_tmp = 500000;
        /////////////// Past tracking ///////////////////////////
        if (nTrackMode) {
            float d_tmp = 50000000;
            for (int j = 0; j < cnt_old; j++) {
                if (mnDistTotal[i][j] >= nTrackDist) continue;

                if (mnDistTotal[i][j] < d_tmp) {
                    d_tmp = mnDistTotal[i][j];
                    j_tmp = j;
                }
                cand++;
            }
        }
        /////////////// End of past tracking ///////////////////
        else {
            if (vnMatchedSeg[i] < nObjsNrLimit) {
                cand = 1;
                j_tmp = vnMatchedSeg[i];
            }
        }

        if (cand) {
            objs_old_flag[j_tmp] = true;
            mnProtoPtsIdx[j_tmp] = mnTmpPtsIdx[i];
            vnProtoPtsCnt[j_tmp] = vnTmpPtsCnt[i];
            mnProtoRCenter[j_tmp] = mnTmpRCenter[i];
            mnProtoCCenter[j_tmp] = mnTmpCCenter[i];
            mnProtoRect[j_tmp] = mnTmpRect[i];
            mnProtoCubic[j_tmp] = mnTmpCubic[i];
            vnProtoLength[j_tmp] = vnTmpLength[i];
            mnProtoClrHist[j_tmp] = mnTmpClrHist[i];
        }
        else {
            objs_new_idx[cnt_new++] = i;
        }
    }







    for (int i = 0; i < cnt_old; i++) {
        if (objs_old_flag[i]) {
            vnProtoMemoryCnt[i]++;
            if(vnProtoMemoryCnt[i] < nTrackCntMem - nTrackCntStable + 2) vnProtoMemoryCnt[i] = nTrackCntMem - nTrackCntStable + 1;
            if(vnProtoMemoryCnt[i] > 100*nTrackCntMem) vnProtoMemoryCnt[i] = 100*nTrackCntMem;
            vProtoStableCnt[i]++;
            if(vProtoStableCnt[i] > 100*nTrackCntStable) vProtoStableCnt[i] = 100*nTrackCntStable;
            vProtoDisapCnt[i] = 0;
        }
        else {
            vnProtoMemoryCnt[i]--;
            vProtoDisapCnt[i]++;
            if(vProtoDisapCnt[i] > 100*nTrackCntStable) vProtoDisapCnt[i] = 100*nTrackCntStable;
            if (vnProtoMemoryCnt[i] >= nTrackCntMem) vnProtoMemoryCnt[i] = nTrackCntMem;
            vProtoStableCnt[i] = 0;
        }
    }
    int cnt_tmp = 0;
    for (int i = 0; i < cnt_old; i++) {
        if (vnProtoMemoryCnt[i] < 0) continue;
        vProtoStableCnt[cnt_tmp] = vProtoStableCnt[i];
        vProtoDisapCnt[cnt_tmp] = vProtoDisapCnt[i];
        vnProtoMemoryCnt[cnt_tmp] = vnProtoMemoryCnt[i];
        mnProtoPtsIdx[cnt_tmp] = mnProtoPtsIdx[i];
        vnProtoPtsCnt[cnt_tmp] = vnProtoPtsCnt[i];
        mnProtoRCenter[cnt_tmp] = mnProtoRCenter[i];
        mnProtoCCenter[cnt_tmp] = mnProtoCCenter[i];
        mnProtoRect[cnt_tmp] = mnProtoRect[i];
        mnProtoCubic[cnt_tmp] = mnProtoCubic[i];
        vnProtoLength[cnt_tmp] = vnProtoLength[i];
        mnProtoClrHist[cnt_tmp] = mnProtoClrHist[i];
        vnProtoFound[cnt_tmp] = vnProtoFound[i];
        vnProtoIdx[cnt_tmp++] = vnProtoIdx[i];
    }
    cnt_old = cnt_tmp;


    std::vector<int> proto_idx(cnt_old, 0);
    std::vector<int> proto_idx_new(cnt_new, 0);
    for (int i = 0; i < cnt_old; i++) proto_idx[i] = vnProtoIdx[i];
    std::sort(proto_idx.begin(), proto_idx.end());
    //for (int i = 0; i < cnt_old; i++) printf("%2d ", proto_idx[i]); printf("%4d\n", cnt_old);
    int cnt_tmp_tmp = 0;
    if (cnt_old > 2 && cnt_new) {
        if (proto_idx[2] > proto_idx[1]) {
            for (int i = 2; i < cnt_old; i++) {
                int diff = proto_idx[i] - proto_idx[i-1];
                if (diff > 1) {
                    for (int j = 0; j < diff-1; j++) {
                        proto_idx_new[cnt_tmp_tmp++] = proto_idx[i-1] + j+1;
                        if (cnt_tmp_tmp >= cnt_new) break;
                    }
                }
                if (cnt_tmp_tmp >= cnt_new) break;
            }
            if (cnt_tmp_tmp < cnt_new) {
                for (int i = 0; i < cnt_new - cnt_tmp_tmp; i++) {
                    proto_idx_new[cnt_tmp_tmp + i] = proto_idx[cnt_old-1] + i+1;
                }
            }
        }
    }
    else for (int i = 0; i < cnt_new; i++) proto_idx_new[i] = i;
    //for (int i = 0; i < cnt_new; i++) printf("%2d ", proto_idx_new[i]); printf("%4d\n\n", cnt_new);


    for (int i = 0; i < cnt_new; i++) {
        vProtoStableCnt[cnt_old + i] = vnTmpStableCnt[objs_new_idx[i]];
        vProtoDisapCnt[cnt_old + i] = vnTmpDisapCnt[objs_new_idx[i]];
        vnProtoMemoryCnt[cnt_old + i] = vnTmpMemoryCnt[objs_new_idx[i]];
        mnProtoPtsIdx[cnt_old + i] = mnTmpPtsIdx[objs_new_idx[i]];
        vnProtoPtsCnt[cnt_old + i] = vnTmpPtsCnt[objs_new_idx[i]];
        mnProtoRCenter[cnt_old + i] = mnTmpRCenter[objs_new_idx[i]];
        mnProtoCCenter[cnt_old + i] = mnTmpCCenter[objs_new_idx[i]];
        mnProtoRect[cnt_old + i] = mnTmpRect[objs_new_idx[i]];
        mnProtoCubic[cnt_old + i] = mnTmpCubic[objs_new_idx[i]];
        vnProtoLength[cnt_old + i] = vnTmpLength[objs_new_idx[i]];
        mnProtoClrHist[cnt_old + i] = mnTmpClrHist[objs_new_idx[i]];
        vnProtoIdx[cnt_old + i] = proto_idx_new[i];
    }
    nProtoCnt = cnt_old + cnt_new;

    if (nProtoCnt >= nObjsNrLimit) printf("Object queue exceeds object no. limit %d\n", nObjsNrLimit);
}



