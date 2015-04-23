/*
 * Functions for Segmentation and tracking
 */

#include <fstream> // for debugging

#include <opencv2/imgproc/imgproc.hpp>

#include "ni/legacy/func_segmentation.h"
#include "ni/legacy/func_operations.h"

#include "ni/3rdparty/munkres/munkres.hpp"

using std::min;
using std::max;

int FuncFindPos(const std::vector<int> &idx_vector, int value) {

    for (size_t i=0; i < idx_vector.size(); i++) {

        if (idx_vector[i] == value) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

void Segm_FlatDepthGrad(const cv::Mat &cvm_input,
                        const std::vector<int> &index,
                        float max, float min,
                        int mode,
                        int scenter,
                        int sband1, int sband2,
                        float factor,
                        std::vector<float> &output_blur,
                        std::vector<float> &output_ct) {

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

void Segm_SmoothDepthGrad(const std::vector<float> &vInput,
                          const std::vector<int> &index,
                          const cv::Size &size,
                          float max, float min,
                          float none,
                          int fmode,
                          int fsize,
                          int smode,
                          int scenter,
                          int sband1, int sband2,
                          float clfac,
                          std::vector<float> &output_blur,
                          std::vector<float> &output_ct) {

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
    //Segm_FlatDepthGrad_old (input_gray_blur, index, size.width, size.height, max, min, smode, scenter, sband1, clfac, output_blur, output_ct);
    Segm_FlatDepthGrad(input_gray_blur, index, max, min, smode, scenter, sband1, sband2, clfac, output_blur, output_ct);
    input_gray.release(); input_gray_blur.release();
}

void Segm_NeighborMatrix(const std::vector<int> &vInputMap,
                         const std::vector<int> &input_idx,
                         int range,
                         int width,
                         std::vector<std::vector<bool> >& mnOut) {

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




/* Creating matrix about information of adjacent segments (for saliency program)
*
* Input:
* vInputMap - input map
* input idx - indices of input map which should be processed
* range - distance of adjacent segments
* width - width of input map
*
* Output:
* mnOut - neighborhood matrix
*/
void Segm_NeighborMatrix1 (std::vector<int> vInputMap, std::vector<int> input_idx, int range, int width, std::vector<std::vector<bool> >& mnOut) {
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

         
/* Checking if two points are in the same segment
 *
 * Input:
 * idx_ref - reference point
 * idx_cand - candidate point
 * nSurfCnt - number of segments
 *
 * Output:
 * seg_map - segment map (which maps every point to the segment number it lies in)
 * seg_list - segment buffer */
void Segm_MatchPoints (int idx_ref, int idx_cand, int nSurfCnt, std::vector<int>& seg_map, std::vector<int>& seg_list) {
    
    /* if not zero (not assigned?)
     * then:
     *      see below
     * else:
     *  assign candidate to ref, ref now belongs to candidate's segment
     */
     if (seg_map[idx_ref]) {
		/* reference point already assigned to a segment
         * if reference and candidate belong to different segments
         * @todo: Why can't we compare seg_map[idx_ref] != seg_map[idx_cand] directly?
         *
         * seg_list contains values > 0.
         *
         * get max(both point's segments)
         * get min(both point's segments)
         * replace each occurence of max with min
         * @todo: Why?
         */
        if(seg_list[seg_map[idx_ref]] != seg_list[seg_map[idx_cand]]) {
            int seg_min = min(seg_list[seg_map[idx_ref]], seg_list[seg_map[idx_cand]]);
            int seg_max = max(seg_list[seg_map[idx_ref]], seg_list[seg_map[idx_cand]]);
            for (int i = 0; i < nSurfCnt; i++) {
                if (seg_list[i] == seg_max) {
                    if (seg_min) seg_list[i] = seg_min;
                    else printf("ddddddddddddddddddddddddddddddddddd\n");
                }
            }
        }
    }
    else seg_map[idx_ref] = seg_map[idx_cand];
}




/* Merging two segments
 * 
 * 		Segments are not merged in the sense of reduced to a single entry
 * 		but rather info of the merged segment is duplicated.
 * Input:
 * ref - reference segment
 * cand - candidate segment
 * clear - flag for merging (true, then merging)
 * n - number segments
 *
 * Output:
 * vnLB - label buffer, contains segments labels
 * vnSB - size buffer, contains segments sizes
 * vbCB - clearing buffer, contains merging flags (indicate if a segment was merged)
 */
void Segm_MergeSegments(int ref, int cand, bool clear, int n, std::vector<int> &vnLB, std::vector<int> &vnSB, std::vector<bool> &vbCB) {
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




/* Main segmentation function
 *
 * Input:
 * vnDGrad - depth-gradient map
 * input_idx - indices of the pixel which should be processed
 * tau_s - threshold for size differences
 * nSegmGradDist - threshold for segmentation (if depth-gradient between two pixel is greater than threshold, they get segmented)
 * nDepthGradNone -  constant for invalid depth-gradient
 * width - width of depth-gradient map
 * nMapSize - size of depth-gradient map
 * x_min, x_max, y_min, y_max - coordinates of the area of the map which should be processed
 *
 * Output:
 * vnLblMap - segment map before post-processing
 * vnLblMapFinal - segment map
 * nSurfCnt - number of segments
 */
void Segmentation (std::vector<float> vnDGrad, std::vector<int> input_idx, int tau_s, float nSegmGradDist, float nDepthGradNone,
                        int width, int nMapSize, int x_min, int x_max, int y_min, int y_max, std::vector<int> &vnLblMap, std::vector<int> &vnLblMapFinal, int &nSurfCnt) {

    int seg_cnt = 1;
    int seg_cnt_final = 1;

    int x, y;
    float nNnDepthGradRef, nSegmDistGradTmp;
    int nIdxCand = 0;
    int nSurfCntTmp = 1, nSurfCntTmpMax = 10000;
    std::vector<int> vnPatchMap(nMapSize, 0);
    std::vector<int> vnLblBuffTmp(nSurfCntTmpMax, 0);
    std::vector<int> vnSBTmp(nSurfCntTmpMax, 0);
    std::vector<float> vnDGrad_local(nMapSize, 1000);
    std::vector<float> vnIdxMissing(input_idx.size(), 0);
    int cnt_miss = 0;

    float dg_high = nDepthGradNone * 0.8;


    for(size_t i = 0 ; i < input_idx.size(); i++) {

        // begin filter of indices at which gradient exceeds upper threshold

        // if gradient value at valid index is > 10% * nDepthGradNone
        if (vnDGrad[input_idx[i]] > nDepthGradNone*0.1){

            // add index to list of misses
            // increment number of missed indices
            vnIdxMissing[cnt_miss++] = input_idx[i];
            continue; // skip iteration altogether
        }

        // end filter

        // in the case of the iteration not skipped

        nNnDepthGradRef = vnDGrad[input_idx[i]];
        vnDGrad_local[input_idx[i]] = nNnDepthGradRef;
        GetPixelPos(input_idx[i], width, x, y);

        if (y > y_min) {
            nIdxCand = input_idx[i] - width; // one row above

            // at first this condition is never true, see below *
            if (vnPatchMap[nIdxCand]) {

                /* assign value to nDSegmDistGradTmp for this iteration
                 *
                 * if depth gradient at index is > 80% * nDepthGradNone
                 * OR
                 * if depth gradient at index (one row up = y-1) is > 80% * nDepthGradNone
                 *      - false the first time around because nDGrad_local initialized with 1000 everywhere
                 *
                 * then: nDSegmDistGradTmp is nDepthGradNone
                 *
                 * else (see below)
                 */
                if (nNnDepthGradRef > dg_high || vnDGrad_local[nIdxCand] > dg_high) nSegmDistGradTmp = nDepthGradNone;
                else {

                    /* else then:
                     *  nDSegmDistGradTmp = |gradient - gradient(y-1)|
                     * first time around gradient(y-1) == 1000
                     */
                    nSegmDistGradTmp = fabs(nNnDepthGradRef - vnDGrad_local[nIdxCand]);

                    /* if abs|difference| < nDSegmGradDist
                     * then:
                     *  check if (current point) and point (one row above)
                     *  are in the same segment
                     *  this call modifies: vnPatchMap, vnLblBuffTmp
                     *          modifies only vnPatchMap if input_idx[i] not assigned to a segment yet,
                     *          by assigning it to candidates's segment
                     *          if it's already assigned
                     *          then: see DSegm_MatchPoints, swaps segment values, @todo why?
                     */
                    if (nSegmDistGradTmp < nSegmGradDist) Segm_MatchPoints(input_idx[i], nIdxCand, nSurfCntTmp, vnPatchMap, vnLblBuffTmp);
                }
            }
        }
        if (x > x_min) {

            /* same logic as y but comparing to left neighbor
             */
            nIdxCand = input_idx[i] - 1;
            if (vnPatchMap[nIdxCand]) {
                if (nNnDepthGradRef > dg_high || vnDGrad_local[nIdxCand] > dg_high) nSegmDistGradTmp = nDepthGradNone;
                else {
                    nSegmDistGradTmp = fabs(nNnDepthGradRef - vnDGrad_local[nIdxCand]);
                    if (nSegmDistGradTmp < nSegmGradDist) Segm_MatchPoints(input_idx[i], nIdxCand, nSurfCntTmp, vnPatchMap, vnLblBuffTmp);
                }
            }
        }

        // at first vnPatchMap is all-zeros
        if (!vnPatchMap[input_idx[i]]) {

            // nSegCntTmp is a counter that starts with 1
            vnPatchMap[input_idx[i]] = nSurfCntTmp;  // assign segment count to patch for this index, not sure what this means yet.
            vnLblBuffTmp[nSurfCntTmp] = nSurfCntTmp; nSurfCntTmp++; // assign segment count to vnLblBuffTmp, not sure what this means yet.
            // if segment counter exceeds max, cap count at max value (=10000, pretty high)
            if (nSurfCntTmp > nSurfCntTmpMax -1) {nSurfCntTmp = nSurfCntTmpMax; printf("ffff %d %d %d\n", (int)i, input_idx[i], nSurfCntTmp); break;}
        }

        if (x < x_min && x > x_max && y < y_min && y > y_max) printf("--------------- %d %d %d %d %d %d %d\n", input_idx[i], x, y, x_min, x_max, y_min, y_max);
        vnSBTmp[vnPatchMap[input_idx[i]]]++;  // increment value for this segment @todo: does this represent pixel count per segment?
    }
    vnLblBuffTmp.resize(nSurfCntTmp);     // reduce vectors to their effective sizes
    vnIdxMissing.resize(cnt_miss);



    ///////////////////////////////////////////////
    /////**   Reordering Label Buffer   **/////////
    /////**   Updating Size Buffer      **/////////
    ///////////////////////////////////////////////
    std::vector<int> vnLB(nSurfCntTmp, 0);
    std::vector<int> vnSB(nSurfCntTmp, 0);
    // seg_cnt still 1 at this point and is incremented with each assignment to vnLB
    // @todo why do we have two ways for incrementing the content of vnLB and vnSB?
    for (int i = 1; i < nSurfCntTmp; i++){
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


    if (seg_cnt < 3) {
        /* In the case of 3 segments
         * We refer to the surface of missed indices as segment 1
         * and decrement effective segment count by 1
         */
        for (int i = 0; i < cnt_miss; i++) vnLblMap[vnIdxMissing[i]] = 1;
        seg_cnt_final = 2;
    }
    else {
        ////// Generation of temporary Labeled Segment Map
        for (size_t i = 0; i < input_idx.size(); i++)
            vnLblMap[input_idx[i]] = vnLB[vnPatchMap[input_idx[i]]]; // @todo ???



        //////////////////////////////////////////////////
        ////////**                             **/////////
        ////////**      Postprocessing         **/////////
        ////////**                             **/////////
        //////////////////////////////////////////////////


        std::vector<std::vector<bool> > mbNeighbor(seg_cnt, std::vector<bool>(seg_cnt, false));
        Segm_NeighborMatrix(vnLblMap, input_idx, 1, width, mbNeighbor);


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        vnLB.assign(seg_cnt, 0);
        std::vector<bool> vbCB(seg_cnt, false); // true for large enough segment

        // indicate which segments are large enough and need not be absorbed by larger segments
        for (int i = 1; i < seg_cnt; i++) {
            vnLB[i] = i;
            if (vnSB[i] > tau_s) vbCB[i] = true;
        }

        // iterate through segments to merge small ones into their largest neighbor
        for (int i = 1; i < seg_cnt; i++) {
            if (vbCB[i]) continue;  // large enough

            int max_n = 0, max_size = 0;
            bool clear = false;
            // iterate through neighbors
            for (int j = 1; j < seg_cnt; j++) {
                if (i == j) continue;   // exclude self of course
                if (mbNeighbor[i][j]) {

                    // for large neighbors find largest one
                    if (vbCB[j]) {if (vnSB[vnLB[j]] > max_size) {max_n = j; max_size = vnSB[vnLB[j]];}}
                    else {
                        // merge small neighbors
                        Segm_MergeSegments(i, j, clear, seg_cnt, vnLB, vnSB, vbCB);
                    }
                } // is neighbor (i connected to j)?
            }

            if (max_n) {
                clear = true;
                Segm_MergeSegments(i, max_n, clear, seg_cnt, vnLB, vnSB, vbCB);
                vbCB[i] = true;
            }
        }


        for (int i = 1; i < seg_cnt; i++){
            if (vnLB[i] == i) vnLB[i] = seg_cnt_final++;
            else vnLB[i] = vnLB[vnLB[i]];
        }

        for (size_t i = 0; i < input_idx.size(); i++)
            vnLblMapFinal[input_idx[i]] = vnLB[vnLblMap[input_idx[i]]];
    }

    if (seg_cnt_final) nSurfCnt = seg_cnt_final - 1;
    if (!seg_cnt_final) printf("----------------------Error\n");
}




/* Pre-processing of the tracking, extracting object properties in current scene
 *
 * Input:
 * nSegmCutSize - minimum pixel count for objects to be processed
 * nDsWidth, nDsHeight - width and height of downsampled segment map
 * vnX, vnY, vnZ - coordinate values for point cloud
 * cvm_rgb_ds - downsampled image
 * stTrack - parameters of tracking process
 *
 * Output:
 * stSurf - properties for object surfaces for tracking (siehe struct SurfProp)
 * nSegmCnt - number of segments for tracking (reduced from the number of segmentes after the segmentation)
 */
void Tracking_Pre (int nSegmCutSize, int nDsWidth, int nDsHeight, std::vector<float> vnX, std::vector<float> vnY, std::vector<float> vnZ, cv:: Mat cvm_rgb_ds, TrackProp stTrack,
                  SurfProp &stSurf, int &nSurfCnt) {
    int final_cnt_new = 0;
    for (int i = 1; i < nSurfCnt; i++) {
        if (stSurf.vnPtsCnt[i] < nSegmCutSize) continue;
        int rx_acc = 0, ry_acc = 0;
        int rx_min = nDsWidth, rx_max = 0, ry_min = nDsHeight, ry_max = 0;
        float cx_acc = 0, cy_acc = 0, cz_acc = 0;
        float cx_min = 99, cx_max = -99, cy_min = 99, cy_max = -99, cz_min = 99, cz_max = -99;
        for (int j = 0; j < stSurf.vnPtsCnt[i]; j++) {
            int rx_tmp, ry_tmp;
            GetPixelPos(stSurf.mnPtsIdx[i][j], nDsWidth, rx_tmp, ry_tmp);
            rx_acc += rx_tmp; ry_acc += ry_tmp;
            if (rx_tmp < rx_min) rx_min = rx_tmp;
            if (rx_tmp > rx_max) rx_max = rx_tmp;
            if (ry_tmp < ry_min) ry_min = ry_tmp;
            if (ry_tmp > ry_max) ry_max = ry_tmp;

            float cx_tmp = vnX[stSurf.mnPtsIdx[i][j]], cy_tmp = vnY[stSurf.mnPtsIdx[i][j]], cz_tmp = vnZ[stSurf.mnPtsIdx[i][j]];
            cx_acc += cx_tmp; cy_acc += cy_tmp; cz_acc += cz_tmp;
            if (cx_tmp < cx_min) cx_min = cx_tmp;
            if (cx_tmp > cx_max) cx_max = cx_tmp;
            if (cy_tmp < cy_min) cy_min = cy_tmp;
            if (cy_tmp > cy_max) cy_max = cy_tmp;
            if (cz_tmp < cz_min) cz_min = cz_tmp;
            if (cz_tmp > cz_max) cz_max = cz_tmp;
        }
        stSurf.vnPtsCnt[final_cnt_new] = stSurf.vnPtsCnt[i];
        stSurf.mnPtsIdx[final_cnt_new] = stSurf.mnPtsIdx[i];
        stSurf.mnRect[final_cnt_new][0] = rx_min;
        stSurf.mnRect[final_cnt_new][1] = ry_min;
        stSurf.mnRect[final_cnt_new][2] = rx_max;
        stSurf.mnRect[final_cnt_new][3] = ry_max;
        stSurf.mnCubic[final_cnt_new][0] = cx_min;
        stSurf.mnCubic[final_cnt_new][1] = cy_min;
        stSurf.mnCubic[final_cnt_new][2] = cz_min;
        stSurf.mnCubic[final_cnt_new][3] = cx_max;
        stSurf.mnCubic[final_cnt_new][4] = cy_max;
        stSurf.mnCubic[final_cnt_new][5] = cz_max;
        stSurf.mnRCenter[final_cnt_new][0] = int(float(rx_acc) / stSurf.vnPtsCnt[i]);
        stSurf.mnRCenter[final_cnt_new][1] = int(float(ry_acc) / stSurf.vnPtsCnt[i]);
        stSurf.mnCCenter[final_cnt_new][0] = int(float(cx_acc) / stSurf.vnPtsCnt[i]);
        stSurf.mnCCenter[final_cnt_new][1] = int(float(cy_acc) / stSurf.vnPtsCnt[i]);
        stSurf.mnCCenter[final_cnt_new][2] = int(float(cz_acc) / stSurf.vnPtsCnt[i]);
        stSurf.vnLength[final_cnt_new] = sqrt(pow((cx_max - cx_min), 2) + pow((cy_max - cy_min), 2)+ pow((cz_max - cz_min), 2));

        stSurf.vnMemCtr[final_cnt_new] = stTrack.CntMem - stTrack.CntStable;
        stSurf.vnStableCtr[final_cnt_new] = 0;
        stSurf.vnLostCtr[final_cnt_new] = stTrack.CntLost + 10;
        Calc3DColorHistogram(cvm_rgb_ds, stSurf.mnPtsIdx[final_cnt_new], stTrack.HistoBin, stSurf.mnColorHist[final_cnt_new]);

        final_cnt_new++;
    }
    nSurfCnt = final_cnt_new;
}






/* Sub-function of the pre-processing of the optimization of the tracking: elemination of the unique elements in Munkres-matrix
 */
void Tracking_OptPreFunc (int seg, int j_min, int nMemsCnt, int nObjsNrLimit, float nTrackDist, float huge, std::vector<int> &vnSurfCandCnt, std::vector<int> &vnMemsCandCnt, std::vector<int> &vnMemCandMin, std::vector<int> &vnMatchedSeg, std::vector<std::vector<float> > &mnDistTmp) {
    for (int j = 0; j < nMemsCnt; j++) {
        if (j == j_min) continue;                       //
        if (mnDistTmp[seg][j] > nTrackDist) continue;   //

        mnDistTmp[seg][j] = huge;
        vnMemsCandCnt[j]--;
        vnSurfCandCnt[seg]--;
        if (!vnMemsCandCnt[j]) vnMemCandMin[j] = nObjsNrLimit;

        //// if there is only one candidate in the colum left
        if (vnMemsCandCnt[j] != 1) continue;
        for (int i = 0; i < seg; i++) {
            if (mnDistTmp[i][j] > nTrackDist) continue;
            vnMatchedSeg[i] = j;
            if (vnSurfCandCnt[i] > 1) Tracking_OptPreFunc (i, j, nMemsCnt, nObjsNrLimit, nTrackDist, huge, vnSurfCandCnt, vnMemsCandCnt, vnMemCandMin, vnMatchedSeg, mnDistTmp);
        }
    }
}


/* Pre-Processing of the optimization of the tracking: elemination of the unique elements in Munkres-matrix
 */
void Tracking_OptPre (int nMemsCnt, int nSurfCnt, int huge, int nObjsNrLimit,
                      const TrackProp &stTrack,
                      std::vector<std::vector<float> > &mnDistTmp, std::vector<int> &vnSurfCandCnt, std::vector<int> &vnSegCandMin, std::vector<int> &vnMemsCandCnt, std::vector<int> &vnMemCandMin, std::vector<int> &vnMatchedSeg) {

    float offset = 0.01;
    for (int i = 0; i < nSurfCnt; i++) {
        float j_min = huge;
        for (int j = 0; j < nMemsCnt; j++) {
            if (mnDistTmp[i][j] >= stTrack.Dist) {mnDistTmp[i][j] = huge; continue;}
            vnSurfCandCnt[i]++;

            if (mnDistTmp[i][j] > j_min) continue;
            if (mnDistTmp[i][j] == j_min) mnDistTmp[i][j] += offset;
            else {j_min = mnDistTmp[i][j]; vnSegCandMin[i] = j;}
        }
    }

    for (int j = 0; j < nMemsCnt; j++) {
        float i_min = huge;
        for (int i = 0; i < nSurfCnt; i++) {
            if (mnDistTmp[i][j] >= stTrack.Dist) continue;
            vnMemsCandCnt[j]++;

            if (mnDistTmp[i][j] > i_min) continue;
            if (mnDistTmp[i][j] == i_min) mnDistTmp[i][j] += offset;
            else {i_min = mnDistTmp[i][j]; vnMemCandMin[j] = i;}
        }
    }

    for (int i = 0; i < nSurfCnt; i++) {
        if (vnSurfCandCnt[i]) {
            //// if no other initial elements in the column
            if (vnMemsCandCnt[vnSegCandMin[i]] < 2) {
                if (vnMatchedSeg[i] < nObjsNrLimit) printf("Error, the segment %d is already matched %d\n", i, vnSegCandMin[i]);
                vnMatchedSeg[i] = vnSegCandMin[i];

                if (vnSurfCandCnt[i] > 1) Tracking_OptPreFunc (i, vnSegCandMin[i], nMemsCnt, nObjsNrLimit, stTrack.Dist, huge, vnSurfCandCnt, vnMemsCandCnt, vnMemCandMin, vnMatchedSeg, mnDistTmp);
            }
        }
        else vnMatchedSeg[i] = nObjsNrLimit;
    }
}



/* Postprocessing of the tracking
  */
void Tracking_Post1(int nAttSizeMin, int nMemsCnt, int cnt_new, std::vector<int> objs_new_no, std::vector<bool> objs_old_flag,
                   std::vector<int> vnMemsPtsCnt, std::vector<std::vector<int> > mnMemsRCenter, std::vector<int> vnSurfPtsCnt, std::vector<std::vector<int> > mnSurfRCenter,
                   std::vector<std::vector<float> > &mnNewSize, std::vector<std::vector<float> > &mnNewPos, bool flag_mat, int framec) {

    int cnt_lose = 0, huge = 10000;
    std::vector<int> aaaa;
    for (int memc = 0; memc < nMemsCnt; memc++) {
        if (!objs_old_flag[memc]) {
            int size = mnNewSize.size();
            mnNewSize.resize(size+1);
            mnNewSize[size].assign(cnt_new, 0);
            mnNewPos.resize(size+1);
            mnNewPos[size].assign(cnt_new, 0);
            cnt_lose++;
            aaaa.resize(size+1);
            aaaa[size] = memc;
            for (int trc = 0; trc < cnt_new; trc++) {
                float s_tmp = float(abs(vnMemsPtsCnt[memc] - vnSurfPtsCnt[objs_new_no[trc]]))/vnMemsPtsCnt[memc];
                if (s_tmp > 0.2) mnNewSize[size][trc] = huge;
                else mnNewSize[size][trc] = s_tmp;
                s_tmp = sqrt(pow(mnMemsRCenter[memc][0] - mnSurfRCenter[objs_new_no[trc]][0], 2) + pow(mnMemsRCenter[memc][1] - mnSurfRCenter[objs_new_no[trc]][1], 2));
                if (s_tmp > nAttSizeMin) mnNewPos[size][trc] = huge;
                else mnNewPos[size][trc] = s_tmp;
            }
        }
    }


    if (flag_mat && (cnt_lose || cnt_new)) {
        char sText[128];
        std::ofstream finn1;
        std::string filename = "mmatrix.txt";

        finn1.open(filename.data(), std::ios::app);

        sprintf(sText, "frame %d    %dx%d\n\n", framec, cnt_lose, cnt_new);
        finn1<< sText;
        for (int j = 0; j < cnt_new; j++) {if (!j) sprintf(sText, "%11d", j); else sprintf(sText, "%26d", j); finn1<< sText;} finn1<< "\n";
        for (int i = 0; i < cnt_lose; i++) {for (int j = 0; j < cnt_new; j++) {if (!j) sprintf(sText, "%2d %10.3f %5d %5d", i, mnNewSize[i][j], vnMemsPtsCnt[aaaa[i]], vnSurfPtsCnt[objs_new_no[j]]); else sprintf(sText, "%10.3f %5d %5d", mnNewSize[i][j], vnMemsPtsCnt[aaaa[i]], vnSurfPtsCnt[objs_new_no[j]]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
        for (int i = 0; i < cnt_lose; i++) {for (int j = 0; j < cnt_new; j++) {if (!j) sprintf(sText, "%2d %10.3f %3d %3d %3d %3d", i, mnNewPos[i][j], mnMemsRCenter[aaaa[i]][0], mnMemsRCenter[aaaa[i]][1], mnSurfRCenter[objs_new_no[j]][0], mnSurfRCenter[objs_new_no[j]][1]); else sprintf(sText, "%10.3f %3d %3d %3d %3d", mnNewPos[i][j], mnMemsRCenter[aaaa[i]][0], mnMemsRCenter[aaaa[i]][1], mnSurfRCenter[objs_new_no[j]][0], mnSurfRCenter[objs_new_no[j]][1]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
        finn1.close();
    }
}


/* Postprocessing of the tracking
  */
void Tracking_Post2(int nMemsCnt, const TrackProp &stTrack, SurfProp stMemsOld, std::vector<int> &vnMemsValidIdx, std::vector<std::vector<float> > &mnMemsRelPose, SurfProp &stMems, int framec, bool flag_mat) {

    std::vector<int> past_idx = vnMemsValidIdx;
    std::vector<std::vector<float> > past_pose = mnMemsRelPose;
    std::vector<std::vector<float> > comp_pose;
    std::vector<int> size_diff;
    std::vector<float> posi_diff;


    //** selecting valid surfaces from memroy **//
    vnMemsValidIdx.resize(0);
    float pii = 3.14159; int valid = 0;
    for (int i = 0; i < nMemsCnt; i++) {
        if (stMems.vnStableCtr[i] < stTrack.CntStable - 1 || stMems.vnStableCtr[i] < 0 || stMems.vnLostCtr[i] > stTrack.CntLost) continue;
        vnMemsValidIdx.resize(valid+1); vnMemsValidIdx[valid++] = stMems.vnIdx[i];
    }

    //** comparing poses of current control surfaces to last control surface in memroy **//
    mnMemsRelPose.assign(valid, std::vector<float>(valid, -10));
    comp_pose.assign(valid, std::vector<float>(valid, 0));
    size_diff.resize(valid, -1); posi_diff.resize(valid, -1);
    if (valid) {

        //* making pose matrix from current control surface *//
        mnMemsRelPose[valid-1][valid-1] = 0;
        for (int i = 0; i < valid-1; i++) {
            mnMemsRelPose[i][i] = 0;
            for (int j = i+1; j < valid; j++) {
                int xx = stMems.mnRCenter[vnMemsValidIdx[i]][0] - stMems.mnRCenter[vnMemsValidIdx[j]][0];
                int yy = stMems.mnRCenter[vnMemsValidIdx[i]][1] - stMems.mnRCenter[vnMemsValidIdx[j]][1];
                float Pose, PoseT;
                if (xx) {
                    PoseT = atan(float(yy)/xx);
                    if (xx > 0) Pose = PoseT;
                    else {if (yy > 0) Pose = PoseT + pii; else Pose = PoseT - pii;}
                }
                else {if (yy > 0) Pose = pii/2; else Pose = -pii/2;}

                mnMemsRelPose[i][j] = Pose;
                mnMemsRelPose[j][i] = -Pose;
            }
        }

        //* comparing poses with last control surfaces *//
        bool flag_ttemp = false;
        for (int i = 0; i < valid; i++) {
            if (stMems.vnStableCtr[vnMemsValidIdx[i]] < stTrack.CntStable) continue;
            int i_past = -1;
            for (int ii = 0; ii < past_idx.size(); ii++) {if (past_idx[ii] == vnMemsValidIdx[i]) {i_past = ii; break;}}
            if (i_past < 0) continue;
            size_diff[i] = abs(stMems.vnPtsCnt[vnMemsValidIdx[i]] - stMemsOld.vnPtsCnt[vnMemsValidIdx[i]]);
            posi_diff[i] = sqrt(pow(stMems.mnRCenter[vnMemsValidIdx[i]][0] - stMemsOld.mnRCenter[vnMemsValidIdx[i]][0], 2) + pow(stMems.mnRCenter[vnMemsValidIdx[i]][1] - stMemsOld.mnRCenter[vnMemsValidIdx[i]][1], 2));
            if (i == valid-1) continue;
            for (int j = i+1; j < valid; j++) {
                if (stMems.vnStableCtr[vnMemsValidIdx[j]] < stTrack.CntStable) continue;
                for (int jj = 0; jj < past_idx.size(); jj++) {
                    if (past_idx[jj] != vnMemsValidIdx[j]) continue;
                    float pose = fabs(mnMemsRelPose[i][j] - past_pose[i_past][jj]);
                    if (pose < pii/2) continue;
                    if (pose > pii) pose = 2*pii + 0.01 - pose;
                    if (pose < pii/2) continue;
                    comp_pose[i][j] = pose; comp_pose[j][i] = pose;
                    flag_ttemp = true;
                    break;
                }
            }
        }


        //* 1. Eleminating right elements from the pose-difference matrix*//
        std::vector<std::vector<float> > comp_pose1 = comp_pose;
        for (int i = 0; i < valid; i++) {
            if (size_diff[i] > stMems.vnPtsCnt[vnMemsValidIdx[i]]*0.12 || posi_diff[i] > 18) continue;
            for (int j = 0; j < valid; j++) {
                if (comp_pose[i][j]) {comp_pose[i][j] = 0; comp_pose[j][i] = 0;}
            }
        }

        //* 2. Sunstituting changed elements in the pose-difference matrix*//
        for (int i = 0; i < valid; i++) {
            if (posi_diff[i] > 8) {
                float diff_min = 100;
                int j_tmp = -1;
                for (int j = 0; j < valid; j++) {
                    if (comp_pose[i][j]) {
                        float temp_diff = sqrt(pow(stMems.mnRCenter[vnMemsValidIdx[i]][0] - stMemsOld.mnRCenter[vnMemsValidIdx[j]][0], 2) + pow(stMems.mnRCenter[vnMemsValidIdx[i]][1] - stMemsOld.mnRCenter[vnMemsValidIdx[j]][1], 2));
                        if (temp_diff < 8 && temp_diff < diff_min) {diff_min = temp_diff; j_tmp = j;}
                    }
                }
                if (j_tmp >= 0) {
                    comp_pose[i][j_tmp] = 0;
                    comp_pose[j_tmp][i] = 0;


                    int tPtsCnt = stMems.vnPtsCnt[vnMemsValidIdx[i]];
                    std::vector<int> tPtsIdx = stMems.mnPtsIdx[vnMemsValidIdx[i]];
                    std::vector<int> tRect = stMems.mnRect[vnMemsValidIdx[i]];
                    std::vector<int> tRCenter = stMems.mnRCenter[vnMemsValidIdx[i]];
                    std::vector<float> tCubic = stMems.mnCubic[vnMemsValidIdx[i]];
                    std::vector<float> tCCenter = stMems.mnCCenter[vnMemsValidIdx[i]];
                    std::vector<float> tColorHist = stMems.mnColorHist[vnMemsValidIdx[i]];
                    float tLength = stMems.vnLength[vnMemsValidIdx[i]];
                    int tStableCtr = stMems.vnStableCtr[vnMemsValidIdx[i]];
                    int tLostCtr = stMems.vnLostCtr[vnMemsValidIdx[i]];
                    int tMemCtr = stMems.vnMemCtr[vnMemsValidIdx[i]];
                    int tMemsFound = stMems.vnFound[vnMemsValidIdx[i]];

                    stMems.vnPtsCnt[vnMemsValidIdx[i]] = stMems.vnPtsCnt[vnMemsValidIdx[j_tmp]];
                    stMems.mnPtsIdx[vnMemsValidIdx[i]] = stMems.mnPtsIdx[vnMemsValidIdx[j_tmp]];
                    stMems.mnRect[vnMemsValidIdx[i]] = stMems.mnRect[vnMemsValidIdx[j_tmp]];
                    stMems.mnRCenter[vnMemsValidIdx[i]] = stMems.mnRCenter[vnMemsValidIdx[j_tmp]];
                    stMems.mnCubic[vnMemsValidIdx[i]] = stMems.mnCubic[vnMemsValidIdx[j_tmp]];
                    stMems.mnCCenter[vnMemsValidIdx[i]] = stMems.mnCCenter[vnMemsValidIdx[j_tmp]];
                    stMems.mnColorHist[vnMemsValidIdx[i]] = stMems.mnColorHist[vnMemsValidIdx[j_tmp]];
                    stMems.vnLength[vnMemsValidIdx[i]] = stMems.vnLength[vnMemsValidIdx[j_tmp]];
                    stMems.vnStableCtr[vnMemsValidIdx[i]] = stMems.vnStableCtr[vnMemsValidIdx[j_tmp]];
                    stMems.vnLostCtr[vnMemsValidIdx[i]] = stMems.vnLostCtr[vnMemsValidIdx[j_tmp]];
                    stMems.vnMemCtr[vnMemsValidIdx[i]] = stMems.vnMemCtr[vnMemsValidIdx[j_tmp]];
                    stMems.vnFound[vnMemsValidIdx[i]] = stMems.vnFound[vnMemsValidIdx[j_tmp]];

                    stMems.vnPtsCnt[vnMemsValidIdx[j_tmp]] = tPtsCnt;
                    stMems.mnPtsIdx[vnMemsValidIdx[j_tmp]] = tPtsIdx;
                    stMems.mnRect[vnMemsValidIdx[j_tmp]] = tRect;
                    stMems.mnRCenter[vnMemsValidIdx[j_tmp]] = tRCenter;
                    stMems.mnCubic[vnMemsValidIdx[j_tmp]] = tCubic;
                    stMems.mnCCenter[vnMemsValidIdx[j_tmp]] = tCCenter;
                    stMems.mnColorHist[vnMemsValidIdx[j_tmp]] = tColorHist;
                    stMems.vnLength[vnMemsValidIdx[j_tmp]] = tLength;
                    stMems.vnStableCtr[vnMemsValidIdx[j_tmp]] = tStableCtr;
                    stMems.vnLostCtr[vnMemsValidIdx[j_tmp]] = tLostCtr;
                    stMems.vnMemCtr[vnMemsValidIdx[j_tmp]] = tMemCtr;
                    stMems.vnFound[vnMemsValidIdx[j_tmp]] = tMemsFound;
                }
            }
        }


        if (flag_mat) {
            char sText[128];
            std::ofstream finn1;
            std::string filename = "mmmatrix.txt";

            finn1.open(filename.data(), std::ios::app);

            sprintf(sText, "pos_diff  %d %dx%d\n", framec, valid, valid);
            finn1<< sText;
            for (int j = 0; j < past_pose.size(); j++) {if (!j) sprintf(sText, "%15d", j); else sprintf(sText, "%8d", j); finn1<< sText;} finn1<< "\n";
            for (int i = 0; i < past_pose.size(); i++) {for (int j = 0; j < past_pose.size(); j++) {if (!j) sprintf(sText, "%2d %3d %8.3f", i, past_idx[i], past_pose[i][j]); else sprintf(sText, "%8.3f", past_pose[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15d", j); else sprintf(sText, "%8d", j); finn1<< sText;} finn1<< "\n";
            for (int i = 0; i < valid; i++) {for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%2d %3d %8.3f", i, vnMemsValidIdx[i], mnMemsRelPose[i][j]); else sprintf(sText, "%8.3f", mnMemsRelPose[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
            for (int i = 0; i < valid; i++) {for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%2d %3d %8.3f", i, vnMemsValidIdx[i], comp_pose1[i][j]); else sprintf(sText, "%8.3f", comp_pose1[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
            for (int i = 0; i < valid; i++) {for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%2d %3d %8.3f", i, vnMemsValidIdx[i], comp_pose[i][j]); else sprintf(sText, "%8.3f", comp_pose[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15d", stMemsOld.vnIdx[vnMemsValidIdx[j]]); else sprintf(sText, "%8d", stMemsOld.vnIdx[vnMemsValidIdx[j]]); finn1<< sText;} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15d", stMems.vnIdx[vnMemsValidIdx[j]]); else sprintf(sText, "%8d", stMems.vnIdx[vnMemsValidIdx[j]]); finn1<< sText;} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15d", stMemsOld.vnPtsCnt[past_idx[j]]); else sprintf(sText, "%8d", stMemsOld.vnPtsCnt[past_idx[j]]); finn1<< sText;} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15d", stMems.vnPtsCnt[vnMemsValidIdx[j]]); else sprintf(sText, "%8d", stMems.vnPtsCnt[vnMemsValidIdx[j]]); finn1<< sText;} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15d", size_diff[j]); else sprintf(sText, "%8d", size_diff[j]); finn1<< sText;} finn1<< "\n";
            for (int j = 0; j < valid; j++) {if (!j) sprintf(sText, "%15.3f", posi_diff[j]); else sprintf(sText, "%8.3f", posi_diff[j]); finn1<< sText;} finn1<< "\n\n";
            finn1.close();
        }
    }
}






/* Main tracking function - matching objects surfaces in current frame to object surfaces in Short-Term memory
 *
 * Input:
 * nSurfCnt - number of object surfaces in current scene
 * nObjsNrLimit - maximum number of object surfaces which can be handled by the system
 * dp_dia - position displacement
 * stTrack -  tracking parameters
 * bin - number of bins for color histogram
 * stSurf - object properties for tracking (siehe struct SurfProp)
 *
 * Output:
 * stMems - properties of object surfaces in the Short-Term Memory (SurfProp is self-defined struct)
 * nMemsCnt - number of object surfaces in Short-Term Memory
 */
void Tracking (int nSurfCnt, int nObjsNrLimit, TrackProp stTrack, int bin, SurfProp stSurf, SurfProp &stMems, int &nMemsCnt,
              std::vector<int> &vnMemsValidIdx, std::vector<std::vector<float> > &mnMemsRelPose, bool flag_mat, int framec) {


    int cnt_new = 0, cnt_old = 0;
    std::vector<int> objs_new_no(nSurfCnt, 0);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////** Generating feature-distance matrix for matching segmentes surfeces with surfaces in the Short-Term Memory **/////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float huge = 100;
    int nDim = max(nSurfCnt, nMemsCnt);
    std::vector<std::vector<float> > mnDistClr(nDim, std::vector<float>(nDim, huge));       // **must be removed
    std::vector<std::vector<float> > mnDistPos(nDim, std::vector<float>(nDim, huge));       // **must be removed
    std::vector<std::vector<float> > mnDistSiz(nDim, std::vector<float>(nDim, huge));       // **must be removed
    std::vector<std::vector<float> > mnDistTotal(nDim, std::vector<float>(nDim, huge));     // (total) distance matrix

    if (nSurfCnt && nMemsCnt) {
        for (int i = 0; i < nSurfCnt; i++) {
            for (int j = 0; j < nMemsCnt; j++) {
                float xc = stSurf.mnCCenter[i][0], yc = stSurf.mnCCenter[i][1], zc = stSurf.mnCCenter[i][2];
                float dp = sqrt(pow(xc - stMems.mnCCenter[j][0], 2) + pow(yc - stMems.mnCCenter[j][1], 2) + pow(zc - stMems.mnCCenter[j][2], 2));
                float ds = (float)abs(stMems.vnLength[j] - stSurf.vnLength[i])/max(stMems.vnLength[j], stSurf.vnLength[i]);
                float dc = 0; for (int ii = 0; ii < bin; ii++) dc += fabs(stMems.mnColorHist[j][ii] - stSurf.mnColorHist[i][ii]);
                mnDistPos[i][j] = dp;
                mnDistSiz[i][j] = ds;
                mnDistClr[i][j] = dc;

                if (dp < stTrack.DPos && ds < stTrack.DSize && dc < stTrack.DClr)
                    mnDistTotal[i][j] = stTrack.FPos * dp + stTrack.FSize * ds + stTrack.FClr * dc;
            }
        }

        if (flag_mat) {
            char sText[128];
            std::ofstream finn1;
            std::string filename = "matrix.txt";

            finn1.open(filename.data(), std::ios::app);

            sprintf(sText, "frame %d    %dx%d\n\n", framec, nSurfCnt, nMemsCnt);
            finn1 << sText;
            finn1 << "pos\n";
            for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%11d", j); else sprintf(sText, "%8d", j); finn1<< sText;} finn1<< "\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistPos[i][j]); else sprintf(sText, "%8.3f", mnDistPos[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";

            finn1 << "size\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistSiz[i][j]); else sprintf(sText, "%8.3f", mnDistSiz[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";

            finn1 << "color\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistClr[i][j]); else sprintf(sText, "%8.3f", mnDistClr[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";

            finn1 << "dist\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistTotal[i][j]); else sprintf(sText, "%8.3f", mnDistTotal[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
            finn1.close();
        }
    }




    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////**                                                              **////////////////////////////////
    /////////////////////////**                            Optimization                      **////////////////////////////////
    /////////////////////////**                                                              **////////////////////////////////
    /////////////////////////------------------------------------------------------------------////////////////////////////////


    //////////////////////////////////////////////////////////////////////////////////////////
    ////////////** Pre-Processing **//////////////////////////////////////////////////////////
    ////////////** Elemination of rows and columns that have a unique minimum match   **//////
    //////////////////////////////////////////////////////////////////////////////////////////
    std::vector<int> vnSurfCandCnt(nSurfCnt, 0);
    std::vector<int> vnSegCandMin(nSurfCnt, nObjsNrLimit);        //
    std::vector<int> vnMemsCandCnt(nMemsCnt, 0);
    std::vector<int> vnMemCandMin(nMemsCnt, nObjsNrLimit);
    std::vector<int> vnMatchedSeg(nSurfCnt, nObjsNrLimit*2);
    std::vector<int> vnMatchedMem(nMemsCnt, nObjsNrLimit*2);

    std::vector<std::vector<float> > mnDistTmp = mnDistTotal;           // specified distance matrix
    Tracking_OptPre (nMemsCnt, nSurfCnt, huge, nObjsNrLimit, stTrack, mnDistTmp, vnSurfCandCnt, vnSegCandMin, vnMemsCandCnt, vnMemCandMin, vnMatchedSeg);


    /////////////////////////////////////////////////////////////////////////////////
    ////////////**              Main Optimization              **////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    std::vector<int> idx_seg;
    int cnt_nn = 0;
    for (int i = 0; i < nSurfCnt; i++) {
        if (vnMatchedSeg[i] > nObjsNrLimit) {
            for (int j = 0; j < nMemsCnt; j++) {
                if (mnDistTmp[i][j] < stTrack.Dist) {
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
        for (int j = 0; j < nMemsCnt; j++) {
            if (!vnMatchedMem[j]) {
                idx_mem.resize(cnt_nn + 1);
                idx_mem[cnt_nn++] = j;
            }
        }


        int munkres_huge = 100;
        int nDimMunkres = max((int)idx_seg.size(), (int)idx_mem.size());
        MunkresMatrix<double> m_MunkresIn(nDimMunkres, nDimMunkres);
        MunkresMatrix<double> m_MunkresOut(nDimMunkres, nDimMunkres);

        for (size_t i = 0; i < idx_seg.size(); i++) {
            for (size_t j = 0; j < idx_mem.size(); j++)
                m_MunkresIn(i,j) = mnDistTmp[idx_seg[i]][idx_mem[j]];
        }

        if (idx_mem.size() > idx_seg.size()) {
            for (int i = (int)idx_seg.size(); i < nDimMunkres; i++) {
                for (int j = 0; j < nDimMunkres; j++) m_MunkresIn(i,j) = rand()% 10 + munkres_huge;
            }
        }
        if (idx_mem.size() < idx_seg.size()) {
            for (int j = (int)idx_mem.size(); j < nDimMunkres; j++) {
                for (int i = 0; i < nDimMunkres; i++) m_MunkresIn(i,j) = rand()% 10 + munkres_huge;
            }

        }

        m_MunkresOut = m_MunkresIn;

        Munkres m;
        m.solve(m_MunkresOut);



        //////* Specifying the output matrix *//////////////////
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

            sprintf(sText, "disttmp\n");
            finn1<< sText;
            for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%11d", j); else sprintf(sText, "%8d", j); finn1<< sText;} finn1<< "\n";
            for (int i = 0; i < nDim; i++) {for (int j = 0; j < nDim; j++) {if (!j) sprintf(sText, "%2d %8.3f", i, mnDistTmp[i][j]); else sprintf(sText, "%8.3f", mnDistTmp[i][j]); finn1<< sText;} finn1<< "\n";} finn1<< "\n";
            finn1.close();
        }
    }
    /////////////////////////------------------------------------------------------------------////////////////////////////////
    /////////////////////////**                                                              **////////////////////////////////
    /////////////////////////**                    End of the optimization                   **////////////////////////////////
    /////////////////////////**                                                              **////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////**                                                              **////////////////////////////////
    /////////////////////////**                        Postprocessing                        **////////////////////////////////
    /////////////////////////**                                                              **////////////////////////////////
    /////////////////////////------------------------------------------------------------------////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////**         Short-Term-Memory             **///////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    //** Update properties of matched segments in the Short-Term-Memory **//
    std::vector<bool> objs_old_flag(nMemsCnt, false);
    SurfProp stMemsOld = stMems;
    for (int i = 0; i < nSurfCnt; i++) {

        //** Assign matched segments to the Short-Term-Memory **//
        int cand = 0, j_tmp = -1;
        if (vnMatchedSeg[i] < nObjsNrLimit) {
            cand = 1;
            j_tmp = vnMatchedSeg[i];
        }

        if (cand) {
            if (j_tmp < 0) {printf("Tracking Error\n"); continue;}

            objs_old_flag[j_tmp] = true;
            stMems.vnPtsCnt[j_tmp] = stSurf.vnPtsCnt[i];
            stMems.mnPtsIdx[j_tmp] = stSurf.mnPtsIdx[i];
            stMems.mnRect[j_tmp] = stSurf.mnRect[i];
            stMems.mnRCenter[j_tmp] = stSurf.mnRCenter[i];
            stMems.mnCubic[j_tmp] = stSurf.mnCubic[i];
            stMems.mnCCenter[j_tmp] = stSurf.mnCCenter[i];
            stMems.mnColorHist[j_tmp] = stSurf.mnColorHist[i];
            stMems.vnLength[j_tmp] = stSurf.vnLength[i];
        }
        else objs_new_no[cnt_new++] = i;
    }




    //** Filter stable objects from the Short-Term-Memory **//
    for (int memc = 0; memc < nMemsCnt; memc++) {
        if (objs_old_flag[memc]) {
            stMems.vnMemCtr[memc]++;
            if (stMems.vnMemCtr[memc] < stTrack.CntMem - stTrack.CntStable + 2) stMems.vnMemCtr[memc] = stTrack.CntMem - stTrack.CntStable + 1;
            if (stMems.vnMemCtr[memc] > 100*stTrack.CntMem) stMems.vnMemCtr[memc] = 100*stTrack.CntMem;
            if (stMems.vnStableCtr[memc] < 0) stMems.vnStableCtr[memc] = 1; else stMems.vnStableCtr[memc]++;
            if (stMems.vnStableCtr[memc] > 100*(stTrack.CntStable+1)) stMems.vnStableCtr[memc] = 100*stTrack.CntStable;
            stMems.vnLostCtr[memc] = 0;
        }
        else {
            if (cnt_new) {
                for (int trc = 0; trc < cnt_new; trc++) {
                }
            }

            stMems.vnMemCtr[memc]--;
            stMems.vnLostCtr[memc]++;
            if (stMems.vnMemCtr[memc] >= stTrack.CntMem) stMems.vnMemCtr[memc] = stTrack.CntMem;
            if (stMems.vnStableCtr[memc] > 0) stMems.vnStableCtr[memc] = 0; else stMems.vnStableCtr[memc]--;
            if (stMems.vnStableCtr[memc] < -100*(stTrack.CntStable+1)) stMems.vnStableCtr[memc] = -100*stTrack.CntStable;
            if (stMems.vnLostCtr[memc] > 100*(stTrack.CntLost+1)) stMems.vnLostCtr[memc] = 100*stTrack.CntLost;
        }
    }

    //** Restack stable objects in the Short-Term-Memory **//
    int cnt_tmp = 0;
    for (int i = 0; i < nMemsCnt; i++) {
        if (stMems.vnMemCtr[i] < 0) continue;
        stMems.vnIdx[cnt_tmp] = stMems.vnIdx[i];
        stMems.vnPtsCnt[cnt_tmp] = stMems.vnPtsCnt[i];
        stMems.mnPtsIdx[cnt_tmp] = stMems.mnPtsIdx[i];
        stMems.mnRect[cnt_tmp] = stMems.mnRect[i];
        stMems.mnRCenter[cnt_tmp] = stMems.mnRCenter[i];
        stMems.mnCubic[cnt_tmp] = stMems.mnCubic[i];
        stMems.mnCCenter[cnt_tmp] = stMems.mnCCenter[i];
        stMems.mnColorHist[cnt_tmp] = stMems.mnColorHist[i];
        stMems.vnLength[cnt_tmp] = stMems.vnLength[i];
        stMems.vnStableCtr[cnt_tmp] = stMems.vnStableCtr[i];
        stMems.vnLostCtr[cnt_tmp] = stMems.vnLostCtr[i];
        stMems.vnMemCtr[cnt_tmp] = stMems.vnMemCtr[i];
        stMems.vnFound[cnt_tmp] = stMems.vnFound[i];
        cnt_tmp++;
    }
    cnt_old = cnt_tmp;


    //** Reusing unused surface indeces to the new appearing surfaces **//
    std::vector<int> mems_idx(cnt_old, 0);
    std::vector<int> mems_idx_new(cnt_new, 0);
    for (int i = 0; i < cnt_old; i++) mems_idx[i] = stMems.vnIdx[i];
    std::sort(mems_idx.begin(), mems_idx.end());
    int cnt_tmp_tmp = 0;
    if (cnt_old > 2 && cnt_new) {
        if (mems_idx[2] > mems_idx[1]) {
            for (int i = 2; i < cnt_old; i++) {
                int diff = mems_idx[i] - mems_idx[i-1];
                if (diff > 1) {
                    for (int j = 0; j < diff-1; j++) {
                        mems_idx_new[cnt_tmp_tmp++] = mems_idx[i-1] + j+1;
                        if (cnt_tmp_tmp >= cnt_new) break;
                    }
                }
                if (cnt_tmp_tmp >= cnt_new) break;
            }
            if (cnt_tmp_tmp < cnt_new)
                for (int i = 0; i < cnt_new - cnt_tmp_tmp; i++) mems_idx_new[cnt_tmp_tmp + i] = mems_idx[cnt_old-1] + i+1;
        }
        else printf("eeeeeeeeeeee \n");
    }
    else for (int i = 0; i < cnt_new; i++) mems_idx_new[i] = i;



    //** Pushing new objects (unmatched segments) on the Short-Term-Memory **//
    for (int i = 0; i < cnt_new; i++) {
        stMems.vnIdx[cnt_old + i] = mems_idx_new[i];
        stMems.vnPtsCnt[cnt_old + i] = stSurf.vnPtsCnt[objs_new_no[i]];
        stMems.mnPtsIdx[cnt_old + i] = stSurf.mnPtsIdx[objs_new_no[i]];
        stMems.mnRect[cnt_old + i] = stSurf.mnRect[objs_new_no[i]];
        stMems.mnRCenter[cnt_old + i] = stSurf.mnRCenter[objs_new_no[i]];
        stMems.mnCubic[cnt_old + i] = stSurf.mnCubic[objs_new_no[i]];
        stMems.mnCCenter[cnt_old + i] = stSurf.mnCCenter[objs_new_no[i]];
        stMems.mnColorHist[cnt_old + i] = stSurf.mnColorHist[objs_new_no[i]];
        stMems.vnLength[cnt_old + i] = stSurf.vnLength[objs_new_no[i]];
        stMems.vnStableCtr[cnt_old + i] = stSurf.vnStableCtr[objs_new_no[i]];
        stMems.vnLostCtr[cnt_old + i] = stSurf.vnLostCtr[objs_new_no[i]];
        stMems.vnMemCtr[cnt_old + i] = stSurf.vnMemCtr[objs_new_no[i]];
    }
    nMemsCnt = cnt_old + cnt_new;

    if (nMemsCnt >= nObjsNrLimit) printf("Object queue exceeds object no. limit %d\n", nObjsNrLimit);



    ////////////////////////////////////////////////////////////////////////////////
    ////////**      2. Postprocessing - Enhancing tracking agility      **//////////
    ////////////////////////////////////////////////////////////////////////////////

    Tracking_Post2 (nMemsCnt, stTrack, stMemsOld, vnMemsValidIdx, mnMemsRelPose, stMems, framec, flag_mat);
}

