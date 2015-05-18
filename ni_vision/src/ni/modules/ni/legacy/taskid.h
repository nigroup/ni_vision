#ifndef TASKID_H
#define TASKID_H

/**
 * @brief ID of Tasks
 */
struct TaskID {
    int nRgbOrg, nRgbDs, nDepth, nInfo, nRecVideo, nSnap;
    int nSegmentation, nSegm, nGSegm, nTrack, nAtt;
    int nRecognition, nRecogOrg, nRecogDs, nSIFT;
    int nRecTime, nRstTime, nPrmInfo, nPrmSett, nPrmSegm, nPrmRecog, nRstPrm;
    int nRes1, nRes2;
};

#endif // TASKID_H
