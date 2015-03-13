#ifndef SURFPROP_H
#define SURFPROP_H

/**
 * @brief struct for encapsulating properties
 * for multiple surfaces
 */
struct SurfProp {
    std::vector<int> vnIdx, vnPtsCnt;
    std::vector<std::vector<int> > mnPtsIdx;
    std::vector<std::vector<int> > mnRect, mnRCenter;
    std::vector<std::vector<float> > mnCubic, mnCCenter;
    std::vector<float> vnLength;
    std::vector<std::vector<float> > mnColorHist;
    std::vector<int> vnMemCtr, vnStableCtr, vnLostCtr;
    std::vector<int> vnFound;
};

#endif // SURFPROP_H
