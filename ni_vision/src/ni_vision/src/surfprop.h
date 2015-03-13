#ifndef SURFPROP_H
#define SURFPROP_H

#include <vector>

/**
 * @brief struct for encapsulating properties
 * for multiple surfaces
 */
struct SurfProp {

    std::vector<int> vnIdx;     ///< index per surface
    std::vector<int> vnPtsCnt;  ///< points occupied by each surface
    std::vector<std::vector<int> > mnPtsIdx;        ///< indicies of points occupied by surface in point cloud
    std::vector<std::vector<int> > mnRect;          ///< Rectangle per surface
    std::vector<std::vector<int> > mnRCenter;       ///< Rectangle center per surface
    std::vector<std::vector<float> > mnCubic;       ///< cube around each surface
    std::vector<std::vector<float> > mnCCenter;     ///< center of cube around each surface
    std::vector<float> vnLength;
    std::vector<std::vector<float> > mnColorHist;   ///< color histogram per surface
    std::vector<int> vnMemCtr;
    std::vector<int> vnStableCtr;
    std::vector<int> vnLostCtr;
    std::vector<int> vnFound;
};

#endif // SURFPROP_H
