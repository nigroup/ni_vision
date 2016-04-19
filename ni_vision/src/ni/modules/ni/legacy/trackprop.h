#ifndef TRACKPROP_H
#define TRACKPROP_H

struct TrackProp {
    int ClrMode, HistoBin;
    double DPos, DSize, DClr, Dist, FPos, FSize, FClr, MFac;
    int CntMem, CntStable, CntLost;
};

#endif // TRACKPROP_H
