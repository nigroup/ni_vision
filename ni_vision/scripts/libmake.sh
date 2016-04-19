#!/bin/bash
rosrun ni_tools libmake -inPath ~/Video/Objektfotos/Training/SIFT/ -obj Schlagsahne -thr 50 -mrect 100 -fox 340 -foy 180 -fow 600 -foh 600 -opfile lib_sift_Schlagsahne_0015 -outPath ~/Video/Objektfotos/Training/ -featuremode 0 -siftsc 3 -siftis 1.6 -siftpt 0.015 -siftlibmode 0 -dthres 0.4 -dispthresh 100 -brief 0 -ext bmp -reinf 1 -tgh 1600 -pthres 20 -mthres 0.4 -tlblank 5 -tlres 3 -interv 20 -tlmax 24 -tlmin 6 -tlmin2 3 -tstdist 0.2

# Parameter:
# inPath - input directory (where the training images are located)
# obj - Name of the object which is trained
# outPath - output path (for models)
# andere können entfernt werden
