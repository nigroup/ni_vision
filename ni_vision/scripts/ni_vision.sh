#!/bin/bash
rosrun ni_vision ni_vision ~input_img:=/camera/rgb/image_color ~input_pc:=/camera/depth_registered/points -dlim 5 -ddmod 0 -dgc 0.5 -dgfmod 2 -dgfsize 5 -dgsmtmod 2 -dgtauf1 0.0018 -dgtauf2 0.0045 -sgtaud 0.04 -sgtaug 0.003 -sggcs 10 -sgtaus 200 -gssigma 0.8 -gsgth 200 -gsmins 150 -trkmod 0 -trkdp 0.5 -trkds 0.3 -trkdc 0.4 -trkd 1.6 -trkpf 0.1 -trksf 0.5 -trkcf 0.4 -trkmf 100 -trkcmod 1 -trkcm 10 -trkcs 1 -trkcl 0 -atttd 0 -attmax 400 -attmin 30 -attpm 8 -attar1 20 -attar2 30 -siftsc 3 -siftis 1.6 -siftpt 0.015 -flknn 2 -fltp 0.9 -flmf 0.70 -flmc 3 -recfeat 20 -recdc 0.45 -reccmm 1 -recvmd 0 -recordmod 1 -tmessfl 500 -snapfm 1 -targetpath ~/Video/TargetImg/ -target Soennecken -libpath ~/Video/library/Lib8B/ -siftlibfile lib_sift_Soennecken_0015.yaml -colorlibfile simplelib_3dch_Soennecken.yaml -nnthresh 0.2 -trovl 0.8 -deltabin 12 -tnumb 3 -tscale 0.1 -torient 0.5233333 -datadir ~/zNiData

# recdc - Farbschwellwert

# - libpath, siftlibfile, colorlibfile, target (alle anderen können entfernt werden)
