#!/usr/bin/env python
# exact C++ implementation of lowe's sift program
# Copyright (C) zerofrog(@gmail.com), 2008-2009
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Lesser GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http:#www.gnu.org/licenses/>.
from numpy import *
import time, Image
from optparse import OptionParser

import siftfastpy

if __name__=='__main__':
    parser = OptionParser(description='Compute SIFT features of an image.')
    parser.add_option('--image',
                      action="store",type='string',dest='image',default='test0.jpg',
                      help='Image to test')
    (options, args) = parser.parse_args()
    im = Image.open(options.image)
    im = im.convert(mode='L') # convert to greyscale
    siftimage = siftfastpy.Image(im.size[0],im.size[1])
    siftimage.SetData(reshape(im.getdata(),im.size[::-1]))

    starttime = time.time()    
    frames,desc = siftfastpy.GetKeypoints(siftimage)
    print '%d  keypoints found in %fs'%(frames.shape[0],time.time()-starttime)

    starttime = time.time()
    framesonly = siftfastpy.GetKeypointFrames(siftimage)
    print 'frames only: %d  keypoints found in %fs'%(framesonly.shape[0],time.time()-starttime)
    starttime = time.time()
    frames2,desc2 = siftfastpy.GetKeypointDescriptors(siftimage,framesonly)
    print 'descriptors only: %d  keypoints found in %fs'%(frames2.shape[0],time.time()-starttime)

    # try with no image doubling
    params = siftfastpy.GetSiftParameters()
    params.DoubleImSize = 0
    siftfastpy.SetSiftParameters(params)
    starttime = time.time()    
    framesonly = siftfastpy.GetKeypointFrames(siftimage)
    print 'No Image Doubling: %d  keypoint frames found in %fs'%(framesonly.shape[0],time.time()-starttime)

    # test
#     from scipy import spatial
#     print 'pos error: ', mean(numpy.min(spatial.distance_matrix(framesonly[:,0:4],frames),0))
#     print 'desc error: ', mean(numpy.min(spatial.distance_matrix(desc2,desc),0))

#     print '%d %d'%(desc.shape[0],desc.shape[1])
#     for i in xrange(frames.shape[0]):
#         print '%d %d %f %f'%(frames[i,1],frames[i,0],frames[i,3],frames[i,2])
#         s = ''
#         for j,d in enumerate(desc[i]):
#             s += str(min(255,int(d*512.0))) + ' '
#             if mod(j,16) == 15:
#                 s += '\n'
#         print s
