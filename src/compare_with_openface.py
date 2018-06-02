"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import time
import cv2

sys.path.append('/mnt/3TB-disk3/data/image-processing/openface')

import openface

def main(args):

    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)

    nrof_images = len(args.image_files)

    #align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, args.image_files[i]))
    print('')
    
    # Print distance matrix
    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %1d     ' % i, end='')
    print('')
    for i in range(nrof_images):
        print('%1d  ' % i, end='')
        for j in range(nrof_images):
            d = getRep(images[i], net) - getRep(images[j], net)
            dist = np.sqrt(np.sum(np.square(d)))
            #res_dist = predict_score(dist)
            res_dist = dist
            print('  %1.4f  ' % res_dist, end='')
        print('')
 
def getRep(rgbImg, net):

    # #get face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg)
    # if bb is None:
    #     raise Exception("Unable to find a face")

    # #align image
    # alignedFace = align.align(args.imgDim, rgbImg, bb,
    #                           landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    # if alignedFace is None:
    #     raise Exception("Unable to align image")

    #embedding face
    rep = net.forward(rgbImg)
    return rep           

def predict_score(dist):
    return np.exp(-(dist*dist)/(2*2.5))

            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        #preprocessing
        #cv2.imshow("i", cropped)
        img_to_norm = cv2.cvtColor(cropped, 37)
        img_to_norm[0] = cv2.equalizeHist(img_to_norm[0])
        #cv2.imshow("hist", img_to_norm)
        prewhitened = cv2.cvtColor(img_to_norm, 39)
        res = cv2.fastNlMeansDenoisingColored(prewhitened,None,10,10,7,21)
        #res = cv2.fastNlMeansDenoisingColored(cropped,None,10,10,7,21)
        #cv2.imshow("img",res)

        aligned = misc.imresize(res, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)

        #luu vao list
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def load_align_parameter(gpu_memory_fraction):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    return pnet, rnet, onet

def align_data(image_paths, image_size, margin, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    dlibModelDir = '/mnt/3TB-disk3/data/image-processing/openface/models/dlib'
    openfaceModelDir = '/mnt/3TB-disk3/data/image-processing/openface/models/openface'

    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    #argument openface
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parser.parse_args()
    main(parse_arguments(sys.argv[1:]))
    print("time done: %s" % (time.time()-time_start))
