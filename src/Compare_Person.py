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
import time, datetime
from flask import Flask, request, redirect, jsonify, send_from_directory, render_template
from flask import flash
import json
from sklearn.metrics.pairwise import cosine_similarity
import requests
import urllib
import cv2
from dateutil import parser
import traceback
from Person import Person
from Image_Face import Image_Face
from Person_Black_List import Person_Black_List
from Image_Face_Black_List import Image_Face_Black_List

UPLOAD_FOLDER = '/data/image-processing/facenet/data/upload/'
UPLOAD_FOLDER_CMT = '/data/image-processing/facenet/data/cmt/'
QUERY_FOLDER = '/data/image-processing/facenet/data/query/'
MODEL_DIR = '/data/image-processing/facenet/20170511-185253/'
JSON_FILE_FACE = '/data/image-processing/facenet/data/upload/save_data_face.json'
JSON_FILE_CMT = '/data/image-processing/facenet/data/cmt/save_data_cmt.json'
IMAGE_SIZE = 180
MARGIN = 44
gpu_memory_fraction = 1


def predict_score_cmt(dist):
    res = np.exp(-(dist * dist)/(2*2.5))
    if(res < 0.8):
        return res /2
    else:
        return res

def get_img_emb_name_In_BlackList(json_file):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    path_data = []
    emb_data = []
    black_list = []
    if len(data_json['face_black_list']) != 0:
        data = data_json['face_black_list']
        for i in range(len(data)):
            image = Image_Face_Black_List(data[i]['images']["path_image"],str(data[i]['images']["emb_vector"]))
            image_cmt = Image_Face_Black_List(data[i]['image_cmt']["path_image"],str(data[i]['image_cmt']["emb_vector"]))
            person = Person_Black_List(data[i]["id"],data[i]["name"],data[i]["age"],data[i]["address"], image,image_cmt)
            black_list.append(person)
            element = data[i]['images']
            path_item = element['path_image']
            path_data.append(path_item)
            emb_item = element['emb_vector']
            emb_data.append(emb_item)
    emb_data = convertstring2vec(emb_data)
    return path_data, emb_data,black_list

def get_img_emb_name_In_BlackListById(json_file,id):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    path_data = []
    emb_data = []
    black_list = []
    if len(data_json['face_black_list']) != 0:
        data = data_json['face_black_list']
        for i in range(len(data)):
            if(data[i]["id"]==id):
                image = Image_Face_Black_List(data[i]['images']["path_image"],str(data[i]['images']["emb_vector"]))
                image_cmt = Image_Face_Black_List(data[i]['image_cmt']["path_image"],str(data[i]['image_cmt']["emb_vector"]))
                person = Person_Black_List(data[i]["id"],data[i]["name"],data[i]["age"],data[i]["address"], image,image_cmt)
                black_list.append(person)
                element = data[i]['images']
                path_item = element['path_image']
                path_data.append(path_item)
                emb_item = element['emb_vector']
                emb_data.append(emb_item)
    emb_data = convertstring2vec(emb_data)
    return path_data, emb_data,black_list
def convertstring2vec(list_emb):
    list_vec = list()
    for i in range(len(list_emb)):
        xx = list_emb[i].split(',')
        xx= np.array(xx)
        xx= xx.astype('float32')
        arr = np.array(xx)
        arr = arr.astype('float32')
        list_vec.append(arr)
    return list_vec
def get_img_emb_name(json_file, id_cmt):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    list_image_cmt = []
    list_image_face = []
    person_info = Person("","","","","","","","",[],"","")
    if len(data_json['face_data']) != 0:
        for i in range(len(data_json['face_data'])):
            data = data_json['face_data']
            if id_cmt == data[i]['id_cmt']:
                data_image = data[i]['image']
                for j in range(len(data_image)):
                    image = Image_Face(data_image[j]["name"],data_image[j]["type"],data_image[j]["path_image"],data_image[j]["vector"],data_image[j]["date"],data_image[j]["location_transaction"])
                    if data_image[j]["type"] == 0:
                        list_image_cmt.append(image)
                    else:
                        list_image_face.append(image)
                person_info = Person(data[i]["id_cmt"],data[i]["name"],data[i]["address"],data[i]["age"],data[i]["job"],data[i]["phone_number"],data[i]["last_address_transaction"],data[i]["first_address_transaction"],[], data[i]["firtdate_transaction"],data[i]["lastdate_transaction"])
    return list_image_cmt, list_image_face,person_info

def get_img_emb_name_lastest(json_file):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    list_image_face = []
    list_image_cmt=[]
    list_person = []
    if len(data_json['face_data']) != 0:
        for i in range(len(data_json['face_data'])):
            data = data_json['face_data']
            data_image = data[i]['image']
            list_image_one_person = []
            list_image_one_person_cmt = []
            for j in range(len(data_image)):
                image = Image_Face(data_image[j]["name"],data_image[j]["type"],data_image[j]["path_image"],data_image[j]["vector"],data_image[j]["date"],data_image[j]["location_transaction"])
                if data_image[j]["type"] != 0:
                    list_image_one_person.append(image)
                else:
                    list_image_one_person_cmt.append(image)
            print("93")
            print (len(list_image_one_person))
            print (len(list_image_one_person_cmt))
            try:
                list_image_face.append(list_image_one_person[getImageLastestElementInList(list_image_one_person)])
            except:
                pass
            try:
                list_image_cmt.append(list_image_one_person_cmt[getImageLastestElementInList(list_image_one_person_cmt)])   
            except:
                pass
            person_info = Person(data[i]["id_cmt"],data[i]["name"],data[i]["address"],data[i]["age"],data[i]["job"],data[i]["phone_number"],data[i]["last_address_transaction"],data[i]["first_address_transaction"],[], data[i]["firtdate_transaction"],data[i]["lastdate_transaction"])
            list_person.append(person_info)
    return  list_image_face,list_person,list_image_cmt



def getImageLastestElementInList(list_image_one_person):
    list_date =[]
    for i in range(len(list_image_one_person)):
        list_date.append(parser.parse(list_image_one_person[i].date).time())
    return list_date.index(max(list_date))

def getListVectorByListObject(list_image):
    list_vec = []
    for i in range(len(list_image)):
        xx = list_image[i].vector.split(',')
        xx= np.array(xx)
        xx= xx.astype('float32')
        arr = np.array(xx)
        arr = arr.astype('float32')
        list_vec.append(arr)
    return list_vec
def getListPathByListObject(list_image):
    list_path = []
    for i in range(len(list_image)):
        path = list_image[i].path_image
        list_path.append(path)
    return list_path

def get_all_id_face_data(json_file):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    
    list_id = []
    if len(data_json['face_data']) != 0:
        data = data_json['face_data']
        for i in range(len(data)):
            list_id.append(data[i]["id_cmt"])
    return list_id
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
    threshold = [ 0.6, 0.8, 0.92 ]  # three steps's threshold
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
    

def predict_score_rbf(dist):
    return np.exp(-(dist*dist)/(2*2.5))

def dist_to_score(dist):
    return 1.0/(1.0 + dist)

def dist_to_cos(emb1, emb2):
    a = cosine_similarity(emb1, emb2)
    return a[0]

def compare_image_1_1(image_files, image_size, margin, pnet, rnet, onet, sess):
    #start comparing
    time_start = time.time()
    #align images
    images = align_data(image_files, image_size, margin, pnet, rnet, onet)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    nrof_images = len(image_files)
    cos_sim = dist_to_cos(np.array(emb[0,:]), np.array(emb[1,:]))
    dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:],emb[1,:]))))
    res_dist = predict_score_rbf(dist)
    score_ret = res_dist

    return score_ret
def align_data_with_bb(image_paths, image_size, margin, pnet, rnet, onet,image_size_for_pretrain):
    try:
        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        
        factor = 0.709 # scale factor
        nrof_samples = len(image_paths)
        
        img_list = [None] * nrof_samples
        for i in range(nrof_samples):
            # print (i)
            b_box = []
            img = misc.imread(os.path.expanduser(image_paths[i]))
            print ("246")
            print (img.shape)

            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

            print (bounding_boxes)
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            img_to_norm = cv2.cvtColor(cropped, 37)
            img_to_norm[0] = cv2.equalizeHist(img_to_norm[0])
            prewhitened = cv2.cvtColor(img_to_norm, 39)
            res = cv2.fastNlMeansDenoisingColored(prewhitened, None, 10, 10, 7, 21)

            if len(b_box) == 0:
                print ("2")
                b_box.append(bb[0])
                b_box.append(bb[1])
                b_box.append(bb[2])
                b_box.append(bb[3])
                # cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
                crop_img = img[bb[1]:bb[3],bb[0]:bb[2],:]
                # return_img = img
                crop_img = facenet.crop(crop_img, None, image_size_for_pretrain)
                crop_img = facenet.flip(crop_img, None)
                # img = crop(img, do_random_crop, image_size)
                # img = flip(img, do_random_flip)

                resize=misc.imresize(crop_img, (image_size_for_pretrain, image_size_for_pretrain), interp='bilinear')
                return_img = resize
            aligned = misc.imresize(res, (image_size_for_pretrain, image_size_for_pretrain), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
        images = np.stack(img_list)
        return images, return_img
    except:
        traceback.print_exc()
        pass
    

def img_to_emb(image_files, image_size, margin, pnet, rnet, onet, sess):
    #align images
    images = align_data(image_files, image_size, margin, pnet, rnet, onet)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    #result = np.array(emb[0,:])

    return emb

def img_to_emb_with_bb(image_files, image_size, margin, pnet, rnet, onet, sess):
    #align images
    images, img_bb = align_data_with_bb(image_files, image_size, margin, pnet, rnet, onet,160)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)
    # emb =  np.random.rand(1,160,160,3)
    # result = np.array(emb[0,:])

    return emb, img_bb
def exclude_data_json(json_file):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    result = "no ok"
    data_json['face_data'] = []
    with open(json_file, 'w') as f:
        json.dump(data_json, f)
        result = 'ok'
    return result
def save_to_json(json_file, image_update, id_cmt,update_person):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    #dict_data = {}
    result = 'not ok'
    index = -1
    field = 'id_cmt'
    if len(data_json['face_data']) != 0:
        for i in range(len(data_json['face_data'])):
            if field in data_json['face_data'][i]:
                if id_cmt == data_json['face_data'][i]['id_cmt']:
                    index = i
                    print(index)
                    break
            else:
                break

    if index == -1:
        dict_data = json.dumps(update_person.serialize_new())
        data_json['face_data'].append(update_person.serialize_new())
    else:
        print("AVCD")
        print (index)
        for i in range(len(image_update)):
            data_json['face_data'][index]['image'].append(image_update[i].serialize())
    
    with open(json_file, 'w') as f:
        json.dump(data_json, f)
        result = 'ok'

    return result

def compare_img_1_n(emb_vector, type):
    result = []
    noof_images = len(emb_vector)
    print(noof_images)
    for i in range(len(emb_vector)):
        for j in range(len(emb_vector)):
            dist = np.sqrt(np.sum(np.square(np.subtract(np.array(emb_vector[i]), np.array(emb_vector[j])))))
            if (type ==1):
                res_dist = predict_score_rbf(dist)
                print(dist)
                if dist >= 0.85:
                    res_dist = res_dist/2
                    print ("type 1: "+ str(res_dist))
            else:
                res_dist = predict_score_cmt(dist)
                print ("type 0: "+ str(res_dist))
            result.append(res_dist)
        break
    print(len(result))
    return result
def compare_img_n_n(emb_vector):
    result = []
    noof_images = len(emb_vector)
    print(noof_images)
    for i in range(len(emb_vector)):
        res_ele = []
        if i==3:
            break
        for j in range(len(emb_vector)):
            if(j<3):
                continue
            dist = np.sqrt(np.sum(np.square(np.subtract(np.array(emb_vector[i]), np.array(emb_vector[j])))))
            res_dist = predict_score_rbf(dist)
            print(dist)
            if dist >= 0.85:
                res_dist = res_dist/2
            res_ele.append(res_dist)
        result.append(res_ele)
        
    print (result)
    result = np.array(result)
    result = result.astype('float32')
    result_max = np.max(result, axis = 0)
    index_of_res = []
    for i in range(len(result_max)):
        i,j = np.where(result == result_max[i])
        index_of_res.append(int(i[0]))
    print (result_max)
    print (index_of_res)
    return result_max,index_of_res

def compare_img_n_n_new(emb_vector,number_compare):
    result = []
    noof_images = len(emb_vector)
    print(noof_images)
    for i in range(len(emb_vector)):
        res_ele = []
        if i==number_compare:
            break
        for j in range(len(emb_vector)):
            if(j<number_compare):
                continue
            dist = np.sqrt(np.sum(np.square(np.subtract(np.array(emb_vector[i]), np.array(emb_vector[j])))))
            res_dist = predict_score_rbf(dist)
            print(dist)
            if dist >= 0.85:
                res_dist = res_dist/2
            res_ele.append(res_dist)
        result.append(res_ele)
        
    print (result)
    result = np.array(result)
    result = result.astype('float32')
    result_max = np.max(result, axis = 0)
    index_of_res = []
    for i in range(len(result_max)):
        i,j = np.where(result == result_max[i])
        index_of_res.append(int(i[0]))
    print (result_max)
    print (index_of_res)
    return result_max,index_of_res

def save_to_json_black_list(json_file,update_person):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    #dict_data = {}
    result = 'not ok'
    index = -1
    field = 'id_cmt'
    if len(data_json['face_black_list']) != 0:
        for i in range(len(data_json['face_black_list'])):
            if field in data_json['face_black_list'][i]:
                if id_cmt == data_json['face_black_list'][i]['id']:
                    index = i
                    print(index)
                    break
            else:
                break

    if index == -1:
        dict_data = json.dumps(update_person.serialize_new())
        data_json['face_black_list'].append(update_person.serialize_new())
    
    with open(json_file, 'w') as f:
        json.dump(data_json, f)
        result = 'ok'

    return result
