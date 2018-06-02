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
import time, datetime
from flask import Flask, request, redirect, jsonify, send_from_directory, render_template
from flask import flash
import json
from sklearn.metrics.pairwise import cosine_similarity
import requests
import urllib
import cv2

global pnet 
global rnet 
global onet  
global sess

app = Flask(__name__)
app.config["MAX_FILE_SIZE"]= 5000000 #5MB

UPLOAD_FOLDER = '/data/image-processing/facenet/data/upload/'
UPLOAD_FOLDER_CMT = '/data/image-processing/facenet/data/cmt/'
QUERY_FOLDER = '/data/image-processing/facenet/data/query/'
MODEL_DIR = '/data/image-processing/facenet/20170511-185253/'
JSON_FILE_FACE = '/data/image-processing/facenet/data/upload/save_data_face.json'
JSON_FILE_CMT = '/data/image-processing/facenet/data/cmt/save_data_cmt.json'
IMAGE_SIZE = 160
MARGIN = 44
gpu_memory_fraction = 1
Ip = '10.0.15.93'
cameraId = 11

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
    dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
    res_dist = predict_score_rbf(dist)
    score_ret = res_dist

    return str(score_ret) + '/' + str(dist) + '/' + str(cos_sim)

def predict_score_rbf(dist):
    return np.exp(-(dist*dist)/(2*2.5))

def dist_to_score(dist):
    return 1.0/(1.0 + dist)

def dist_to_cos(emb1, emb2):
    a = cosine_similarity(emb1, emb2)
    return a[0]

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

def align_data_with_bb(image_paths, image_size, margin, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    
    factor = 0.709 # scale factor
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        b_box = []
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
        if len(b_box) == 0:
            b_box.append(bb[0])
            b_box.append(bb[1])
            b_box.append(bb[2])
            b_box.append(bb[3])
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
            return_img = img
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images, return_img

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
    images, img_bb = align_data_with_bb(image_files, image_size, margin, pnet, rnet, onet)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    #result = np.array(emb[0,:])

    return emb, img_bb

def save_to_json(json_file, image_name, owner_name, emb_vector):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    #dict_data = {}
    result = 'not ok'
    index = -1
    field = 'owner'
    if len(data_json['face_data']) != 0:
        for i in range(len(data_json['face_data'])):
            if field in data_json['face_data'][i]:
                if owner_name == data_json['face_data'][i]['owner']:
                    index = i
                    print(index)
                    break
            else:
                break

    if index == -1:
        images = []
        image_item = {"image_name": image_name, "emb_vector": emb_vector.tolist()}
        images.append(image_item)
        dict_data = { "owner": owner_name, "images": images}
        data_json['face_data'].append(dict_data)
    else:
        image_item = {"image_name": image_name, "emb_vector": emb_vector.tolist()}
        data_json['face_data'][index]['images'].append(image_item)

    with open(json_file, 'w') as f:
        json.dump(data_json, f)
        result = 'ok'

    return result

def get_img_emb_name(json_file, owner_name):
    data_json = json.loads(open(json_file).read().decode('utf-8'))
    name_data = []
    emb_data = []
    if len(data_json['face_data']) != 0:
        for i in range(len(data_json['face_data'])):
            if owner_name == data_json['face_data'][i]['owner']:
                for j in range(len(data_json['face_data'][i]['images'])):
                    element = data_json['face_data'][i]['images'][j]
                    name_item = element['image_name']
                    name_data.append(name_item)
                    emb_item = np.array(element['emb_vector'])
                    emb_data.append(emb_item)

                break

    return name_data, emb_data

def compare_img_1_n(emb_vector):
    result = []
    noof_images = len(emb_vector)
    print(noof_images)
    for i in range(len(emb_vector)):
        for j in range(len(emb_vector)):
            dist = np.sqrt(np.sum(np.square(np.subtract(np.array(emb_vector[i]), np.array(emb_vector[j])))))
            res_dist = predict_score_rbf(dist)
            print(dist)
            if dist >= 0.85:
                res_dist = res_dist/2
            result.append(str(res_dist).encode('utf-8'))
        break
    print(len(result))
    return result

def main():
    global pnet 
    global rnet 
    global onet 
    global sess

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Load the model
    facenet.load_model(MODEL_DIR)

    pnet, rnet, onet = load_align_parameter(gpu_memory_fraction)

    app.run(host='0.0.0.0',port=2907,debug=False)

@app.route('/compare',methods=['POST'])
def compareFace():
    if 'origin' not in request.files:

        return jsonify({"message": "no file uploaded"})

    file_origin = request.files['origin']
    file_compare = request.files['compare']

    if file_origin.filename == '':

        flash('No selected file')

        return redirect(request.url)

    print ("uploading image...")
    image_files = []
    filename_origin = file_origin.filename
    filename_compare = file_compare.filename
    image_files.append(UPLOAD_FOLDER + str(filename_origin))
    image_files.append(UPLOAD_FOLDER + str(filename_compare))

    file_origin.save(os.path.join(UPLOAD_FOLDER,filename_origin))
    file_compare.save(os.path.join(UPLOAD_FOLDER,filename_compare))

    a = compare_image_1_1(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)

    print (a)

    try:
        return json.dumps(a)
    except: 
        return json.dumps(a)

@app.route('/upload',methods=['POST'])
def upload_image():
    #lay ten owner
    owner = request.form['owner']
    directory = os.path.join(UPLOAD_FOLDER, owner)
    data = requests.get("http://"+Ip+":8080/shot.jpg").content
    
    all_image_file_in_db = []
    #time_start = time.time()
    count = 0
    if not os.path.exists(directory):
        os.makedirs(directory)

    while(True):
        #lay va luu anh tu camera ip
        image_files_item = []
        date = datetime.datetime.now()
        filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"
        img_dir = directory + "/" + filename_image
        image_files_item.append(str(os.path.join(UPLOAD_FOLDER, owner, filename_image)).encode('utf-8'))
        urllib.urlretrieve("http://"+Ip+":8080/shot.jpg",img_dir)
        arr = np.asarray(bytearray(data))
        rgb = cv2.imdecode(arr,1)
        rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_dir,rgb)
        print("done")
        all_image_file_in_db.extend(image_files_item)

        #lay emb cua anh
        print(image_files_item)
        #emb_vector_item = img_to_emb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
        emb_vector_item, img_bb = img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
        emb_vector_item = np.array(emb_vector_item[0])

        #luu emb va ten anh ten owner vao file json
        cv2.imwrite(img_dir, img_bb)
        result = save_to_json(JSON_FILE_FACE, img_dir, owner, emb_vector_item)
        print(result)

        count += 1
        if count >= 10:
            break

        time.sleep(1)

    #so sanh anh cmt voi anh trong db
    #doc emb va ten cua anh cmt
    image_name = []
    emb_vector = []
    name_db, emb_db = get_img_emb_name(JSON_FILE_CMT, owner)
    emb_vector.extend(emb_db)
    image_name.extend(name_db)
    file_cmt = os.path.join(UPLOAD_FOLDER_CMT,owner,str(name_db[0]))

    #doc emb va name tu file trong db
    name_db, emb_db = get_img_emb_name(JSON_FILE_FACE, owner)
    if len(name_db) == 0:
        return jsonify({"message": "no data for this person"})

    image_name.extend(name_db)
    emb_vector.extend(emb_db)

    #compare emb anh query voi tat ca emb trong database cua owner
    result = compare_img_1_n(emb_vector)

    #return duong link anh
    print("return value")
    try:
        return jsonify({"result_file" : str(all_image_file_in_db), "result_dist": result})
    except:
        return jsonify({"message": "Error"})         

@app.route('/compare_with_database', methods=['POST'])
def compare_with_database():

    owner = request.form['owner']
    directory = os.path.join(QUERY_FOLDER, owner)
    data = requests.get("http://"+Ip+":8080/shot.jpg").content
    date = datetime.datetime.now()
    filename_compare = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"  

    if not os.path.exists(directory):
        os.makedirs(directory)  

    print ("Comparing image...")
    #save image query vao folder owner
    image_files = []
    image_files.append(os.path.join(QUERY_FOLDER, owner, filename_compare))
    urllib.urlretrieve("http://"+Ip+":8080/shot.jpg",directory + "/" + filename_compare)
    arr = np.asarray(bytearray(data))
    rgb = cv2.imdecode(arr,1)
    rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(directory + "/" + filename_compare,rgb)
    #align va emb anh query
    image_name = []
    emb_vector = []
    #emb_item = img_to_emb(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
    emb_item, img_bb = img_to_emb_with_bb(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
    emb_item = np.array(emb_item[0])
    emb_vector.append(emb_item)
    image_name.append(filename_compare)
    cv2.imwrite(directory + "/" + filename_compare, img_bb)

    #lay tat ca emb cua owner ra tu file json 
    name_db, emb_db = get_img_emb_name(JSON_FILE_FACE, owner)

    if len(name_db) == 0:
        return jsonify({"message": "no data for this person"})

    image_name.extend(name_db)
    emb_vector.extend(emb_db)

    #compare emb anh query voi tat ca emb trong database cua owner
    result = compare_img_1_n(emb_vector)

    #return ket qua compare
    try:
        return jsonify({"image_query": str(image_files[0]), "result": result, "file_name": image_name})
    except: 
        return jsonify({"message": "Error"})    

@app.route('/upload_cmt', methods=['POST'])
def upload_cmt():
    owner = request.form['owner']
    directory = os.path.join(UPLOAD_FOLDER_CMT, owner)
    data = requests.get("http://"+Ip+":8080/shot.jpg").content
    date = datetime.datetime.now()
    filename_compare = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"  

    if not os.path.exists(directory):
        os.makedirs(directory)  

    print ("Comparing image...")
    #save image query vao folder owner
    image_files = []
    image_files.append(os.path.join(UPLOAD_FOLDER_CMT, owner, filename_compare))
    img_dir = directory + "/" + filename_compare
    urllib.urlretrieve("http://"+Ip+":8080/shot.jpg",img_dir)
    arr = np.asarray(bytearray(data))
    rgb = cv2.imdecode(arr,1)
    rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_dir,rgb)

    #align va emb anh cmt
    #emb_item = img_to_emb(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
    emb_item, img_bb = img_to_emb_with_bb(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
    emb_item = np.array(emb_item[0])

    #luu vao file json
    cv2.imwrite(img_dir, img_bb)
    result = save_to_json(JSON_FILE_CMT, img_dir, owner, emb_item)

    #return 
    print("return value")
    file_path = str(image_files[0])
    try:
        return jsonify({"result" : file_path})
    except:
        return jsonify({"message": "Error"})

@app.route('/demo',methods=['GET'])
def render_related():
    return render_template("related.html")

@app.route('/demo1',methods=['GET'])
def render_verifi():
    return render_template("hello.html")

if __name__ == "__main__":
    main()