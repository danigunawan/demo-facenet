from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
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
import Compare_Person
import Face
import Image_Face

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
Ip = '10.0.1.136'
cameraId = 11

global pnet 
global rnet 
global onet  
global sess

def main():
    global pnet 
    global rnet 
    global onet  
    global sess
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Load the model
    facenet.load_model(MODEL_DIR)
    pnet,rnet,onet = Compare_Person.load_align_parameter(gpu_memory_fraction)
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

    a = Compare_Person.compare_image_1_1(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)

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
        emb_vector_item, img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
        emb_vector_item = np.array(emb_vector_item[0])

        #luu emb va ten anh ten owner vao file json
        cv2.imwrite(img_dir, img_bb)
        result = Compare_Person.save_to_json(JSON_FILE_FACE, img_dir, owner, emb_vector_item)
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
    name_db, emb_db = Compare_Person.get_img_emb_name(JSON_FILE_FACE, owner)
    if len(name_db) == 0:
        return jsonify({"message": "no data for this person"})

    image_name.extend(name_db)
    emb_vector.extend(emb_db)

    #compare emb anh query voi tat ca emb trong database cua owner
    result = Compare_Person.compare_img_1_n(emb_vector)

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

    emb_item, img_bb = Compare_Person.img_to_emb_with_bb(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
    emb_item = np.array(emb_item[0])
    emb_vector.append(emb_item)
    image_name.append(filename_compare)
    cv2.imwrite(directory + "/" + filename_compare, img_bb)

    name_db, emb_db = Compare_Person.get_img_emb_name(JSON_FILE_FACE, owner)

    if len(name_db) == 0:
        return jsonify({"message": "no data for this person"})

    image_name.extend(name_db)
    emb_vector.extend(emb_db)

    #compare emb anh query voi tat ca emb trong database cua owner
    result = Compare_Person.compare_img_1_n(emb_vector)

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
    emb_item, img_bb = Compare_Person.img_to_emb_with_bb(image_files, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
    emb_item = np.array(emb_item[0])

    #luu vao file json
    cv2.imwrite(img_dir, img_bb)
    result = Compare_Person.save_to_json(JSON_FILE_CMT, img_dir, owner, emb_item)

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