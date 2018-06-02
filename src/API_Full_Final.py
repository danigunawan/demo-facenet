from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from scipy import misc
import numpy as np
import tensorflow as tf
import numpy as np
import sys
import os
from os import walk
import argparse
import facenet
import align.detect_face
import time, datetime
from flask import Flask, request, redirect, jsonify, send_from_directory, render_template
from flask import flash
import json
from sklearn.metrics.pairwise import cosine_similarity
from base64 import *
import ast
import urllib
import cv2
import Compare_Person
import traceback
from Person import Person
from Image_Face import Image_Face
from Person_Black_List import Person_Black_List
from Image_Face_Black_List import Image_Face_Black_List
from flask import Response
from os.path import basename
import requests
import base64
from pprint import pprint
import math
import pickle
from sklearn.svm import SVC

app = Flask(__name__)
app.config["MAX_FILE_SIZE"]= 5000000 #5MB

UPLOAD_FOLDER = 'static/upload_test/'
UPLOAD_FOLDER_CMT = 'static/cmt/'
UPLOAD_FOLDER_BLACK_LIST = 'static/black_list/'
QUERY_FOLDER = 'static/query/'
MODEL_DIR = '../20170511-185253/'
JSON_FILE_FACE = '../data/upload/Face_Data.json'
JSON_FILE_FACE_PRE_CHECK= '../data/upload/Face_Data_Pre.json'
JSON_FILE_FACE_RAW= '../data/upload/Face_Data_Raw.json'
JSON_FILE_BLACK_LIST = '../data/BlackList/Face_Black_List.json'
CLASSIFIER_FILENAME_EXP = 'models/my_classifier_infore.pkl'
FOLDER_MUSTER= "static/muster"
IMAGE_SIZE = 180
MARGIN = 44
gpu_memory_fraction = 1
Ip = '192.168.1.235'
cameraId = 11
thresh = 0.8
global pnet 
global rnet 
global onet  
global sess
global model
global class_names

@app.route('/insertDataRaw',methods=['POST'])
def InsertDataRaw():
    try:
        id = request.form['id']

        directory = os.path.join(UPLOAD_FOLDER, id)

        emb_vector = []
        path_image = []
        list_all_new_Image = []

        if not os.path.exists(directory):
            os.makedirs(directory)
        # save anh cmt
        directory_cmt = os.path.join(UPLOAD_FOLDER_CMT, id)
        if not os.path.exists(directory_cmt):
            os.makedirs(directory_cmt)
        if 'myfile' not in request.files:

            return jsonify({"message": "no file uploaded"})
        image_files_item_cmt = []
        file = request.files['myfile']
        print("77")
        print (file)
        if file.filename == '':

            flash('No selected file')

            return redirect(request.url)

        print ("uploading image...")

        filename = file.filename
        print (filename)
        
        file.save(directory_cmt + "/" + filename)
        image_files_item_cmt.append(directory_cmt + "/" + filename)
        # ket thuc save anh cmt
        try:
            emb_vector_item_cmt, img_bb_cmt = Compare_Person.img_to_emb_with_bb(image_files_item_cmt, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
        except:
            print ("Loi khong nhan duoc mat cmt")
            return "Khong nhan duoc mat chung minh thu"
            pass 

        emb_vector_item_cmt = np.array(emb_vector_item_cmt[0])

        img_bb_cmt = cv2.cvtColor(img_bb_cmt, cv2.COLOR_BGR2RGB)

        cv2.imwrite((directory_cmt + "/" + filename).encode('utf-8'), img_bb_cmt)

        img_cmt = Image_Face(filename,0,directory_cmt + "/" + filename, emb_vector_item_cmt, datetime.datetime.now(),"")

        list_all_new_Image.append(img_cmt)
        
        cap = cv2.VideoCapture('http://192.168.1.5:8081')
        for i in range(3):
            ret,data = cap.read()
            image_files_item = []    
            date = datetime.datetime.now()
            
            filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"
            img_dir = directory + "/" + filename_image
            image_files_item.append(str(os.path.join(UPLOAD_FOLDER, id, filename_image)).encode('utf-8'))
            filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"

            path_image.append(img_dir)
            cv2.imwrite(directory + "/" + filename_image,data)

            rgb = data
 
            cv2.imwrite(directory + "/" + filename_image,rgb)
            try:
                emb_vector_item, img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
                emb_vector_item = np.array(emb_vector_item[0])

                emb_vector.append(emb_vector_item)
                img_bb = cv2.cvtColor(img_bb, cv2.COLOR_RGB2BGR)
                cv2.imwrite((img_dir).encode('utf-8'), img_bb)
            except:
                print ("Khong nhan phat hien duoc mat chup")
                return "Loi lay anh face"
                pass
            img_face = Image_Face(filename_image,1,directory + "/" + filename_image, emb_vector_item, datetime.datetime.now(),"")
            list_all_new_Image.append(img_face)
        person_info = Person(id,"","",1,"","","","",list_all_new_Image,datetime.datetime.now().strftime("%s %B %d, %Y"),datetime.datetime.now().strftime("%s %B %d, %Y"))
        Compare_Person.save_to_json(JSON_FILE_FACE_PRE_CHECK,list_all_new_Image,id,person_info)
        return jsonify(dict(status = "success", image_lastest = list_all_new_Image[-1].path_image))
    except:
        traceback.print_exc()
        return "error"
        pass

@app.route('/check_black_list',methods=['POST'])
def CheckBlackList():
    try:
        id = request.form['id']
        
        list_face_cmt, list_face_capture,person_info = Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id)
        list_face_capture = list_face_capture[-3:]

        print ("154 "+str(len(list_face_capture)))
        
        list_face_pre_vec = Compare_Person.getListVectorByListObject(list_face_capture)
        print ("156 "+str(len(list_face_pre_vec)))
        path_image = Compare_Person.getListPathByListObject(list_face_capture)

        emb_vector = []
        list_all_new_Image = []
        list_all_new_Image.extend(list_face_cmt)

        list_all_new_Image.extend(list_face_capture)

        path_data, emb_data,black_list = Compare_Person.get_img_emb_name_In_BlackList(JSON_FILE_BLACK_LIST)
        print ("156 "+str(type(emb_data)))
        emb_vector.extend(list_face_pre_vec)


        emb_vector.extend(emb_data)
        print ("170 "+str(len(emb_vector)))
        result,index_of_res = np.array(Compare_Person.compare_img_n_n(emb_vector))
        result = result.astype('float32')
        print (result)
        filter= np.where(result > thresh)[0]
        print (path_image)
        result = result[filter[:]]
        index_of_res= index_of_res[filter[:]]
        

        print (result)
        print (index_of_res)

        list_path =''
        for i in range(len(path_image)):
            list_path = path_image[i]+ ','
        list_path = list_path[:list_path.rfind(',')]
        
        list_result = []
        path_result = []
        if(len(filter) > 0):
            
            for i in range(len(filter)):
                list_result.append(black_list[filter[i]])
                path_result.append(path_data[filter[i]])
            json_res =[]
            for i in range(len(result)):
                res = {
                    "score": str(result[i]),
                    "image_black_list": path_result[i] ,
                    "image_query": path_image[int(index_of_res[i])]
                }
                json_res.append(res)
            json_res.sort(key=score_compare, reverse=True)
            data =[]
            for i in range(len(json_res)):
                for j in range(len(black_list)):
                    if(black_list[j].image.path_image == json_res[i]["image_black_list"]):
                        data.append(black_list[j].serialize())
                        break
            return jsonify(dict(status= 1,info = data, result=json_res,image_lastest= path_image[-1]))
        else:
            return jsonify({"status": 0, "info": [], "result": [],"image_lastest": path_image[-1]} )
    except:
        traceback.print_exc()
        print ("Loi o black list")
        return jsonify({"status": 2, "info": [], "score": [],"image_lastest": ""} )
        pass    
def score_compare(json):
    try:
        # Also convert to int since update_time will be string.  When comparing
        # strings, "10" is smaller than "2".
        return float(json['score'])
    except KeyError:
        return 0
@app.route('/checkSimilarity',methods=['POST'])
def checkSimilarity():
    try:

        print("Check mat chung minh thu")
        id = str(request.form['id_cmt'])
        print (id)
       
        
        list_all_new_Image = []

        # lay anh tu truc tiep mat va dua ra vector
        emb_vector_face=[]
        emb_vector_cmt = []
        list_image_cmt, list_face_pre,person_info_pre = Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id)

        list_all_new_Image.extend(list_image_cmt)

        list_image_cmt = list_image_cmt[-1:]

        list_face_pre = list_face_pre[-3:]

        list_image_cmt_vec = Compare_Person.getListVectorByListObject(list_image_cmt)

        emb_vector_cmt.extend(list_image_cmt_vec)

        list_face_pre_vec = Compare_Person.getListVectorByListObject(list_face_pre)

        list_path_pre = Compare_Person.getListPathByListObject(list_face_pre)

        list_all_new_Image.extend(list_face_pre)
        list_id = Compare_Person.get_all_id_face_data(JSON_FILE_FACE)
        print (list_id)
        if (id in list_id):
            emb_vector_face.extend(list_face_pre_vec)
            # Ton tai trong he thong
            print ("Ton tai trong he thong")
            list_face_cmt, list_face_capture,person_info = Compare_Person.get_img_emb_name(JSON_FILE_FACE,id)
            list_vec_cmt = Compare_Person.getListVectorByListObject(list_face_cmt)
            list_vec_capture = Compare_Person.getListVectorByListObject(list_face_capture)
            emb_vector_cmt.extend(list_vec_cmt)
            emb_vector_face.extend(list_vec_capture)

            print ("199 "+ str(len(list_face_capture)))
            print ("200 " + str(len(list_face_pre_vec)))
            print ("201 " + str(len(emb_vector_face)))
            result_cmt = Compare_Person.compare_img_1_n(emb_vector_cmt, 0)
            result_cmt = result_cmt[1:]
            image_cmt_similar_best = list_face_cmt[result_cmt.index(max(result_cmt))]

            result_face,index_of_res = Compare_Person.compare_img_n_n(emb_vector_face)
            max_value = np.argmax(result_face,axis=0)
            print ("206 "+str(max_value))
            image_face_similar_best = list_face_capture[int(max_value)]     

            image_face_similar_best = list_face_capture[index_of_res[int(max_value)]]
            # Compare_Person.save_to_json(JSON_FILE_FACE,list_all_new_Image,id,person_info)
            return jsonify({"status":1,
                            "id": id,
                            "cmt":{"score" : str(max(result_cmt)), "path_new_image": list_image_cmt[-1].path_image, "path_best_similar": image_cmt_similar_best.path_image},
                            "face":{"score" : str(result_face[max_value]), "path_new_image": list_path_pre[index_of_res[max_value]], "path_best_similar": image_face_similar_best.path_image},
                            "info": person_info.serialize()
        })
        else:
            emb_vector_face.extend(emb_vector_cmt)
            emb_vector_face.extend(list_face_pre_vec)
            score_compare = Compare_Person.compare_img_1_n(emb_vector_face,0)
            score_compare = score_compare[1:]
            
            score_compare, list_face_pre = zip(*sorted(zip(score_compare, list_face_pre),reverse=True))
            person_info_pre.image = list_all_new_Image
            # Compare_Person.save_to_json(JSON_FILE_FACE,list_face_pre,id,person_info_pre)
            json_res = []
            for i in range(len(score_compare)):
                json_item = {
                    "score": score_compare[i],
                    "path_face_compare": list_face_pre[i].path_image
                }
                json_res.append(json_item)
            return jsonify(dict(status= 0,info = person_info_pre.serialize(), score_best = max(score_compare), path_new_cmt= list_image_cmt[-1].path_image, detail_res_compare= json_res))
    except:
        traceback.print_exc()
        return "error"
        pass
    

@app.route('/realtime_image_vs_image',methods=['POST'])
def realtime_existed():
    try:
        # 0 : black list
        # 1: khong phai black list
        id_raw = request.form['id_raw']
        id_query = request.form['id_query']
        type = int(request.form['type'])
        cap = cv2.VideoCapture('http://192.168.1.5:8081')

        ret, data = cap.read()

        # data=cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        directory = os.path.join(UPLOAD_FOLDER, id_query)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_files_item = []
        em_vec_ss_face=[]
        list_all_new_Image = []
        date = datetime.datetime.now()
        # lay anh cmt va dua ra vector
        filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"
        
        # image_files_item.append(str(os.path.join(UPLOAD_FOLDER, id_raw, filename_image)).encode('utf-8'))
        image_files_item.append(str(os.path.join(UPLOAD_FOLDER, id_query, filename_image)).encode('utf-8'))

        # arr = np.asarray(bytearray(data))
        # rgb= cv2.imdecode(arr,1)
        # rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = data
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(directory + "/" + filename_image,rgb)

        _,____, person_info= Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id_query)
        try:
            emb_vector_item,img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
            emb_vector_item = np.array(emb_vector_item[0])

            em_vec_ss_face.append(emb_vector_item)
            img_bb = cv2.cvtColor(img_bb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(directory + "/" + filename_image, img_bb)
            img_face = Image_Face(filename_image,1,directory + "/" + filename_image, emb_vector_item, datetime.datetime.now(),"")
            list_all_new_Image.append(img_face)
            if(type == 1):
                list_face_cmt, list_face_capture,___ = Compare_Person.get_img_emb_name(JSON_FILE_FACE,id_raw)
                list_vec_capture = Compare_Person.getListVectorByListObject(list_face_capture)

                em_vec_ss_face.extend(list_vec_capture)
            else:
                _,list_face_capture,__ = Compare_Person.get_img_emb_name_In_BlackListById(JSON_FILE_BLACK_LIST,id_raw)
                em_vec_ss_face.extend(list_face_capture)
                
            result_face = Compare_Person.compare_img_1_n(em_vec_ss_face,1)
            result_face = result_face[1:]

            Compare_Person.save_to_json(JSON_FILE_FACE_PRE_CHECK,list_all_new_Image,id_query,person_info)
            return jsonify(dict(score_cmt = max(result_face),new_image = directory + "/" + filename_image))
        except:
            traceback.print_exc()
            return jsonify(dict(score_cmt = 0,new_image = directory + "/" + filename_image))
            pass
    except:
        traceback.print_exc()
        return "error"
        pass

@app.route('/realtime_image_vs_cmt',methods=['POST'])
def realtime_not_existed():
    try:
        id_raw = request.form['id_raw']
        id_query = request.form['id_query']

        cap = cv2.VideoCapture('http://192.168.1.5:8081')

        ret, data = cap.read()

        # data=cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        directory = os.path.join(UPLOAD_FOLDER, id_query)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_files_item = []
        em_vec_ss_cmt=[]

        list_all_new_Image = []
        date = datetime.datetime.now()
        # lay anh cmt va dua ra vector
        filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"
        
        image_files_item.append(str(os.path.join(UPLOAD_FOLDER, id_query, filename_image)).encode('utf-8'))
        
        rgb = data
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(directory + "/" + filename_image,rgb)
        try:
            emb_vector_item,img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
            emb_vector_item = np.array(emb_vector_item[0])
            em_vec_ss_cmt.append(emb_vector_item)
            img_bb = cv2.cvtColor(img_bb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(directory + "/" + filename_image, img_bb)
            img_face = Image_Face(filename_image,1,directory + "/" + filename_image, emb_vector_item, datetime.datetime.now(),"")
            list_all_new_Image.append(img_face)

            list_face_cmt, list_face_capture,person_info = Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id_raw)
            list_vec_cmt = Compare_Person.getListVectorByListObject(list_face_cmt)

            em_vec_ss_cmt.extend(list_vec_cmt)

            result_cmt = Compare_Person.compare_img_1_n(em_vec_ss_cmt,0)
            result_cmt = result_cmt[1:]

            Compare_Person.save_to_json(JSON_FILE_FACE_PRE_CHECK,list_all_new_Image,id_query,person_info)
            return jsonify(dict(score_cmt = max(result_cmt),new_image = directory + "/" + filename_image))
        except:
            return jsonify(dict(score_cmt = 0,new_image = directory + "/" + filename_image))
    except:
        traceback.print_exc()
        return "error"
        pass


@app.route('/check_fake',methods=['POST'])
def check_fake():
    try:
        id = request.form['id']

        directory = os.path.join(UPLOAD_FOLDER, id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_files_item = []

        em_vec_ss_face=[]
        # lay ra vector da luu o file tam
        _, list_face_pre,person_info_pre = Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id)
        list_face_pre = list_face_pre[-3:]
        list_face_pre_vec = Compare_Person.getListVectorByListObject(list_face_pre)
        em_vec_ss_face.extend(list_face_pre_vec)
        list_path_pre = Compare_Person.getListPathByListObject(list_face_pre)
        # lay ra tat ca anh moi nhat 
        list_image_face,list_person,list_image_cmt = Compare_Person.get_img_emb_name_lastest(JSON_FILE_FACE)
        list_image_face_vec = Compare_Person.getListVectorByListObject(list_image_face)
        path_image_face = Compare_Person.getListPathByListObject(list_image_face)
        em_vec_ss_face.extend(list_image_face_vec)

        result,index_of_res = np.array(Compare_Person.compare_img_n_n(em_vec_ss_face))
        result = result.astype('float32')
        filter= np.where(result > 0.8)[0]
        
        result = result[filter[:]]
        index_of_res= index_of_res[filter[:]]

        if(len(filter) > 0):
            print (len(result))
            print (len(index_of_res))
            result, index_of_res = zip(*sorted(zip(result, index_of_res),reverse=True))
            json_res = []
            for i in range(len(filter)):
                print (index_of_res[i])
                item = list_person[filter[i]]
                json_data = item.serialize()
                json_data["score"] = str(result[i])
                json_data["image_raw"]= list_path_pre[int(index_of_res[i])]
                json_data["image_query"] = list_image_face[filter[i]].path_image
                try:
                    json_data["image_cmt"] = list_image_cmt[filter[i]].path_image
                except:
                    json_data["image_cmt"] = ""
                    pass
                json_res.append(json_data)
            print (json_res)
            return jsonify(dict(status = 1, result = json_res))
        else:
            return jsonify(dict(status = 0 ,result = [],image_compare = '',image_raw=list_path_pre[-1]))
    except:
        traceback.print_exc()
        return "error"
        pass
@app.route('/addToDB',methods=['POST'])
def clear_cache():
    try: 
        id = request.form['id']
        list_image_cmt,list_face_pre,person_info_pre = Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id)
        list_all_new_Image = []
        list_all_new_Image.extend(list_image_cmt)
        list_all_new_Image.extend(list_face_pre)
        person_info_pre.image = list_all_new_Image
        Compare_Person.save_to_json(JSON_FILE_FACE,list_all_new_Image,id,person_info_pre)
        Compare_Person.exclude_data_json(JSON_FILE_FACE_PRE_CHECK)
        return jsonify(dict(status = 1))
    except:
        traceback.print_exc()
        return jsonify(dict(status = 0))
        pass
@app.route('/add-black-list',methods=['POST'])
def InsertDataBlackList():
    try:
        id = request.form['blackcmt']
        name = str(request.form['blackname'])

        directory = os.path.join(UPLOAD_FOLDER, id)

        emb_vector = []
        path_image = []
        list_all_new_Image = []

        if not os.path.exists(directory):
            os.makedirs(directory)
        # save anh cmt
        directory_cmt = os.path.join(UPLOAD_FOLDER_BLACK_LIST, id)
        if not os.path.exists(directory_cmt):
            os.makedirs(directory_cmt)
        print (1)
        if 'myfile' not in request.files and 'myfile_face' not in request.files:
            return jsonify({"message": "no file uploaded"})
        image_files_item_cmt = []
        file = request.files['myfile']
        print (file)
        if file.filename == '':

            flash('No selected file')

            return redirect(request.url)

        print ("uploading image...")

        filename = file.filename
        print (filename)
        
        file.save(directory_cmt + "/" + filename)
        image_files_item_cmt.append(directory_cmt + "/" + filename)
        ############
        
        image_files_item_face = []
        file_face = request.files['myfile_face']
        print (file_face)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        print ("uploading image...")

        filename_face = file_face.filename
        print (filename_face)
        
        file_face.save(directory_cmt + "/" + filename_face)
        image_files_item_face.append(directory_cmt + "/" + filename_face)

        try:
            emb_vector_item_cmt, img_bb_cmt = Compare_Person.img_to_emb_with_bb(image_files_item_cmt, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)

            emb_vector_item_cmt = np.array(emb_vector_item_cmt[0])

            img_bb_cmt = cv2.cvtColor(img_bb_cmt, cv2.COLOR_RGB2BGR)

            cv2.imwrite((directory_cmt + "/" + filename).encode('utf-8'), img_bb_cmt)

            emb_vector_item_face, img_bb_face = Compare_Person.img_to_emb_with_bb(image_files_item_face, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)

            emb_vector_item_face = np.array(emb_vector_item_face[0])

            img_bb_face = cv2.cvtColor(img_bb_face, cv2.COLOR_RGB2BGR)

            cv2.imwrite((directory_cmt + "/" + filename_face).encode('utf-8'), img_bb_face)


        except:
            print ("Loi khong nhan duoc mat cmt")
            return "Khong nhan duoc mat chung minh thu"
            pass 
        img_cmt = Image_Face_Black_List(directory_cmt + "/" + filename, emb_vector_item_cmt)
        img_face = Image_Face_Black_List(directory_cmt + "/" + filename_face, emb_vector_item_face)
        person_info = Person_Black_List(id,name,"","",img_face,img_cmt)
        # person_info = Person_Black_List(id,name,"","",img_face,img_face)
        try:
            Compare_Person.save_to_json_black_list(JSON_FILE_BLACK_LIST,person_info)
        except:
            traceback.print_exc()
            pass
        
        return "success"
    except:
        return "error"
        pass
@app.route('/getblack_list',methods=['POST'])
def get_Black_List():
    _,__, black_list = Compare_Person.get_img_emb_name_In_BlackList(JSON_FILE_BLACK_LIST)
    return jsonify(dict(face_black_list = [idx.serialize() for idx in black_list]))

@app.route('/detectFaceData',methods=['POST'])
def detectFaceData():
    try:
    	for (dirpath, dirnames, filenames) in walk("images"):
    		for filename in filenames:
        		try:
	        		emb_vector_item_cmt, img_bb_cmt = Compare_Person.img_to_emb_with_bb(dirpath + "/" + filename, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
	        		
	        		img_bb_cmt = cv2.cvtColor(img_bb_cmt, cv2.COLOR_BGR2RGB)
	        		cv2.imwrite((dirpath + "/faces/" + filename).encode('utf-8'), img_bb_cmt)

	        		#luu file json
	        		img_cmt = Image_Face(filename, 0, dirpath + "/faces/" + filename, emb_vector_item_cmt, datetime.datetime.now(), "")
	        		Compare_Person.save_to_json(JSON_FILE_FACE_RAW, list_all_new_Image, dirnames, "person_info")
        		except:
        			pass
    except:
        print ("Loi khong nhan duoc mat cmt")
        return "Khong nhan duoc mat chung minh thu"
        pass     

@app.route('/demo',methods=['GET'])
def render_related():
    # with open(JSON_FILE_BLACK_LIST) as data_file:    
    #   data = json.load(data_file)

    # black_list = data["face_black_list"]
    # print (black_list)
    return render_template("related.html")

@app.route('/demo-1',methods=['GET'])
def render_verified():
    # with open(JSON_FILE_BLACK_LIST) as data_file:    
    #   data = json.load(data_file)

    # black_list = data["face_black_list"]
    # print (black_list)
    return render_template("verified.html")

@app.route('/classifier',methods=['POST'])
def classifier():
    # with open('data_face.json') as data_file:
    # 	data = json.load(data_file)
    time = datetime.datetime.now()
    if not os.path.exists(FOLDER_MUSTER):
        os.makedirs(FOLDER_MUSTER)
    date = datetime.datetime.now()

    # print (date)
    cap = cv2.VideoCapture('http://14.162.147.186:8080/video')

    ret, data = cap.read()
    # print (data)
    # data = cv2.imread("/mnt/c/Users/ASUS/Desktop/image.jpg",1)
    # num_rows, num_cols = data.shape[:2]
    # # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 270, 1)
    # # data = cv2.warpAffine(data, rotation_matrix, (num_rows, num_cols))
    filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"
    image_files_item = list()

    # image_files_item.append(str(os.path.join(FOLDER_MUSTER,filename_image)).encode('utf-8'))
    image_files_item.append(FOLDER_MUSTER + "/" + filename_image)

    cv2.imwrite(FOLDER_MUSTER + "/" + filename_image,data)

    # rgb = data
    
    try:
        emb_vector_item, img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
        emb_array = np.array(emb_vector_item[0])

        img_bb = cv2.cvtColor(img_bb, cv2.COLOR_RGB2BGR)
        filename_face = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+"_face.png"
        cv2.imwrite((FOLDER_MUSTER + "/" + filename_face).encode('utf-8'), img_bb)
    except:
        print (traceback.print_exc())
        print ("Khong nhan phat hien duoc mat chup")
        return jsonify({"message": "NONE"})
        pass
    # print (emb_array)
    print (emb_array.shape)
    emb_array = np.reshape(emb_array,(-1,128))
    # emb_array = np.array(str(data["face_data"][0]["images"][0]["emb_vec"]).split(','), dtype=float)
    print (emb_array.shape)
    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    print (class_names[best_class_indices[0]])
    print (best_class_probabilities[0])
    print ("Time classify: "+ str(datetime.datetime.now()-time))
    if best_class_probabilities[0] * 100 >= 0:
    	return jsonify({"message": class_names[best_class_indices[0]] + "|" + str(best_class_probabilities[0])+"|"+FOLDER_MUSTER + "/" + filename_image})	
    else:
    	return jsonify({"message": FOLDER_MUSTER + "/" + filename_image})

def main():
    global pnet 
    global rnet 
    global onet  
    global sess
    global model
    global class_names
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Load the model
    facenet.load_model(MODEL_DIR)
    pnet,rnet,onet = Compare_Person.load_align_parameter(gpu_memory_fraction)

    with open(CLASSIFIER_FILENAME_EXP, 'rb') as infile:
		(model, class_names) = pickle.load(infile)

    app.run(host='0.0.0.0',port=2907,debug=False)
    # app.run(host='0.0.0.0',port=5000,debug=False)

if __name__ == "__main__":
    main()

