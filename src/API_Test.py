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

app = Flask(__name__)
app.config["MAX_FILE_SIZE"]= 5000000 #5MB

UPLOAD_FOLDER = '../data/upload_test/'
UPLOAD_FOLDER_CMT = '../data/cmt/'
QUERY_FOLDER = '../data/query/'
MODEL_DIR = '../20170511-185253/'
JSON_FILE_FACE = '../data/upload/Face_Data.json'
JSON_FILE_FACE_PRE_CHECK= '../data/upload/Face_Data_Pre.json'
JSON_FILE_BLACK_LIST = '../data/BlackList/Face_Black_List.json'
IMAGE_SIZE = 160
MARGIN = 44
gpu_memory_fraction = 1
Ip = '10.0.2.96'
cameraId = 11
thresh = 0.1
global pnet 
global rnet 
global onet  
global sess

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
            return jsonify(dict(status = "error", message = "Khong nhan nhan dang duoc mat cmt"))
            pass 

        emb_vector_item_cmt = np.array(emb_vector_item_cmt[0])

        img_bb_cmt = cv2.cvtColor(img_bb_cmt, cv2.COLOR_BGR2RGB)

        cv2.imwrite((directory_cmt + "/" + filename).encode('utf-8'), img_bb_cmt)

        img_cmt = Image_Face(filename,0,directory_cmt + "/" + filename, emb_vector_item_cmt, datetime.datetime.now(),"")

        list_all_new_Image.append(img_cmt)
        
        
        for i in range(3):
            data = requests.get("http://"+Ip+":8080/shot.jpg").content

            image_files_item = []    
            date = datetime.datetime.now()

            filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"
            img_dir = directory + "/" + filename_image
            image_files_item.append(str(os.path.join(UPLOAD_FOLDER, id, filename_image)).encode('utf-8'))
            filename_image = str(cameraId) + "_" + str(date.strftime("%Y_%m_%d_%H_%M_%S"))+".png"

            path_image.append(img_dir)
            urllib.urlretrieve("http://"+Ip+":8080/shot.jpg",directory + "/" + filename_image)

            arr = np.asarray(bytearray(data))
            rgb = cv2.imdecode(arr,1)
            rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            cv2.imwrite(directory + "/" + filename_image,rgb)
            print (arr.shape)
            try:
                emb_vector_item, img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
                emb_vector_item = np.array(emb_vector_item[0])

                emb_vector.append(emb_vector_item)
            
                cv2.imwrite((img_dir).encode('utf-8'), img_bb)
                img_face = Image_Face(filename_image,1,directory + "/" + filename_image, emb_vector_item, datetime.datetime.now(),"")
                list_all_new_Image.append(img_face)
            except:
                print ("Khong nhan phat hien duoc mat chup")
                pass
            
        if len(list_all_new_Image) == 1:
            return jsonify(dict(status = "error", message = "Khong nhan nhan dang duoc mat chup"))
        person_info = Person(id,"","",1,"","","","",list_all_new_Image,datetime.datetime.now().strftime("%s %B %d, %Y"),datetime.datetime.now().strftime("%s %B %d, %Y"))
        Compare_Person.save_to_json(JSON_FILE_FACE_PRE_CHECK,list_all_new_Image,id,person_info)
        return jsonify(dict(status = "success", image_lastest = list_all_new_Image[-1].path_image))
    except:
        return "error"
        pass

@app.route('/check_black_list',methods=['POST'])
def CheckBlackList():
    try:
        id = request.form['id']
        
        list_face_cmt, list_face_capture,person_info = Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id)
        print ("154 "+str(len(list_face_capture)))
        list_face_pre_vec = Compare_Person.getListVectorByListObject(list_face_capture)
        print ("156 "+str(len(list_face_pre_vec)))
        path_image = Compare_Person.getListPathByListObject(list_face_capture)

        emb_vector = []
        list_all_new_Image = []
        list_all_new_Image.extend(list_face_cmt)

        list_all_new_Image.extend(list_face_capture)

        path_data, emb_data,black_list = Compare_Person.get_img_emb_name_In_BlackList(JSON_FILE_BLACK_LIST)
        print ("156 "+str(len(emb_data)))
        emb_vector.extend(list_face_pre_vec)

        emb_vector.extend(emb_data)
        print ("170 "+str(len(emb_vector)))
        result,index_of_res = np.array(Compare_Person.compare_img_n_n_new(emb_vector, len(list_face_capture)))
        result = result.astype('float32')
        print (result)
        filter= np.where(result > thresh)[0]
        print (path_image)
        result = result[filter[:]]
        index_of_res= index_of_res[filter[:]]
        
        list_path =''
        for i in range(len(path_image)):
            list_path = path_image[i]+ ','
        list_path = list_path[:list_path.rfind(',')]
    	
        list_result = []
        path_result = []
        if(len(filter) > 0): 
            print ("1")
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
            return jsonify(dict(status= 1,info = [ob.serialize() for ob in black_list], result=json_res,image_lastest= path_image[-1]))
        else:
            return jsonify({"status": 0, "info": [], "result": [],"image_lastest": path_image[-1]} )
    except:
        traceback.print_exc()
        print ("Loi o black list")
        return jsonify({"status": 2, "info": [], "score": [],"image_lastest": ""} )
        pass    

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

            result_face,index_of_res = Compare_Person.compare_img_n_n_new(emb_vector_face, len(list_face_capture))
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

        data = requests.get("http://"+Ip+":8080/shot.jpg").content
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
        arr = np.asarray(bytearray(data))
        rgb= cv2.imdecode(arr,1)
        rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(directory + "/" + filename_image,rgb)

        _,____, person_info= Compare_Person.get_img_emb_name(JSON_FILE_FACE_PRE_CHECK,id_query)
        try:
            emb_vector_item,img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
            emb_vector_item = np.array(emb_vector_item[0])

            em_vec_ss_face.append(emb_vector_item)
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

        data = requests.get("http://"+Ip+":8080/shot.jpg").content
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
        
        arr = np.asarray(bytearray(data))
        rgb= cv2.imdecode(arr,1)
        rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(directory + "/" + filename_image,rgb)
        try:
            emb_vector_item,img_bb = Compare_Person.img_to_emb_with_bb(image_files_item, IMAGE_SIZE, MARGIN, pnet, rnet, onet, sess)
            emb_vector_item = np.array(emb_vector_item[0])
            em_vec_ss_cmt.append(emb_vector_item)
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

        list_face_pre_vec = Compare_Person.getListVectorByListObject(list_face_pre)
        em_vec_ss_face.extend(list_face_pre_vec)
        list_path_pre = Compare_Person.getListPathByListObject(list_face_pre)
        # lay ra tat ca anh moi nhat 
        list_image_face,list_person,list_image_cmt = Compare_Person.get_img_emb_name_lastest(JSON_FILE_FACE)
        list_image_face_vec = Compare_Person.getListVectorByListObject(list_image_face)
        path_image_face = Compare_Person.getListPathByListObject(list_image_face)
        em_vec_ss_face.extend(list_image_face_vec)

        result,index_of_res = np.array(Compare_Person.compare_img_n_n_new(em_vec_ss_face, len(list_face_pre)))
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
    
@app.route('/demo',methods=['GET'])
def render_related():
    # with open(JSON_FILE_BLACK_LIST) as data_file:    
    # 	data = json.load(data_file)

    # black_list = data["face_black_list"]
    # print (black_list)
    return render_template("related.html")
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
    # app.run(host='0.0.0.0',port=5000,debug=False)

if __name__ == "__main__":
    main()

