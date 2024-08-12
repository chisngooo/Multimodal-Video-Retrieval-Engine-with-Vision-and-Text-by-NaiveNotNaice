
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import numpy as np
import pandas as pd
import glob 
import json 

from utils.search_by_od import search_od
from utils.search_by_place import search_place
from utils.search_by_asr import search_video_scenes
from utils.search_by_ocr import search_ocr
from utils.query_processing import Translation
from utils.faiss import Myfaiss
from elasticsearch import Elasticsearch


# http://0.0.0.0:5001/home?index=0

# Đọc dữ liệu từ các file JSON với mã hóa UTF-8

# Kết nối đến Elasticsearch
es = Elasticsearch(["http://localhost:9200"])





# Index dữ liệu vào Elasticsearch
# Tên chỉ mục
index_name1 = 'ocr'

# Đảm bảo rằng chỉ mục đã tồn tại, nếu không thì tạo một chỉ mục mới
if not es.indices.exists(index=index_name1):
    es.indices.create(index=index_name1)
        # Đọc dữ liệu từ file JSON và index vào Elasticsearch
    with open('DataBase/OCR.json', 'r', encoding='utf-8') as file:
        ocr_data = json.load(file)
        # Index dữ liệu vào Elasticsearch
        for key, value in ocr_data.items():
            es.index(index=index_name1, id=key, body={
                'scene': key,
                'ocr': value  # Lưu trực tiếp giá trị ASR
            })


index_name2 = 'place'
# Đảm bảo rằng chỉ mục đã tồn tại, nếu không thì tạo một chỉ mục mới
if not es.indices.exists(index=index_name2):
    es.indices.create(index=index_name2)
    # Đọc dữ liệu từ file JSON và index vào Elasticsearch
    with open('DataBase/PLACE.json', 'r', encoding='utf-8') as file:
        place_data = json.load(file)
        # Index dữ liệu vào Elasticsearch
        for key, value in place_data.items():
            es.index(index=index_name2, id=key, body={
                'scene': key,
                'place': value  # Lưu giá trị trực tiếp vào trường 'place'
            })



index_name3 = 'object'
# Đảm bảo rằng chỉ mục đã tồn tại, nếu không thì tạo một chỉ mục mới
if not es.indices.exists(index=index_name3):
    es.indices.create(index=index_name3)
        # Đọc dữ liệu từ file JSON và index vào Elasticsearch
    with open('DataBase/OBJECT.json', 'r', encoding='utf-8') as file:
        object_data = json.load(file)
        # Index dữ liệu vào Elasticsearch
        for key, value in object_data.items():
            if isinstance(value, list):
                es.index(index=index_name3, id=key, body={
                    'scene': key,
                    'objects': ' '.join(value)  # Chuyển danh sách objects thành chuỗi
                })
            else:
                print(f"Skipping {key}: Data is not a list")
            
            
            
index_name4 = 'asr'
# Đảm bảo rằng chỉ mục đã tồn tại, nếu không thì tạo một chỉ mục mới
if not es.indices.exists(index=index_name4):
    es.indices.create(index=index_name4)
    # Đọc dữ liệu từ file JSON và index vào Elasticsearch
    with open('DataBase/ASR.json', 'r', encoding='utf-8') as file:
        asr_data = json.load(file)
        # Index dữ liệu vào Elasticsearch
        for key, value in asr_data.items():
            es.index(index=index_name4, id=key, body={
                'scene': key,
                'asr': value  # Lưu trực tiếp giá trị ASR
            })
                


# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'  # Thay thế bằng khóa bảo mật của bạn

####### CONFIG #########
with open('path_index.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {}
for key, value in json_dict.items():
   DictImagePath[int(key)] = value 

LenDictPath = len(DictImagePath)
bin_file='faiss_index.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")
########################

@app.route('/home')
@app.route('/')
def thumbnailimg():
    print("load_iddoc")

    index = request.args.get('index')
    if index is None:
        index = 0
    else:
        index = int(index)

    imgperindex = 100
    LenDictPath = len(DictImagePath)

    pagefile = []
    page_filelist = []
    list_idx = []

    if LenDictPath - 1 > index + imgperindex:
        first_index = index * imgperindex
        last_index = index * imgperindex + imgperindex
    else:
        first_index = index * imgperindex
        last_index = LenDictPath

    tmp_index = first_index
    while tmp_index < last_index:
        page_filelist.append(DictImagePath[tmp_index])
        list_idx.append(tmp_index)
        tmp_index += 1    

    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': id})

    datapage = {'num_page': int(LenDictPath / imgperindex) + 1, 'pagefile': pagefile}
    
    return render_template('home.html', data=datapage)

@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=120)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    datapage = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=datapage)

@app.route('/textsearch')

def text_search():
    start_time = time.time()
    global data
    text_query = request.args.get('textquery')
    
    # Sử dụng MyFaiss để tìm kiếm văn bản với số kết quả k=120
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=120)

    imgperindex = 100  # Số hình ảnh trên mỗi trang

    # Tạo danh sách các khung hình và ID tương ứng
    pagefile = [{'imgpath': imgpath, 'id': int(id)} for imgpath, id in zip(list_image_paths, list_ids)]

    num_page = (LenDictPath // imgperindex) + 1
    datapage = {'num_page': num_page, 'pagefile': pagefile}
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    return render_template('home.html', data=datapage)

@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    # fpath = fpath
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280,720))

    # print(img.shape)
    # img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
    #                2.23, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/placesearch')
def place_search():
    global data
    query_place = request.args.get('placequery')
    matching_frame_ids = search_place(query_place, "place", 100)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
    num_page = (LenDictPath // 100) + 1
    datapage = {'num_page': num_page, 'pagefile': pagefile}
    return render_template('home.html', data=datapage)



@app.route('/objectsearch')
def object_search():
    query_objects = request.args.getlist('objectquery')
    print(query_objects)
    # Tìm kiếm các frame với OCR tương ứng
    matching_frame_ids = search_od(query_objects, "object", 100)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
    num_page = (LenDictPath // 100) + 1
    datapage = {'num_page': num_page, 'pagefile': pagefile}
    return render_template('home.html', data=datapage)


@app.route('/ocrsearch')
def ocr_search():
    query_ocr = request.args.get('query')
    # Tìm kiếm các frame với OCR tương ứng
    matching_frame_ids = search_ocr(query_ocr, "ocr", 100)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
    num_page = (LenDictPath // 100) + 1
    datapage = {'num_page': num_page, 'pagefile': pagefile}
    return render_template('home.html', data=datapage)



@app.route('/asrsearch')
def asr_search():
    query_asr = request.args.get('query')
    # Tìm kiếm các frame với ASR tương ứng
    matching_frame_ids = search_video_scenes(query_asr, "asr", 100)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
   
    num_page = (LenDictPath // 100) + 1
    datapage = {'num_page': num_page, 'pagefile': pagefile}
    
    return render_template('home.html', data=datapage)




if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
