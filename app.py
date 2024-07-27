
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import glob 
import json 

from utils.search_by_od import search_frames_with_all_objects
from utils.search_by_place import search_frames_with_any_place
from utils.query_processing import Translation
from utils.faiss import Myfaiss

# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import BertModel, BertTokenizer


# http://0.0.0.0:5001/home?index=0

import json

# Đọc dữ liệu từ các file JSON với mã hóa UTF-8
with open('DataBase/merged_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)




# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')

####### CONFIG #########
with open('combined_index_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {}
for key, value in json_dict.items():
   DictImagePath[int(key)] = value 

LenDictPath = len(DictImagePath)
bin_file='combined_faiss_index.bin'
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

    data = {'num_page': int(LenDictPath / imgperindex) + 1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=100)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

@app.route('/textsearch')
def text_search():
    print("text search")

    pagefile = []
    text_query = request.args.get('textquery')
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=100)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

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
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/placesearch')
def place_search():
    query_place = request.args.get('placequery')
    frame_ids = search_frames_with_any_place(query_place, data)

    pagefile = [{'imgpath': DictImagePath[frame_id], 'id': frame_id} for frame_id in frame_ids]
    data = {'num_page': int(LenDictPath / 100) + 1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)


@app.route('/objectsearch')
def object_search():
    query_objects = request.args.getlist('objectquery')
    frame_ids = search_frames_with_all_objects(query_objects, data)

    pagefile = [{'imgpath': DictImagePath[frame_id], 'id': frame_id} for frame_id in frame_ids]
    data = {'num_page': int(LenDictPath / 100) + 1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)


def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def search_with_bert(query, query_type, data, top_n=100):
    query_embedding = encode_text(query).numpy().reshape(1, -1)
    scores = []

    for frame_id, frame_data in data.items():
        if query_type == 'ocr':
            embedding = np.array(frame_data['ocr_embedding']).reshape(1, -1)
        elif query_type == 'asr':
            embedding = np.array(frame_data['asr_embedding']).reshape(1, -1)
        else:
            raise ValueError("Invalid query type")

        score = cosine_similarity(query_embedding, embedding)[0][0]
        scores.append((frame_id, score))

    # Sắp xếp theo điểm số và lấy top_n
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_results = sorted_scores[:top_n]
    return [frame_id for frame_id, score in top_results]


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
