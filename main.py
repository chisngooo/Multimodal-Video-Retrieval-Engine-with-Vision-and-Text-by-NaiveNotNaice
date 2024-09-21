from fastapi import FastAPI, Request, Query, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langdetect import detect
from typing import List, Tuple, Optional
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss
from utils.faiss_sbert import Myfaiss_sbert
from utils.ElasticSearch import search_by_od, search_by_ocr, search_by_ic, search_by_asr
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from urllib.parse import unquote


es = Elasticsearch(["http://localhost:9200"])
index_name1 = 'ocr'
if not es.indices.exists(index=index_name1):
    es.indices.create(index=index_name1)
    with open('DataBase/OCR.json', 'r', encoding='utf-8') as file:
        ocr_data = json.load(file)
        for key, value in ocr_data.items():
            es.index(index=index_name1, id=key, body={
                'scene': key,
                'ocr': value 
            })

index_name2 = 'ic'
if not es.indices.exists(index=index_name2):
    es.indices.create(index=index_name2)
    with open('DataBase/IC.json', 'r', encoding='utf-8') as file:
        ic_data = json.load(file)
        for key, value in ic_data.items():
            es.index(index=index_name2, id=key, body={
                'scene': key,
                'ic': value 
            })

index_name3 = 'object'
if not es.indices.exists(index=index_name3):
    mapping = {
        "mappings": {
            "properties": {
                "objects": {
                    "type": "nested",
                    "properties": {
                        "quantity": {"type": "integer"},
                        "name": {"type": "keyword"},
                        "attribute": {"type": "keyword"}
                    }
                }
            }
        }
    }
    with open('DataBase/OBJECT.json', 'r') as file:
        data = json.load(file)
    es.indices.create(index=index_name3, body=mapping, ignore=400)


    def gen_docs():
        for key, objects in data.items():
            yield {
                "_index": index_name3,
                "_id": key,  # Use the key as the document ID
                "_source": {
                    "objects": [
                        {"quantity": obj[0], "name": obj[1], "attribute": obj[2]}
                        for obj in objects
                    ]
                }
            }
    bulk(es, gen_docs())

index_name4 = 'asr'
if not es.indices.exists(index=index_name4):
    es.indices.create(index=index_name4)
    with open('DataBase/ASR.json', 'r', encoding='utf-8') as file:
        asr_data = json.load(file)
        for key, value in asr_data.items():
            es.index(index=index_name4, id=key, body={
                'scene': key,
                'asr': value
            })
                

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/DataBase", StaticFiles(directory="DataBase"), name="DataBase")

DictImagePath, LenDictPath, MyFaiss, MyFaiss_sbert = {}, 0, None, None

async def load_data() -> Tuple[dict, int]:
    with open('data/path_index_mid.json') as json_file:
        json_dict = json.load(json_file)
        DictImagePath = {int(key): value for key, value in json_dict.items()}
        LenDictPath = len(DictImagePath)
    return DictImagePath, LenDictPath

async def load_data_mid() -> Tuple[dict, int]:
    with open('data/path_index_clip.json') as json_file:
        json_dict = json.load(json_file)
        DictImagePath = {int(key): value for key, value in json_dict.items()}
        LenDictPath = len(DictImagePath)
    return DictImagePath, LenDictPath


@app.on_event("startup")
async def startup_event():
    global DictImagePath, LenDictPath, MyFaiss, MyFaiss_sbert,DictImagePath2, LenDictPath2
    DictImagePath, LenDictPath = await load_data()
    DictImagePath2, LenDictPath2 = await load_data_mid()
    bin_file = 'data/faiss_index.bin'
    bin_file_sbert = 'data/sbert.bin'
    MyFaiss = Myfaiss(bin_file , DictImagePath2, 'cuda', Translation(), "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    MyFaiss_sbert = Myfaiss_sbert(bin_file_sbert, DictImagePath, 'cuda',Translation())
    
class QueryParams(BaseModel):
    query: Optional[str] = Query(None)
    page: int = Query(1, ge=1)
    imgid: Optional[int] = Query(None)
    faiss: int = Query(1, ge=1)
    
@app.get("/", response_class=HTMLResponse)
async def show_images(request: Request, params: QueryParams = Depends()):
    limit = 100
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = [{"id": key, "imgpath": value} for key, value in DictImagePath.items()][start_idx:end_idx]
    num_pages = (LenDictPath // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages
    })

@app.get("/img", response_class=HTMLResponse)
async def img(
    request: Request,
    params: QueryParams = Depends()
):
    if MyFaiss is None:
        return HTMLResponse(content="MyFaiss not initialized", status_code=500)
    _, list_ids, _, list_image_paths = MyFaiss.image_search(params.imgid, k=180)
    print(list_ids)
    limit = 100 
    pagefile = [{'imgpath': imgpath, 'id': int(id)} for imgpath, id in zip(list_image_paths, list_ids)]
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = pagefile[start_idx:end_idx]

    num_pages = (len(pagefile) // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages,
        "imgid": params.imgid
    })



@app.get("/clip_BigG14", response_class=HTMLResponse)
async def clip_BigG14(
    request: Request,
    params: QueryParams = Depends()
):
    limit = 100 
    list_ids = []
    _, list_ids, _, list_image_paths = MyFaiss.text_search(params.query, k=400)
    pagefile = [{'imgpath': imgpath, 'id': int(id)} for imgpath, id in zip(list_image_paths, list_ids)]
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = pagefile[start_idx:end_idx]
    
    num_pages = (len(pagefile) // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages,
        "query": params.query,
        "faiss": params.faiss
    })



@app.get("/ocr", response_class=HTMLResponse)
async def ocrsearch(
    request: Request,
    params: QueryParams = Depends()
):
    matching_frame_ids = search_by_ocr(params.query, index_name1, 180)
    print(matching_frame_ids)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
    limit = 100
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = pagefile[start_idx:end_idx]

    num_pages = (len(pagefile) // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages,
        "query": params.query
    })

@app.get("/ic", response_class=HTMLResponse)
async def icsearch(
    request: Request,
    params: QueryParams = Depends()
):
    if MyFaiss_sbert is None:
        return HTMLResponse(content="MyFaiss not initialized", status_code=500)
    _, list_ids, _, list_image_paths = MyFaiss_sbert.text_search(params.query, k=180)
    print(list_ids)
    limit = 100 
    pagefile = [{'imgpath': imgpath, 'id': int(id)} for imgpath, id in zip(list_image_paths, list_ids)]
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = pagefile[start_idx:end_idx]

    num_pages = (len(pagefile) // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages,
        "query": params.query,
    })


@app.get("/asr", response_class=HTMLResponse)
async def asrsearch(
    request: Request,
    params: QueryParams = Depends()
):
    matching_frame_ids = search_by_asr(params.query, index_name4, 180)
    print(matching_frame_ids)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
    limit = 100
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = pagefile[start_idx:end_idx]

    num_pages = (len(pagefile) // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages,
        "query": params.query
    })

@app.get("/obj", response_class=HTMLResponse)
async def objectsearch(
    request: Request,
    query: Optional[List[str]] = Query(None),
    params: QueryParams = Depends()
):

    query = [i.split() for i in [unquote(item) for item in query]]
    for i in query:
        i[1] = i[1].replace("+"," ")
        if i[0] == "None":
            pass
        else:
            i[0] = int(i[0])
    matching_frame_ids = search_by_od(query, index_name3)
    print(matching_frame_ids)
    pagefile = [{'imgpath': DictImagePath[int(frame_id)], 'id': str(frame_id)} for frame_id in matching_frame_ids]
    limit = 100
    start_idx = (params.page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = pagefile[start_idx:end_idx]

    num_pages = (len(pagefile) // limit) + 1

    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": paginated_data,
        "page": params.page,
        "num_pages": num_pages,
        "query": query
    })