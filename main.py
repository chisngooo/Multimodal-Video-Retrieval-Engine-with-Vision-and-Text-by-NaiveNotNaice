from fastapi import FastAPI, Request, Query, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss
from utils.faiss_sbert import Myfaiss_sbert
from utils.ElasticSearch import search_by_od, search_by_ocr, search_by_ic, search_by_asr
from elasticsearch import Elasticsearch
from urllib.parse import unquote
from utils.elasticsearch_indexer import index_data

es = Elasticsearch(["http://localhost:9200"])
index_data(es, 'DataBase')
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
    matching_frame_ids = search_by_ocr(params.query, "ocr", 180)
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
    _, list_ids, _, list_image_paths = MyFaiss_sbert.text_search(params.query, k=400)
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
    matching_frame_ids = search_by_asr(params.query, "asr", 180)
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
    matching_frame_ids = search_by_od(query, "object")
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