from fastapi import FastAPI, Request, Query, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss
from utils.faiss_opai import Myfaiss_opai
from utils.faiss_quickgelu import Myfaiss_quigelu
from utils.faiss_robert import Myfaiss_robert
from utils.elasticsearch import search_by_od, search_by_ocr, search_by_asr
from elasticsearch import Elasticsearch
from urllib.parse import unquote
from utils.elasticsearch_indexer import index_data

es = Elasticsearch(["http://localhost:9200"])
index_data(es, 'data/es_data')
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/data", StaticFiles(directory="data"), name="data")

DictImagePath, LenDictPath,DictImagePath2, LenDictPath2, MyFaiss_bigg14_laion,MyFaiss_b32, MyFaiss_l14, MyFaiss_g14_laion, MyFaiss_bigg14_datacomp, MyFaiss_h14_apple, MyFaiss_h14_quickgelu, MyFaiss_robert = {}, 0,{}, 0, None, None, None, None, None, None, None, None

async def load_data() -> Tuple[dict, int]:
    with open('data/index/path_index_mid.json') as json_file:
        json_dict = json.load(json_file)
        DictImagePath = {int(key): value for key, value in json_dict.items()}
        LenDictPath = len(DictImagePath)
    return DictImagePath, LenDictPath

async def load_data_clip() -> Tuple[dict, int]:
    with open('data/index/path_index_clip.json') as json_file:
        json_dict = json.load(json_file)
        DictImagePath = {int(key): value for key, value in json_dict.items()}
        LenDictPath = len(DictImagePath)
    return DictImagePath, LenDictPath


@app.on_event("startup")
async def startup_event():
    global DictImagePath, LenDictPath, MyFaiss_bigg14_laion,MyFaiss_b32, MyFaiss_l14, MyFaiss_g14_laion, MyFaiss_bigg14_datacomp,MyFaiss_h14_apple, MyFaiss_h14_quickgelu, MyFaiss_robert, DictImagePath2, LenDictPath2
    DictImagePath, LenDictPath = await load_data()
    DictImagePath2, LenDictPath2 = await load_data_clip()
    bigg14_laion = 'data/bin/bigg14_laion.bin'
    b32 = 'data/bin/b32.bin'
    g14_laion = 'data/bin/g14_laion.bin'
    h14_apple = 'data/bin/h14_apple.bin'
    h14_quickgelu = 'data/bin/h14_quickgelu.bin'
    l14 = 'data/bin/l14.bin'
    bigg14_datacomp = 'data/bin/bigg14_datacomp.bin'
    robert = 'data/bin/robert.bin'
    ims = 'data/bin/is.bin'
    MyFaiss_bigg14_laion = Myfaiss(bigg14_laion , DictImagePath2, 'cuda', Translation(), "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    MyFaiss_b32 = Myfaiss_opai(b32 , DictImagePath2, 'cuda', Translation(), "ViT-B/32")
    MyFaiss_g14_laion = Myfaiss(g14_laion , DictImagePath2, 'cuda', Translation(), "hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K")
    MyFaiss_h14_apple = Myfaiss(h14_apple , DictImagePath2, 'cpu', Translation(), "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378")
    MyFaiss_h14_quickgelu = Myfaiss_quigelu(h14_quickgelu , DictImagePath2, 'cuda', Translation(), "ViT-H-14-378-quickgelu")
    MyFaiss_l14 = Myfaiss_opai(l14 , DictImagePath2, 'cuda', Translation(), "ViT-L/14@336px")
    MyFaiss_bigg14_datacomp = Myfaiss(bigg14_datacomp , DictImagePath2, 'cpu', Translation(), "hf-hub:UCSC-VLAA/ViT-bigG-14-CLIPA-datacomp1B")
    
    MyFaiss_robert = Myfaiss_robert(robert, ims, DictImagePath, 'cuda',Translation())
    
    
class QueryParams(BaseModel):
    query: Optional[str] = Query(None)
    page: int = Query(1, ge=1)
    imgid: Optional[int] = Query(None)
    faiss: int = Query(0, ge=0)
    
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
    if params.faiss == 1:
        _, list_ids, _, list_image_paths = MyFaiss_bigg14_laion.image_search(params.imgid, k=400) 
    elif params.faiss == 2:
        _, list_ids, _, list_image_paths = MyFaiss_bigg14_datacomp.image_search(params.imgid, k=400) 
    elif params.faiss == 3:
        _, list_ids, _, list_image_paths = MyFaiss_g14_laion.image_search(params.imgid, k=400) 
    elif params.faiss == 4:
        _, list_ids, _, list_image_paths = MyFaiss_h14_quickgelu.image_search(params.imgid, k=400) 
    elif params.faiss == 5:
        _, list_ids, _, list_image_paths = MyFaiss_h14_apple.image_search(params.imgid, k=400) 
    elif params.faiss == 6:
        _, list_ids, _, list_image_paths = MyFaiss_l14.image_search(params.imgid, k=400) 
    elif params.faiss == 7:
        _, list_ids, _, list_image_paths = MyFaiss_b32.image_search(params.imgid, k=400) 
    else:
        _, list_ids, _, list_image_paths = MyFaiss_robert.image_search(params.imgid, k=400) 

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
        "imgid": params.imgid,
        "faiss": params.faiss
    })


@app.get("/clip", response_class=HTMLResponse)
async def clip(
    request: Request,
    params: QueryParams = Depends()
):
    limit = 100
    list_ids = []
    if (params.faiss == 1):
        _, list_ids, _, list_image_paths = MyFaiss_bigg14_laion.text_search(params.query, k=400)
    elif (params.faiss == 2):
        _, list_ids, _, list_image_paths = MyFaiss_bigg14_datacomp.text_search(params.query, k=400)
    elif (params.faiss == 3):
        _, list_ids, _, list_image_paths = MyFaiss_g14_laion.text_search(params.query, k=400)
    elif (params.faiss == 4):
        _, list_ids, _, list_image_paths = MyFaiss_h14_quickgelu.text_search(params.query, k=400)
    elif (params.faiss == 5):
        _, list_ids, _, list_image_paths = MyFaiss_h14_apple.text_search(params.query, k=400)
    elif (params.faiss == 6):
        _, list_ids, _, list_image_paths = MyFaiss_l14.text_search(params.query, k=400)
    elif (params.faiss == 7):
        _, list_ids, _, list_image_paths = MyFaiss_b32.text_search(params.query, k=400)
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
        "query": params.query,
    })

@app.get("/ic", response_class=HTMLResponse)
async def icsearch(
    request: Request,
    params: QueryParams = Depends()
):
    if MyFaiss_robert is None:
        return HTMLResponse(content="MyFaiss not initialized", status_code=500)
    _, list_ids, _, list_image_paths = MyFaiss_robert.text_search(params.query, k=400)
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
        "query": params.query
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
