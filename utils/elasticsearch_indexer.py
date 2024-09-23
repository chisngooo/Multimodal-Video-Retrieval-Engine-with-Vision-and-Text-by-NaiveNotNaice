import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def index_data(es, data_dir):
    index_name1 = 'ocr'
    if not es.indices.exists(index=index_name1):
        es.indices.create(index=index_name1)
        with open(f'{data_dir}/OCR.json', 'r', encoding='utf-8') as file:
            ocr_data = json.load(file)
            for key, value in ocr_data.items():
                es.index(index=index_name1, id=key, body={
                    'scene': key,
                    'ocr': value 
                })

    index_name2 = 'ic'
    if not es.indices.exists(index=index_name2):
        es.indices.create(index=index_name2)
        with open(f'{data_dir}/IC.json', 'r', encoding='utf-8') as file:
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
        with open(f'{data_dir}/OBJECT.json', 'r') as file:
            data = json.load(file)
        es.indices.create(index=index_name3, body=mapping, ignore=400)

        def gen_docs():
            for key, objects in data.items():
                yield {
                    "_index": index_name3,
                    "_id": key,  
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
        with open(f'{data_dir}/ASR.json', 'r', encoding='utf-8') as file:
            asr_data = json.load(file)
            for key, value in asr_data.items():
                es.index(index=index_name4, id=key, body={
                    'scene': key,
                    'asr': value
                })

if __name__ == "__main__":
    es = Elasticsearch(["http://localhost:9200"])
    index_data(es, 'DataBase')
