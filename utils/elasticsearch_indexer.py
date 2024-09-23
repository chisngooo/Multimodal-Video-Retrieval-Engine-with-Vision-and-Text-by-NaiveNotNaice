import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def index_data(es, data_dir):
    def bulk_index(index_name, file_name, field_name):
        with open(f'{data_dir}/{file_name}.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        actions = [
            {
                "_index": index_name,
                "_id": key,
                "_source": {
                    'scene': key,
                    field_name: value
                }
            }
            for key, value in data.items()
        ]
        bulk(es, actions)
    
    index_name1 = 'ocr'
    if not es.indices.exists(index=index_name1):
        es.indices.create(index=index_name1)
        bulk_index(index_name1, 'OCR', 'ocr')

    index_name2 = 'ic'
    if not es.indices.exists(index=index_name2):
        es.indices.create(index=index_name2)
        bulk_index(index_name2, 'IC', 'ic')

    index_name4 = 'asr'
    if not es.indices.exists(index=index_name4):
        es.indices.create(index=index_name4)
        bulk_index(index_name4, 'ASR', 'asr')

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

