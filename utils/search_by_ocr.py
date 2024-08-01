from elasticsearch import Elasticsearch

def search_ocr(query, index_name='ocr', size=100):
    # Kết nối đến Elasticsearch
    es = Elasticsearch(["http://localhost:9200"])

    # Tìm kiếm dữ liệu
    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["ocr"],
                "fuzziness": "AUTO"  # Tùy chọn để cho phép khớp không chính xác
            }
        },
        "size": size  # Trả về số lượng kết quả tùy ý
    }

    response = es.search(index=index_name, body=search_query)

    # Lấy danh sách các ID từ kết quả tìm kiếm
    ids = [hit['_id'] for hit in response['hits']['hits']]
    
    return ids
# In kết quả tìm kiếm
for hit in response['hits']['hits']:
    print(f"ID: {hit['_id']}, Scene: {hit['_source']['scene']}, ASR: {hit['_source']['ocr']}")