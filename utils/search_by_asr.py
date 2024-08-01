from elasticsearch import Elasticsearch

def search_video_scenes(query, index_name='video_scenes', size=100):
    # Kết nối đến Elasticsearch
    es = Elasticsearch(["http://localhost:9200"])

    # Tìm kiếm dữ liệu
    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["asr"],
                "fuzziness": "AUTO"  # Tùy chọn để cho phép khớp không chính xác
            }
        },
        "size": size  # Trả về số lượng kết quả tùy ý
    }

    response = es.search(index=index_name, body=search_query)

    # Lấy danh sách các ID từ kết quả tìm kiếm
    ids = [hit['_id'] for hit in response['hits']['hits']]
    
    return ids
