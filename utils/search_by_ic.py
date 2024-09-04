from elasticsearch import Elasticsearch

def search_ic(query, index_name, size):
    # Kết nối đến Elasticsearch
    es = Elasticsearch(["http://localhost:9200"])

    # Tìm kiếm dữ liệu
    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["ic"],
                "fuzziness": "AUTO"  # Tùy chọn để cho phép khớp không chính xác
            }
        },
        "size": size  # Trả về số lượng kết quả tùy ý
    }

    response = es.search(index=index_name, body=search_query)

    hits = response['hits']['hits']

    # Lấy các ID khớp
    matching_ids = [hit['_id'] for hit in hits]
    num_matching = len(matching_ids)
    if num_matching < 100:
        # Truy vấn để lấy các tài liệu không khớp hoặc có điểm số thấp
        additional_query = {
            "query": {
                "bool": {
                    "must_not": [
                        {
                            "ids": {
                                "values": matching_ids
                            }
                        }
                    ]
                }
            },
            "size": 100 - num_matching
        }
        
        additional_response = es.search(index=index_name, body=additional_query)
        additional_hits = additional_response['hits']['hits']
        
        # Lấy các ID của tài liệu bổ sung
        additional_ids = [hit['_id'] for hit in additional_hits]
        matching_ids = matching_ids + additional_ids
    else:
        matching_ids = matching_ids
        
    return matching_ids
