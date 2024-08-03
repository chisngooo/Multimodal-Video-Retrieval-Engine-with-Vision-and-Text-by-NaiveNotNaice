from elasticsearch import Elasticsearch


def search_od(query, index_name, size):
    # Tạo điều kiện truy vấn cho từng từ khóa trong mảng
    es = Elasticsearch(["http://localhost:9200"])
    must_conditions = [{"match": {"objects": keyword}} for keyword in query]

    # Tìm kiếm dữ liệu với nhiều điều kiện match
    search_query = {
        "query": {
            "bool": {
                "must": must_conditions
            }
        },
        "size": size  # Trả về số lượng kết quả tùy ý
    }

    response = es.search(index=index_name, body=search_query)
    hits = response['hits']['hits']

    # Lấy các ID khớp
    matching_ids = [hit['_id'] for hit in hits]
    num_matching = len(matching_ids)

    if num_matching < size:
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
            "size": size - num_matching
        }
        
        additional_response = es.search(index=index_name, body=additional_query)
        additional_hits = additional_response['hits']['hits']
        
        # Lấy các ID của tài liệu bổ sung
        additional_ids = [hit['_id'] for hit in additional_hits]
        matching_ids.extend(additional_ids)  # Thêm các ID bổ sung vào danh sách kết quả
    else:
        # Nếu số lượng tài liệu khớp đủ
        matching_ids = matching_ids
    
    return matching_ids



