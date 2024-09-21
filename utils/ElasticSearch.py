from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def search_by_ocr(query, index_name, size):
    es = Elasticsearch(["http://localhost:9200"])

    exact_query = {
        "query": {
            "match": {
                "ocr": {
                    "query": query,
                    "operator": "and"
                }
            }
        },
        "size": size
    }

    exact_response = es.search(index=index_name, body=exact_query)
    exact_hits = exact_response['hits']['hits']
    exact_results = [{'id': hit['_id'], 'score': hit['_score']} for hit in exact_hits]

    fuzzy_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["ocr"],
                "fuzziness": "AUTO"
            }
        },
        "size": size
    }

    fuzzy_response = es.search(index=index_name, body=fuzzy_query)
    fuzzy_hits = fuzzy_response['hits']['hits']
    fuzzy_results = [{'id': hit['_id'], 'score': hit['_score']} for hit in fuzzy_hits]

    exact_ids = {result['id'] for result in exact_results}
    unique_fuzzy_results = [result for result in fuzzy_results if result['id'] not in exact_ids]

    all_results = exact_results + unique_fuzzy_results
    matching_ids = [result['id'] for result in all_results]

    return matching_ids


def search_by_od(query, index_name):
    must_clauses = []
    for q in query:
        quantity, name, attribute = q
        clause = {
            "nested": {
                "path": "objects",
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"objects.name": name}} if name != "None" else {"match_all": {}},
                            {"match": {"objects.attribute": attribute}} if attribute != "None" else {"match_all": {}}
                        ]
                    }
                }
            }
        }
        if quantity != "None":
            clause["nested"]["query"]["bool"]["must"].append(
                {"range": {"objects.quantity": {"gte": int(quantity)}}}
            )
        must_clauses.append(clause)

    body = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        },
        "_source": False
    }

    results = es.search(index=index_name, body=body, size=200)
    return list(map(int, [hit['_id'] for hit in results['hits']['hits']]))


def search_by_ic(query, index_name, size):
    es = Elasticsearch(["http://localhost:9200"])

    search_query = {
        "query": {
            "more_like_this": {
                "fields": ["ic"],
                "like": [
                    {
                        "doc": {
                            "ic": query
                        }
                    }
                ],
                "min_term_freq": 1,
                "max_query_terms": 12
            }
        },
        "size": size
    }

    response = es.search(index=index_name, body=search_query)
    hits = response['hits']['hits']
    matching_ids = [hit['_id'] for hit in hits]
    num_matching = len(matching_ids)
    
    if num_matching < size:
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
        additional_ids = [hit['_id'] for hit in additional_hits]
        matching_ids.extend(additional_ids)
    
    return matching_ids

def search_by_asr(query, index_name, size):
    es = Elasticsearch(["http://localhost:9200"])

    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["asr"],
                "fuzziness": "AUTO"
            }
        },
        "size": size
    }

    response = es.search(index=index_name, body=search_query)
    hits = response['hits']['hits']
    matching_ids = [hit['_id'] for hit in hits]
    num_matching = len(matching_ids)

    if num_matching < 100:
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
        additional_ids = [hit['_id'] for hit in additional_hits]
        matching_ids = matching_ids + additional_ids
    else:
        matching_ids = matching_ids
        
    return matching_ids