from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
# Connect to Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Define the index name
index_name = "object"



# Search function
def search_od(query):
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
        "_source": False  # Don't return the source document
    }

    results = es.search(index=index_name, body=body)
    
    # Return the document IDs (which are the original keys)
    return list(map(int,[hit['_id'] for hit in results['hits']['hits']]))

