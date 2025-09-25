import requests
import json

url = "http://localhost:8000/retrieve-rrf"
credential_key = "your_credential_key"
rag_api_key = "your_rag_api_key"

headers = {
    "Content-Type": "application/json",
    "x-dep-ticket": credential_key,
    "api-key": rag_api_key
}

fields = {
    "index_name": "your_index_name",
    "permission_group": ["ds"],
    "query_text": "Sample query",
    "num_results_doc": 5,
    "fields_exclude": ["v_merge_title_content"],
    "filter": {
        "example_field_name": ["png"]
    }
}

json_data = json.dumps(fields)
response = requests.request("POST",url,headers=headers,data=json_data)

print(response)
print(response.text)
