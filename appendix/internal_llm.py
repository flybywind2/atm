import uuid
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
api_base_url = "https://api.openai.com/v1"
credential_key = "your_credential_key"

llm = ChatOpenAI(
    base_url=api_base_url,
    model="openai/gpt-oss:120b",
    default_headers={
        "x-dep-ticket": credential_key,
        "Send-System-Name": "System_Name",
        "User-ID": "ID",
        "User-Type": "AD",
        "Prompt-Msg-Id": str(uuid.uuid4()),
        "Completion-Msg-Id": str(uuid.uuid4()),
    },
)

print(llm.invoke("Hello, how are you?"))