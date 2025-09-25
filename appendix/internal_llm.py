import uuid
from langchain_openai import ChatOpenAI
import os

# 환경 변수에서 설정을 읽어 유연하게 구성합니다.
# EXTERNAL_LLM_API_KEY / EXTERNAL_LLM_API_URL / EXTERNAL_LLM_TICKET 등을 사용할 수 있습니다.
api_key = os.getenv("EXTERNAL_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

api_base_url = os.getenv("EXTERNAL_LLM_API_URL", "https://api.openai.com/v1")
credential_key = os.getenv("EXTERNAL_LLM_TICKET", "")

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

if __name__ == "__main__":
    print(llm.invoke("Hello, how are you?"))
