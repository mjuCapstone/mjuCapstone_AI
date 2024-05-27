from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re
from typing import Dict

app = FastAPI()

load_dotenv()
assistant_id = os.getenv("ASSISTANT_ID")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class ContentRequest(BaseModel):
    content: str

@app.post("/api/v1/recommend/chat")
async def recommend(request: ContentRequest):
    content = request.content
    # content += "답은 json{name : String, gram : int, kcal : int, carbohydrate : int, protein : int, fat : int}  형식으로 부탁해"
    print(content)
    try:
        # create thread
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ]
        )

        # create run by our assistant
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant_id
        )

        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        message_content = messages[0].content[0].text

        # message_content가 Text 객체인 경우, .value 속성을 통해 실제 텍스트를 가져옵니다
        if hasattr(message_content, 'value'):
            message_content = message_content.value

        # JSON 문자열 추출을 위한 정규 표현식
        json_match = re.search(r'```json\n(.*?)\n```', message_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

            # JSON 문자열을 파이썬 딕셔너리로 변환
            data = json.loads(json_str)

            # 파이썬 딕셔너리를 그대로 반환
            return data
        else:
            raise HTTPException(status_code=400, detail="JSON 형식의 데이터를 찾을 수 없습니다.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)
