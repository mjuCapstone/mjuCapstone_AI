from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

assistant_id = os.getenv("ASSISTANT_ID")
api_key_env = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key_env
)

logging.basicConfig(level=logging.INFO)

@app.post("/api/v1/recommend/chat")
async def recommend(request: Dict[str, Any]):
    content = request.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Content not provided")

    logging.info(f"Received content: {content}")

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
        logging.info(f"Received message content: {message_content}")

        # message_content가 Text 객체인 경우, .value 속성을 통해 실제 텍스트를 가져옵니다
        if hasattr(message_content, 'value'):
            message_content = message_content.value

        # JSON 문자열 추출을 위한 정규 표현식
        json_match = re.search(r'```json\n(.*?)\n```', message_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

            # JSON 문자열을 파이썬 리스트로 변환
            data = json.loads(json_str)

            # 응답 데이터를 로그에 출력
            logging.info(f"Response data: {json.dumps(data, ensure_ascii=False)}")

            # 응답 반환 (JSON 배열 형태로 직접 반환)
            return data
        else:
            logging.error("JSON 형식의 데이터를 찾을 수 없습니다.")
            raise HTTPException(status_code=400, detail="JSON 형식의 데이터를 찾을 수 없습니다.")

    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)
