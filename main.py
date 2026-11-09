from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from litellm import completion, completion_cost

app = FastAPI()

# السماح للواجهة بالاتصال بالسيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    api_key: str
    PRICE_OLD_IN: float = 2.50 / 1_000_000
    PRICE_OLD_OUT: float = 10.00 / 1_000_000

@app.post("/route")
async def route_chat(request: ChatRequest):
    os.environ["GROQ_API_KEY"] = request.api_key
    
    try:
        # 1. الموزع الذكي
        prompt_router = "أنت موزع. اكتب EASY إذا كان السؤال بسيطاً، و HARD إذا كان معقداً (كود، تحليل، خطة). اكتب كلمة واحدة فقط."
        route_decision = completion(
            model="groq/openai/gpt-oss-20b",
            messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": request.prompt}],
            temperature=0, reasoning_effort="low"
        )
        route_type = route_decision.choices[0].message.content.strip().upper()
    except Exception as e:
        return {"error": f"فشل الاتصال: {str(e)}"}

    # 2. التوجيه
    if "HARD" in route_type:
        model_chosen = "groq/openai/gpt-oss-120b"
        model_name = "HARD (Heavy Model)"
    else:
        model_chosen = "groq/openai/gpt-oss-20b"
        model_name = "EASY (Economic Model)"

    try:
        concise_prompt = {"role": "system", "content": "أجب باختصار شديد وبدون مقدمات."}
        response = completion(
            model=model_chosen,
            messages=[concise_prompt, {"role": "user", "content": request.prompt}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return {"error": f"فشل توليد الإجابة: {str(e)}"}

    # 3. الحسابات الحقيقية
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    real_cost_new = completion_cost(completion_response=response)
    old_cost = (tokens_in * request.PRICE_OLD_IN) + (tokens_out * request.PRICE_OLD_OUT)
    
    saved = old_cost - real_cost_new
    if saved < 0: saved = 0 

    return {
        "route": route_type,
        "model_used": model_name,
        "answer": answer,
        "old_cost": old_cost,
        "new_cost": real_cost_new,
        "saved": saved
    }
