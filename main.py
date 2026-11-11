from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from litellm import completion, completion_cost

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    try:
        prompt_router = "أنت موزع. اكتب EASY إذا كان السؤال بسيطاً، و HARD إذا كان معقداً. اكتب كلمة واحدة فقط."
        route_decision = completion(
            model="groq/openai/gpt-oss-20b",
            messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": request.prompt}],
            temperature=0,
            api_key=request.api_key,
            drop_params=True
        )
        route_type = route_decision.choices[0].message.content.strip().upper()
    except Exception as e:
        return {"error": f"فشل الاتصال: {str(e)}"}

    if "HARD" in route_type:
        model_chosen = "groq/openai/gpt-oss-120b"
        model_name = "HARD (Heavy Model)"
    else:
        model_chosen = "groq/openai/gpt-oss-20b"
        model_name = "EASY (Economic Model)"

    try:
        response = completion(
            model=model_chosen,
            messages=[{"role": "system", "content": "أجب باختصار شديد وبدون مقدمات."}, {"role": "user", "content": request.prompt}],
            api_key=request.api_key,
            drop_params=True
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return {"error": f"فشل توليد الإجابة: {str(e)}"}

    # 3. الحسابات الحقيقية (باستخدام completion_cost)
    try:
        real_cost_new = completion_cost(completion_response=response)
    except Exception:
        # خطة احتياطية: أسعار Groq الفعلية التقريبية
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        real_cost_new = (tokens_in * 0.075 / 1_000_000) + (tokens_out * 0.30 / 1_000_000)

    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    old_cost = (tokens_in * request.PRICE_OLD_IN) + (tokens_out * request.PRICE_OLD_OUT)
    
    saved = max(0, old_cost - real_cost_new)

    return {
        "route": route_type,
        "model_used": model_name,
        "answer": answer,
        "old_cost": old_cost,
        "new_cost": real_cost_new,
        "saved": saved
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
