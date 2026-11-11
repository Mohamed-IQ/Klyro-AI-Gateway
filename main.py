from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import uvicorn
from litellm import completion, cost_per_token, completion_cost

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

class ConnectRequest(BaseModel):
    api_key: str

PROVIDER_MODEL_MAP = {
    "groq": {
        "router": "groq/llama3-8b-8192",
        "strong": "groq/openai/gpt-oss-120b",
        "economy": "groq/openai/gpt-oss-20b",
    },
    "openai": {
        "router": "gpt-4o-mini",
        "strong": "gpt-4o",
        "economy": "gpt-4o-mini",
    },
    "anthropic": {
        "router": "anthropic/claude-3-haiku-20240307",
        "strong": "anthropic/claude-3-5-sonnet-20240620",
        "economy": "anthropic/claude-3-haiku-20240307",
    },
}

def detect_provider_and_models(api_key: str):
    """اكتشاف المزود والنماذج بناءً على بادئة المفتاح (الطريقة الأضمن)"""
    api_key = api_key.strip()
    if api_key.startswith("gsk_"):
        provider = "groq"
    elif api_key.startswith("sk-ant-"):
        provider = "anthropic"
    elif api_key.startswith("sk-or-"):
        provider = "openrouter"
    elif api_key.startswith("sk-"):
        provider = "openai"
    else:
        # إذا كانت البادئة غير معروفة، نستخدم Groq كافتراضي
        provider = "groq"

    models = PROVIDER_MODEL_MAP.get(provider, PROVIDER_MODEL_MAP["groq"])
    return provider, models

def smart_heuristic_router(prompt: str) -> str:
    text = prompt.strip()
    lower_text = text.lower()
    
    if len(text) < 20:
        simple_greetings = ["مرحبا", "اهلا", "السلام", "هاي", "hello", "hi", "hey", "شكرا"]
        if any(g in lower_text for g in simple_greetings) or len(text.split()) <= 4:
            return "EASY"
    
    if "```" in text or re.search(r"(def |class |import |function |const |let |var |public |private )", text):
        return "HARD"
    if re.search(r"[=+\-*/^%]", text) and len(text) > 20:
        return "HARD"
        
    hard_keywords = ["كود", "code", "برمجة", "دالة", "function", "خطأ", "error", "تحليل بيانات", "data analysis", "خطة عمل", "business plan"]
    if any(k in lower_text for k in hard_keywords):
        return "HARD"
        
    return "UNKNOWN" 

def get_live_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        prompt_cost, completion_cost_val = cost_per_token(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        total = prompt_cost + completion_cost_val
        if total > 0: return total
    except Exception:
        pass

    try:
        class MockResponse:
            def __init__(self, model, prompt_tokens, completion_tokens):
                self.model = model
                self.usage = type('obj', (object,), {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens
                })
        mock_resp = MockResponse(model_name, prompt_tokens, completion_tokens)
        cost = completion_cost(completion_response=mock_resp)
        if cost > 0: return cost
    except Exception:
        pass

    return 0.0

@app.post("/detect_models")
async def detect_models(request: ConnectRequest):
    if not request.api_key or len(request.api_key) < 5:
        return {"error": "مفتاح API غير صالح", "status": "error"}
    
    provider, models = detect_provider_and_models(request.api_key)
    
    try:
        completion(
            model=models["economy"],
            messages=[{"role": "user", "content": "hi"}],
            api_key=request.api_key,
            max_tokens=1,
            drop_params=True
        )
    except Exception as e:
        return {"error": f"فشل الاتصال بالمزود. تأكد من صحة المفتاح. ({str(e)})", "status": "error"}

    return {
        "status": "success",
        "provider": provider,
        "strong_model": models["strong"],
        "economy_model": models["economy"]
    }

@app.post("/route")
async def route_chat(request: ChatRequest):
    provider, models = detect_provider_and_models(request.api_key)
    heuristic_decision = smart_heuristic_router(request.prompt)
    
    if heuristic_decision == "EASY":
        model_chosen = models["economy"]
        model_name = f"ECONOMY (Heuristic) - {provider}"
    elif heuristic_decision == "HARD":
        model_chosen = models["strong"]
        model_name = f"STRONG (Heuristic) - {provider}"
    else:
        try:
            prompt_router = """
أنت مدير مالي وتقني (CTO) صارم جداً في تقليل التكاليف.
مهمتك: تصنيف سؤال المستخدم إلى EASY أو HARD. اكتب كلمة واحدة فقط.
- EASY: التحيات، الأسئلة العامة، الترجمات، التعريفات، الأسئلة الحسابية البسيطة، كتابة رسائل قصيرة، المحادثات العادية.
- HARD: كتابة أكواد برمجية، تصحيح أخطاء برمجية، تحليل بيانات معقدة، حل مسائل رياضية معقدة، أو إنشاء خطط عمل واستراتيجيات كبيرة.
إذا كنت متردداً، صنفه EASY.
اكتب EASY أو HARD فقط.
"""
            route_decision = completion(
                model=models["router"],
                messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": request.prompt}],
                temperature=0,
                api_key=request.api_key,
                drop_params=True
            )
            route_type = route_decision.choices[0].message.content.strip().upper()
        except Exception:
            route_type = "EASY"

        if "HARD" in route_type:
            model_chosen = models["strong"]
            model_name = f"STRONG (LLM Router) - {provider}"
        else:
            model_chosen = models["economy"]
            model_name = f"ECONOMY (LLM Router) - {provider}"

    try:
        response = completion(
            model=model_chosen,
            messages=[{"role": "system", "content": "أجب باختصار شديد وبدون مقدمات."}, {"role": "user", "content": request.prompt}],
            api_key=request.api_key,
            drop_params=True
        )
        answer = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
    except Exception as e:
        return {"error": f"فشل توليد الإجابة: {str(e)}", "status": "error"}

    cost_strong = get_live_cost(models["strong"], tokens_in, tokens_out)
    cost_actual = get_live_cost(model_chosen, tokens_in, tokens_out)
    routing_saved = max(0, cost_strong - cost_actual) if "ECONOMY" in model_name else 0.0
    cost_gpt4o = get_live_cost("gpt-4o", tokens_in, tokens_out)
    total_saved = max(0, cost_gpt4o - cost_actual)

    return {
        "status": "success",
        "provider": provider,
        "route": "HARD" if "STRONG" in model_name else "EASY",
        "model_used": model_name,
        "strong_model_selected": models["strong"],
        "economy_model_selected": models["economy"],
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_strong_provider": cost_strong,
        "new_cost": cost_actual,
        "routing_saved": routing_saved,
        "cost_gpt4o": cost_gpt4o,
        "total_saved": total_saved
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
