from fastapi import FastAPI, Request, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import time
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
    api_key = api_key.strip()
    if api_key.startswith("gsk_"): provider = "groq"
    elif api_key.startswith("sk-ant-"): provider = "anthropic"
    elif api_key.startswith("sk-or-"): provider = "openrouter"
    elif api_key.startswith("sk-"): provider = "openai"
    else: provider = "groq"
    return provider, PROVIDER_MODEL_MAP.get(provider, PROVIDER_MODEL_MAP["groq"])

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
        p_cost, c_cost = cost_per_token(model=model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if p_cost + c_cost > 0: return p_cost + c_cost
    except Exception: pass
    try:
        class MockResponse:
            def __init__(self, model, prompt_tokens, completion_tokens):
                self.model = model
                self.usage = type('obj', (object,), {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})
        cost = completion_cost(completion_response=MockResponse(model_name, prompt_tokens, completion_tokens))
        if cost > 0: return cost
    except Exception: pass
    return 0.0

# --- دالة داخلية مشتركة لمعالجة الطلبات (تستخدمها الواجهة والشركات) ---
async def process_routing_logic(prompt: str, api_key: str):
    provider, models = detect_provider_and_models(api_key)
    heuristic = smart_heuristic_router(prompt)
    
    if heuristic == "EASY":
        model_chosen, route_type = models["economy"], "EASY"
    elif heuristic == "HARD":
        model_chosen, route_type = models["strong"], "HARD"
    else:
        try:
            prompt_router = "أنت مدير مالي صارم. صنف السؤال إلى EASY أو HARD. (EASY للأسئلة العادية، HARD للأكواد والتحليل). اكتب كلمة واحدة فقط."
            route_decision = completion(model=models["router"], messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": prompt}], temperature=0, api_key=api_key, drop_params=True)
            route_type = route_decision.choices[0].message.content.strip().upper()
            if "HARD" not in route_type: route_type = "EASY"
        except Exception: route_type = "EASY"
        
        model_chosen = models["strong"] if route_type == "HARD" else models["economy"]

    try:
        response = completion(
            model=model_chosen,
            messages=[{"role": "system", "content": "أجب باختصار شديد وبدون مقدمات."}, {"role": "user", "content": prompt}],
            api_key=api_key, drop_params=True
        )
        answer = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
    except Exception as e:
        return {"status": "error", "error": str(e)}

    cost_strong = get_live_cost(models["strong"], tokens_in, tokens_out)
    cost_actual = get_live_cost(model_chosen, tokens_in, tokens_out)
    routing_saved = max(0, cost_strong - cost_actual) if "economy" in model_chosen else 0.0
    cost_gpt4o = get_live_cost("gpt-4o", tokens_in, tokens_out)
    
    return {
        "status": "success",
        "provider": provider,
        "route": route_type,
        "model_used": model_chosen,
        "strong_model_selected": models["strong"],
        "economy_model_selected": models["economy"],
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_strong_provider": cost_strong,
        "new_cost": cost_actual,
        "routing_saved": routing_saved,
        "cost_gpt4o": cost_gpt4o,
        "total_saved": max(0, cost_gpt4o - cost_actual)
    }

# --- 1. Endpoint الواجهة التجريبية ---
@app.post("/detect_models")
async def detect_models(request: ConnectRequest):
    if not request.api_key or len(request.api_key) < 5:
        return {"error": "مفتاح API غير صالح", "status": "error"}
    provider, models = detect_provider_and_models(request.api_key)
    try:
        completion(model=models["economy"], messages=[{"role": "user", "content": "hi"}], api_key=request.api_key, max_tokens=1, drop_params=True)
    except Exception as e:
        return {"error": f"فشل الاتصال. ({str(e)})", "status": "error"}
    return {"status": "success", "provider": provider, "strong_model": models["strong"], "economy_model": models["economy"]}

@app.post("/route")
async def route_chat(request: ChatRequest):
    return await process_routing_logic(request.prompt, request.api_key)

# --- 2. Endpoint الشركات (متوافق 100% مع OpenAI SDK) ---
@app.post("/v1/chat/completions")
async def enterprise_chat_completions(request: Request):
    # 1. سحب المفتاح من الهيدر (Authorization: Bearer sk-...)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": {"message": "Missing or invalid Authorization header", "type": "invalid_request_error"}})
    
    api_key = auth_header.replace("Bearer ", "").strip()

    # 2. سحب البيانات من الطلب
    try:
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            raise ValueError("No messages found")
        # استخراج آخر سؤال من المستخدم
        prompt = messages[-1].get("content", "")
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})

    # 3. تطبيق التوجيه الذكي
    result = await process_routing_logic(prompt, api_key)

    if result["status"] == "error":
        return JSONResponse(status_code=500, content={"error": {"message": result["error"], "type": "klyro_routing_error"}})

    # 4. تحويل الرد لصيغة OpenAI القياسية (حتى لا يشعر نظام الشركة بأي فرق)
    return {
        "id": f"chatcmpl-klyro-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        # نرجع اسم الموديل اللي طلبته الشركة عشان ما ينكسر نظامهم (حتى لو استخدمنا Groq بالخلفية)
        "model": body.get("model", "gpt-4o"), 
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["answer"]},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": result["tokens_in"],
            "completion_tokens": result["tokens_out"],
            "total_tokens": result["tokens_in"] + result["tokens_out"]
        },
        # إحنه نضيف معلومات Klyro كـ "حقل مخفي" للعميل يشوفه بالـ Dashboard
        "klyro_metadata": {
            "routing_saved": result["routing_saved"],
            "cost_actual": result["new_cost"],
            "cost_gpt4o": result["cost_gpt4o"],
            "smart_route": result["route"]
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
