from fastapi import FastAPI, Request, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import time
import uvicorn
import litellm
from litellm import completion, cost_per_token, completion_cost, get_model_info

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

# --- 1. اكتشاف المزود (بدون تخمين) ---
def detect_provider(api_key: str):
    api_key = api_key.strip()
    if api_key.startswith("gsk_"): return "groq"
    elif api_key.startswith("sk-ant-"): return "anthropic"
    elif api_key.startswith("sk-or-"): return "openrouter"
    elif api_key.startswith("sk-"): return "openai"
    return "groq" # الافتراضي

# --- 2. السحر الجديد: الاكتشاف الديناميكي للموديلات ---
def get_dynamic_models(provider: str):
    """جلب كل الموديلات من LiteLLM وتصنيفها تلقائياً بناءً على السعر"""
    provider_models = []
    
    # نمر على كل الموديلات الموجودة بالمكتبة
    for model_name, info in litellm.model_cost.items():
        # نتأكد إن الموديل تابع لنفس المزود، ونوعه "محادثة" (مو توليد صور أو صوت)
        if info.get("litellm_provider") == provider and info.get("mode") == "chat":
            in_cost = info.get("input_cost_per_token", 0)
            out_cost = info.get("output_cost_per_token", 0)
            total_cost = in_cost + out_cost
            
            # نستبعد الموديلات اللي سعرها صفر أو غير محدد
            if total_cost > 0:
                provider_models.append({
                    "name": model_name,
                    "cost": total_cost
                })
    
    # إذا ما لقينا موديلات (مثلاً المزود جديد)، نرجع قيم افتراضية آمنة
    if not provider_models:
        return "groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"
    
    # نرتب الموديلات حسب السعر (من الأرخص للأغلى)
    provider_models.sort(key=lambda x: x["cost"])
    
    # الاقتصادي هو الأرخص، القوي هو الأغلى
    economy_model = provider_models[0]["name"]
    strong_model = provider_models[-1]["name"]
    
    return strong_model, economy_model

# --- 3. الموزع الذكي (Heuristics) ---
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

# --- 4. دالة حساب التكلفة الحية ---
def get_live_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        p_cost, c_cost = cost_per_token(model=model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if p_cost + c_cost > 0: return p_cost + c_cost
    except Exception: pass
    return 0.0

# --- 5. دالة المعالجة المشتركة ---
async def process_routing_logic(prompt: str, api_key: str):
    provider = detect_provider(api_key)
    strong_model, economy_model = get_dynamic_models(provider)
    
    heuristic = smart_heuristic_router(prompt)
    
    if heuristic == "EASY":
        model_chosen, route_type = economy_model, "EASY"
    elif heuristic == "HARD":
        model_chosen, route_type = strong_model, "HARD"
    else:
        # الموزع الذكي (نستخدم الاقتصادي كموزع لتوفير المال)
        try:
            prompt_router = "صنف السؤال إلى EASY أو HARD. (EASY للأسئلة العادية، HARD للأكواد والتحليل). اكتب كلمة واحدة فقط."
            route_decision = completion(model=economy_model, messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": prompt}], temperature=0, api_key=api_key, drop_params=True)
            route_type = route_decision.choices[0].message.content.strip().upper()
            if "HARD" not in route_type: route_type = "EASY"
        except Exception: route_type = "EASY"
        
        model_chosen = strong_model if route_type == "HARD" else economy_model

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

    cost_strong = get_live_cost(strong_model, tokens_in, tokens_out)
    cost_actual = get_live_cost(model_chosen, tokens_in, tokens_out)
    routing_saved = max(0, cost_strong - cost_actual) if "economy" in model_chosen or model_chosen == economy_model else 0.0
    cost_gpt4o = get_live_cost("gpt-4o", tokens_in, tokens_out)
    
    return {
        "status": "success",
        "provider": provider,
        "route": route_type,
        "model_used": model_chosen,
        "strong_model_selected": strong_model,
        "economy_model_selected": economy_model,
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_strong_provider": cost_strong,
        "new_cost": cost_actual,
        "routing_saved": routing_saved,
        "cost_gpt4o": cost_gpt4o,
        "total_saved": max(0, cost_gpt4o - cost_actual)
    }

# --- 6. Endpoints ---
@app.post("/detect_models")
async def detect_models(request: ConnectRequest):
    if not request.api_key or len(request.api_key) < 5:
        return {"error": "مفتاح API غير صالح", "status": "error"}
    
    provider = detect_provider(request.api_key)
    strong_model, economy_model = get_dynamic_models(provider)
    
    # فحص سريع جداً للتأكد من أن المفتاح يعمل
    try:
        completion(model=economy_model, messages=[{"role": "user", "content": "hi"}], api_key=request.api_key, max_tokens=1, drop_params=True)
    except Exception as e:
        return {"error": f"فشل الاتصال. ({str(e)})", "status": "error"}
        
    return {
        "status": "success", 
        "provider": provider, 
        "strong_model": strong_model, 
        "economy_model": economy_model
    }

@app.post("/route")
async def route_chat(request: ChatRequest):
    return await process_routing_logic(request.prompt, request.api_key)

@app.post("/v1/chat/completions")
async def enterprise_chat_completions(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": {"message": "Missing header", "type": "invalid_request_error"}})
    
    api_key = auth_header.replace("Bearer ", "").strip()
    try:
        body = await request.json()
        messages = body.get("messages", [])
        prompt = messages[-1].get("content", "")
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})

    result = await process_routing_logic(prompt, api_key)

    if result["status"] == "error":
        return JSONResponse(status_code=500, content={"error": {"message": result["error"], "type": "klyro_error"}})

    return {
        "id": f"chatcmpl-klyro-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o"), 
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result["answer"]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": result["tokens_in"], "completion_tokens": result["tokens_out"], "total_tokens": result["tokens_in"] + result["tokens_out"]},
        "klyro_metadata": {
            "routing_saved": result["routing_saved"],
            "cost_actual": result["new_cost"],
            "smart_route": result["route"]
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
