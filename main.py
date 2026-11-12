from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache
import os
import re
import time
import logging
import uvicorn
import litellm
from litellm import acompletion, cost_per_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("klyro")

REQUEST_TIMEOUT = 60  # ثواني
DEFAULT_SYSTEM_PROMPT = "أجب باختصار شديد وبدون مقدمات."

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

# --- 1. اكتشاف المزود ---
def detect_provider(api_key: str):
    api_key = api_key.strip()
    if api_key.startswith("gsk_"): return "groq"
    elif api_key.startswith("sk-ant-"): return "anthropic"
    elif api_key.startswith("sk-or-"): return "openrouter"
    elif api_key.startswith("sk-"): return "openai"  # يشمل sk-proj-
    return "groq" # الافتراضي

# --- 2. الاكتشاف الديناميكي للموديلات (مع كاش) ---
_EXCLUDE = ("embedding", "whisper", "tts", "audio", "realtime", "image",
            "moderation", "guard", "search", "transcribe", "-pro", "preview",
            "deprecated", "vision-preview", "instruct-", "davinci", "babbage")

@lru_cache(maxsize=16)
def get_dynamic_models(provider: str):
    provider_models = []

    for model_name, info in litellm.model_cost.items():
        if info.get("litellm_provider") != provider or info.get("mode") != "chat":
            continue
        lname = model_name.lower()
        if any(x in lname for x in _EXCLUDE):
            continue
        if info.get("deprecation_date"):
            continue

        in_cost = info.get("input_cost_per_token") or 0
        out_cost = info.get("output_cost_per_token") or 0
        total_cost = in_cost + out_cost

        if total_cost > 0:
            provider_models.append({"name": model_name, "cost": total_cost})

    # قيم افتراضية آمنة
    if not provider_models:
        return "groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"

    provider_models.sort(key=lambda x: x["cost"])

    economy_model = provider_models[0]["name"]
    strong_model = provider_models[-1]["name"]

    return strong_model, economy_model

# --- 3. الموزع الذكي (Heuristics) ---
_GREETINGS = ["مرحبا", "اهلا", "أهلا", "السلام", "هاي", "hello", "hi", "hey", "شكرا", "شكراً"]
_HARD_AR = ["كود", "برمجة", "دالة", "خطأ", "تحليل بيانات", "خطة عمل"]
_HARD_EN_RE = re.compile(r"\b(code|function|error|bug|debug|data analysis|business plan|algorithm)\b", re.I)
_CODE_RE = re.compile(r"(^|\s)(def |class |import |function |const |let |var |public |private )")
_MATH_RE = re.compile(r"\d\s*[=+\-*/^%]\s*\d|[a-zA-Z]\s*=\s*[\w(]")

def smart_heuristic_router(prompt: str) -> str:
    text = prompt.strip()
    lower_text = text.lower()

    if len(text) < 20:
        words = re.findall(r"\w+", lower_text)
        if words and all(w in _GREETINGS for w in words):
            return "EASY"
        if any(w in _GREETINGS for w in words) and len(words) <= 3:
            return "EASY"

    if "```" in text or _CODE_RE.search(text):
        return "HARD"
    if _MATH_RE.search(text):
        return "HARD"
    if _HARD_EN_RE.search(text) or any(k in lower_text for k in _HARD_AR):
        return "HARD"
    return "UNKNOWN"

# --- 4. دالة حساب التكلفة الحية ---
def get_live_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        p_cost, c_cost = cost_per_token(model=model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if p_cost + c_cost > 0: return p_cost + c_cost
    except Exception: pass
    return 0.0

def _content_to_text(content) -> str:
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text")
    return str(content) if content is not None else ""

# --- 5. دالة المعالجة المشتركة ---
async def process_routing_logic(prompt: str, api_key: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                                history: list = None, extra: dict = None):
    provider = detect_provider(api_key)
    strong_model, economy_model = get_dynamic_models(provider)

    heuristic = smart_heuristic_router(prompt)

    if heuristic == "EASY":
        model_chosen, route_type = economy_model, "EASY"
    elif heuristic == "HARD":
        model_chosen, route_type = strong_model, "HARD"
    else:
        try:
            prompt_router = "صنف السؤال إلى EASY أو HARD. (EASY للأسئلة العادية، HARD للأكواد والتحليل). اكتب كلمة واحدة فقط."
            route_decision = await acompletion(
                model=economy_model,
                messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": prompt}],
                temperature=0, max_tokens=5, api_key=api_key, drop_params=True, timeout=20
            )
            route_type = (route_decision.choices[0].message.content or "").strip().upper()
            if "HARD" not in route_type: route_type = "EASY"
            else: route_type = "HARD"
        except Exception as e:
            logger.warning("router classification failed: %s", type(e).__name__)
            route_type = "EASY"

        model_chosen = strong_model if route_type == "HARD" else economy_model

    # بناء الرسائل: كامل السياق إن وُجد
    if history:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    try:
        response = await acompletion(
            model=model_chosen,
            messages=messages,
            api_key=api_key, drop_params=True, timeout=REQUEST_TIMEOUT,
            num_retries=1,
            **(extra or {})
        )
        answer = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
    except Exception as e:
        logger.error("completion failed (%s): %s", model_chosen, type(e).__name__)
        return {"status": "error", "error": f"فشل الطلب إلى المزود ({type(e).__name__}). تحقق من المفتاح أو الموديل."}

    cost_strong = get_live_cost(strong_model, tokens_in, tokens_out)
    cost_actual = get_live_cost(model_chosen, tokens_in, tokens_out)

    # التوفير فقط إذا اختير موديل مختلف عن القوي
    routing_saved = max(0, cost_strong - cost_actual) if model_chosen != strong_model else 0.0
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
    if not request.api_key or len(request.api_key.strip()) < 10:
        return {"error": "مفتاح API غير صالح", "status": "error"}

    api_key = request.api_key.strip()
    provider = detect_provider(api_key)
    strong_model, economy_model = get_dynamic_models(provider)

    try:
        await acompletion(model=economy_model, messages=[{"role": "user", "content": "hi"}],
                          api_key=api_key, max_tokens=1, drop_params=True, timeout=20)
    except Exception as e:
        logger.warning("detect_models failed: %s", type(e).__name__)
        return {"error": f"فشل الاتصال ({type(e).__name__}). تحقق من المفتاح.", "status": "error"}

    return {
        "status": "success",
        "provider": provider,
        "strong_model": strong_model,
        "economy_model": economy_model
    }

@app.post("/route")
async def route_chat(request: ChatRequest):
    return await process_routing_logic(request.prompt, request.api_key.strip())

@app.post("/v1/chat/completions")
async def enterprise_chat_completions(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": {"message": "Missing header", "type": "invalid_request_error"}})

    api_key = auth_header.replace("Bearer ", "").strip()
    try:
        body = await request.json()
        messages = body.get("messages", [])

        last_message = messages[-1] if messages else {}
        prompt = _content_to_text(last_message.get("content", ""))

        system_prompt = DEFAULT_SYSTEM_PROMPT
        history = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                if system_prompt == DEFAULT_SYSTEM_PROMPT:
                    system_prompt = _content_to_text(msg.get("content", "")) or DEFAULT_SYSTEM_PROMPT
            elif role in ("user", "assistant"):
                history.append({"role": role, "content": _content_to_text(msg.get("content", ""))})

        extra = {}
        for k in ("temperature", "max_tokens", "top_p"):
            if body.get(k) is not None:
                extra[k] = body[k]

    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})

    result = await process_routing_logic(prompt, api_key, system_prompt, history, extra)

    if result["status"] == "error":
        return JSONResponse(status_code=500, content={"error": {"message": result["error"], "type": "klyro_error"}})

    return {
        "id": f"chatcmpl-klyro-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model_used"],
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
