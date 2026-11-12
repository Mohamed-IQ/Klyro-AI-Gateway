from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import time
import asyncio
import hashlib
import logging
import httpx
import uvicorn
import litellm
from litellm import acompletion, cost_per_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("klyro")

REQUEST_TIMEOUT = 60  # ثواني
DEFAULT_SYSTEM_PROMPT = "أجب باختصار شديد وبدون مقدمات."
HARD_SYSTEM_PROMPT = "أجب بشكل كامل ودقيق. إذا طُلب كود فاكتبه كاملاً داخل كتلة كود، ولا تسأل أسئلة توضيحية إلا عند الضرورة القصوى."

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
    history: list = None   # اختياري: [{"role": "user"|"assistant", "content": "..."}]

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

# --- 2. الاكتشاف الحقيقي للموديلات حسب مفتاح المستخدم ---
_EXCLUDE = ("embedding", "whisper", "tts", "audio", "realtime", "image",
            "moderation", "guard", "search", "transcribe", "-pro", "preview",
            "deprecated", "instruct-", "davinci", "babbage", "orpheus", "playai")

MODELS_TTL = 3600          # ساعة
PRICES_TTL = 86400         # يوم
_models_cache = {}         # sha256(key) -> (timestamp, strong, economy)
PRICE_OVERRIDE = {}        # model -> (input_per_token, output_per_token) من API المزود مباشرة
_last_price_refresh = 0.0

class InvalidKey(Exception):
    pass

async def refresh_prices_if_needed():
    """تحديث جدول أسعار litellm من المصدر مرة باليوم."""
    global _last_price_refresh
    if time.time() - _last_price_refresh < PRICES_TTL:
        return
    _last_price_refresh = time.time()
    try:
        getter = getattr(litellm, "get_model_cost_map", None)
        if getter:
            fresh = await asyncio.to_thread(getter, url=litellm.model_cost_map_url)
            if fresh:
                litellm.model_cost.update(fresh)
                logger.info("price table refreshed (%d models)", len(fresh))
    except Exception as e:
        logger.warning("price refresh failed: %s", type(e).__name__)

async def fetch_accessible_models(provider: str, api_key: str):
    """يجلب الموديلات المتاحة فعلياً لمفتاح المستخدم من API المزود."""
    urls = {
        "groq": "https://api.groq.com/openai/v1/models",
        "openai": "https://api.openai.com/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "anthropic": "https://api.anthropic.com/v1/models?limit=1000",
    }
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"} if provider == "anthropic" \
        else {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(urls[provider], headers=headers)
    if r.status_code in (401, 403):
        raise InvalidKey()
    r.raise_for_status()
    return r.json().get("data", [])

def _usable(name: str) -> bool:
    lname = name.lower()
    return not any(x in lname for x in _EXCLUDE)

def build_candidates(provider: str, items: list):
    """يرجع قائمة [(name, total_price)] لموديلات الدردشة المتاحة والمسعّرة فعلياً."""
    out = []
    for it in items:
        mid = it.get("id") if isinstance(it, dict) else None
        if not mid or not _usable(mid):
            continue

        if provider == "openrouter":
            pr = it.get("pricing") or {}
            try:
                p_in, p_out = float(pr.get("prompt", 0)), float(pr.get("completion", 0))
            except (TypeError, ValueError):
                continue
            if p_in + p_out <= 0:
                continue
            name = f"openrouter/{mid}"
            PRICE_OVERRIDE[name] = (p_in, p_out)   # السعر الحقيقي من المزود
            out.append((name, p_in + p_out))
            continue

        name = f"groq/{mid}" if provider == "groq" else mid
        info = litellm.model_cost.get(name)
        if not info or info.get("mode") != "chat" or info.get("deprecation_date"):
            continue
        total = (info.get("input_cost_per_token") or 0) + (info.get("output_cost_per_token") or 0)
        if total > 0:
            out.append((name, total))
    return out

async def get_dynamic_models(provider: str, api_key: str):
    """(strong, economy) حسب الموديلات المتاحة فعلاً لمفتاح المستخدم وأسعارها الحقيقية."""
    await refresh_prices_if_needed()
    key_id = hashlib.sha256(api_key.encode()).hexdigest()
    cached = _models_cache.get(key_id)
    if cached and time.time() - cached[0] < MODELS_TTL:
        return cached[1], cached[2]

    candidates = []
    try:
        items = await fetch_accessible_models(provider, api_key)
        candidates = build_candidates(provider, items)
    except InvalidKey:
        raise
    except Exception as e:
        logger.warning("model listing failed (%s): %s", provider, type(e).__name__)

    if not candidates:
        # بديل: جدول litellm الثابت لهذا المزود
        for name, info in litellm.model_cost.items():
            if info.get("litellm_provider") != provider or info.get("mode") != "chat":
                continue
            if not _usable(name) or info.get("deprecation_date"):
                continue
            total = (info.get("input_cost_per_token") or 0) + (info.get("output_cost_per_token") or 0)
            if total > 0:
                candidates.append((name, total))

    if not candidates:
        if provider == "groq":
            return "groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"
        raise RuntimeError("no_models")

    candidates.sort(key=lambda x: x[1])
    economy_model, strong_model = candidates[0][0], candidates[-1][0]
    _models_cache[key_id] = (time.time(), strong_model, economy_model)
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

# --- 4. دالة حساب التكلفة الحية (None = السعر غير معروف) ---
def get_live_cost(model_name: str, prompt_tokens: int, completion_tokens: int):
    if model_name in PRICE_OVERRIDE:
        p_in, p_out = PRICE_OVERRIDE[model_name]
        return p_in * prompt_tokens + p_out * completion_tokens
    try:
        p_cost, c_cost = cost_per_token(model=model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if p_cost + c_cost > 0:
            return p_cost + c_cost
    except Exception:
        pass
    return None

def _content_to_text(content) -> str:
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text")
    return str(content) if content is not None else ""

# --- 5. دالة المعالجة المشتركة ---
async def process_routing_logic(prompt: str, api_key: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                                history: list = None, extra: dict = None):
    provider = detect_provider(api_key)
    try:
        strong_model, economy_model = await get_dynamic_models(provider, api_key)
    except InvalidKey:
        return {"status": "error", "error": "مفتاح API غير صالح أو منتهي."}
    except RuntimeError:
        return {"status": "error", "error": "لا توجد موديلات متاحة لهذا المفتاح."}

    router_cost = 0.0
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
            u = route_decision.usage
            rc = get_live_cost(economy_model, u.prompt_tokens, u.completion_tokens)
            router_cost = rc or 0.0
            route_type = (route_decision.choices[0].message.content or "").strip().upper()
            if "HARD" not in route_type: route_type = "EASY"
            else: route_type = "HARD"
        except Exception as e:
            logger.warning("router classification failed: %s", type(e).__name__)
            route_type = "EASY"

        model_chosen = strong_model if route_type == "HARD" else economy_model

    # الاختصار الشديد يُطبَّق على الأسئلة السهلة فقط
    if route_type == "HARD" and system_prompt == DEFAULT_SYSTEM_PROMPT:
        system_prompt = HARD_SYSTEM_PROMPT

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

    # مقارنة حقيقية: نفس التوكنز لو استُخدم أقوى موديل متاح لمفتاح المستخدم
    cost_strong = get_live_cost(strong_model, tokens_in, tokens_out)
    cost_actual = get_live_cost(model_chosen, tokens_in, tokens_out)
    pricing_known = cost_strong is not None and cost_actual is not None

    if pricing_known:
        new_total = cost_actual + router_cost          # التكلفة الفعلية + تكلفة المصنِّف
        routing_saved = cost_strong - new_total        # قيمة حقيقية (قد تكون سالبة)
    else:
        new_total = (cost_actual or 0.0) + router_cost
        routing_saved = 0.0                            # لا نحسب توفيراً بسعر مجهول
    cost_strong = cost_strong or 0.0

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
        "new_cost": new_total,
        "router_cost": router_cost,
        "routing_saved": routing_saved,
        "baseline_model": strong_model,
        "pricing_known": pricing_known,
        # الاسم القديم محفوظ للتوافق مع الواجهة: صار يمثل تكلفة أقوى موديل لمزود المستخدم
        "cost_gpt4o": cost_strong,
        "total_saved": routing_saved
    }

# --- 6. Endpoints ---
@app.post("/detect_models")
async def detect_models(request: ConnectRequest):
    if not request.api_key or len(request.api_key.strip()) < 10:
        return {"error": "مفتاح API غير صالح", "status": "error"}

    api_key = request.api_key.strip()
    provider = detect_provider(api_key)
    try:
        # التحقق يتم بقراءة قائمة الموديلات (بدون استهلاك رصيد)
        strong_model, economy_model = await get_dynamic_models(provider, api_key)
    except InvalidKey:
        return {"error": "مفتاح API غير صالح أو منتهي.", "status": "error"}
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
    history = []
    for m in (request.history or [])[-10:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            history.append({"role": m["role"], "content": _content_to_text(m.get("content", ""))})
    history.append({"role": "user", "content": request.prompt})
    return await process_routing_logic(request.prompt, request.api_key.strip(), history=history)

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
