from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import time
import asyncio
import hashlib
import json
from collections import OrderedDict
from typing import Optional
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

# --- إعدادات تقليل التوكنز ---
CACHE_MAX = 1000               # أقصى عدد ردود محفوظة
CACHE_TTL = 3600               # صلاحية الرد المحفوظ (ثانية)
HISTORY_MAX_MESSAGES = 10      # أقصى عدد رسائل سابقة تُرسل للموديل
HISTORY_MAX_CHARS = 6000       # أقصى حجم للسياق المرسل (حروف)
_response_cache = OrderedDict()  # key -> (timestamp, answer, tokens_in, tokens_out, model, route)

def trim_history(history: list):
    """يقص السياق: آخر N رسائل وبحد أقصى للحروف، مع الإبقاء دائماً على آخر رسالة."""
    if not history:
        return history, 0
    original = len(history)
    trimmed = history[-HISTORY_MAX_MESSAGES:]
    while len(trimmed) > 1 and sum(len(m.get("content", "")) for m in trimmed) > HISTORY_MAX_CHARS:
        trimmed = trimmed[1:]
    # أول رسالة يجب أن تكون user
    while len(trimmed) > 1 and trimmed[0].get("role") != "user":
        trimmed = trimmed[1:]
    return trimmed, original - len(trimmed)

def routing_text(prompt: str, history: list) -> str:
    """نص التوجيه: الرسالة الحالية، ومعها آخر سؤال سابق إذا كانت قصيرة (تتابع للسياق)."""
    if history and len(prompt.strip()) < 40:
        prev_users = [m.get("content", "") for m in history[:-1] if m.get("role") == "user"]
        if prev_users:
            return prev_users[-1][:500] + "\n" + prompt
    return prompt

def cache_key(api_key: str, system_prompt: str, messages: list, extra: dict) -> str:
    """مفتاح الكاش معزول لكل مفتاح API (لا يُخزَّن المفتاح نفسه)."""
    payload = json.dumps([system_prompt, messages, sorted((extra or {}).items())], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((hashlib.sha256(api_key.encode()).hexdigest() + payload).encode()).hexdigest()

def cache_get(key: str):
    item = _response_cache.get(key)
    if not item:
        return None
    if time.time() - item[0] > CACHE_TTL:
        _response_cache.pop(key, None)
        return None
    _response_cache.move_to_end(key)
    return item

def cache_set(key: str, value: tuple):
    _response_cache[key] = (time.time(),) + value
    _response_cache.move_to_end(key)
    while len(_response_cache) > CACHE_MAX:
        _response_cache.popitem(last=False)

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
    no_cache: Optional[bool] = False  # للاختبارات: تعطيل الكاش
    history: Optional[list] = None   # اختياري: [{"role": "user"|"assistant", "content": "..."}]

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

# ✨ تم إضافة كلمات مفتاحية جديدة لاصطياد الأسئلة الصعبة
_HARD_AR_WORDS = [
    "كود", "اكواد", "أكواد", "برمجة", "برمجه", "دالة", "خطأ", "خطا", "تحليل بيانات", "خطة عمل",
    "قاعدة بيانات", "استعلام", "جدول", "حلل", "تحليل", "قارن", "مقارنة", "صمم", "تصميم", "ناقش", 
    "معمارية", "أداء", "تحسين", "بايثون", "جافا", "جافاسكريبت", "خوارزمية", "نظام", "تطبيق", 
    "مشروع", "قطار", "رياضيات", "احسب", "اشرح"
]

_HARD_AR_RE = re.compile(
    r"(?<!\w)(?:[وف]?(?:لل|[بلك]?(?:ال)?))(?:" + "|".join(_HARD_AR_WORDS) + r")(?!\w)"
)

# ✨ تم إضافة كلمات إنجليزية جديدة (SQL, Database, Analyze, إلخ)
_HARD_EN_RE = re.compile(r"\b(code|function|error|bug|debug|data analysis|business plan|algorithm|sql|database|query|analyze|compare|design|architecture|optimize|python|javascript|java|train|math)\b", re.I)

# ✨ تم إضافة كلمات SQL لاكتشاف الأكواد المكتوبة مباشرة
_CODE_RE = re.compile(r"(^|\s)(def |class |import |function |const |let |var |public |private |select |from |where |join )", re.I)

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
    if _HARD_EN_RE.search(text) or _HARD_AR_RE.search(text):
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
                                history: list = None, extra: dict = None, use_cache: bool = True):
    provider = detect_provider(api_key)
    try:
        strong_model, economy_model = await get_dynamic_models(provider, api_key)
    except InvalidKey:
        return {"status": "error", "error": "مفتاح API غير صالح أو منتهي."}
    except RuntimeError:
        return {"status": "error", "error": "لا توجد موديلات متاحة لهذا المفتاح."}

    router_cost = 0.0

    # 1) تقليل التوكنز: قص السياق المرسل
    history_dropped = 0
    if history:
        history, history_dropped = trim_history(history)
        base_messages = list(history)
    else:
        base_messages = [{"role": "user", "content": prompt}]

    # 2) كاش: نفس السؤال بنفس السياق = بدون أي استدعاء للمزود (تكلفة صفر)
    cacheable = use_cache and (extra or {}).get("temperature") in (None, 0, 0.0)
    ckey = cache_key(api_key, system_prompt, base_messages, extra) if cacheable else None
    if ckey:
        hit = cache_get(ckey)
        if hit:
            _, c_answer, c_in, c_out, c_model, c_route = hit
            c_strong = get_live_cost(strong_model, c_in, c_out)
            known = c_strong is not None
            return {
                "status": "success", "provider": provider, "route": c_route,
                "model_used": c_model, "strong_model_selected": strong_model,
                "economy_model_selected": economy_model, "answer": c_answer,
                "tokens_in": 0, "tokens_out": 0,
                "cost_strong_provider": c_strong or 0.0,
                "new_cost": 0.0, "router_cost": 0.0,
                "routing_saved": (c_strong or 0.0) if known else 0.0,
                "baseline_model": strong_model, "pricing_known": known,
                "cache_hit": True, "history_dropped": history_dropped,
                "cost_gpt4o": c_strong or 0.0,
                "total_saved": (c_strong or 0.0) if known else 0.0
            }

    # 3) التوجيه (يراعي السياق للرسائل القصيرة)
    r_text = routing_text(prompt, base_messages)
    heuristic = smart_heuristic_router(prompt)
    if heuristic == "UNKNOWN" and r_text != prompt:
        heuristic = smart_heuristic_router(r_text)

    if heuristic == "EASY":
        model_chosen, route_type = economy_model, "EASY"
    elif heuristic == "HARD":
        model_chosen, route_type = strong_model, "HARD"
    else:
        try:
            # ✨ تم تعديل تعليمات المصنف لتكون أكثر صرامة
            prompt_router = (
                "أنت خبير في تصنيف الأسئلة. صنف السؤال إلى EASY أو HARD.\n"
                "- EASY: تحيات، أسئلة عامة بسيطة، معلومات بديهية، ترجمة كلمات قصيرة.\n"
                "- HARD: برمجة، أكواد، SQL، تحليل، مقارنة، تصميم أنظمة، رياضيات معقدة، خطط عمل، أو أي سؤال يتطلب خطوات تفكير متعددة أو إجابة طويلة.\n"
                "اكتب كلمة واحدة فقط: EASY أو HARD."
            )
            route_decision = await acompletion(
                model=economy_model,
                messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": r_text}],
                temperature=0, max_tokens=5, api_key=api_key, drop_params=True, timeout=20
            )
            u = route_decision.usage
            rc = get_live_cost(economy_model, u.prompt_tokens, u.completion_tokens)
            router_cost = rc or 0.0
            route_type = (route_decision.choices[0].message.content or "").strip().upper()
            route_type = "HARD" if "HARD" in route_type else "EASY"
        except Exception as e:
            logger.warning("router classification failed: %s", type(e).__name__)
            route_type = "EASY"

        model_chosen = strong_model if route_type == "HARD" else economy_model

    # الاختصار الشديد يُطبَّق على الأسئلة السهلة فقط
    eff_system = system_prompt
    if route_type == "HARD" and system_prompt == DEFAULT_SYSTEM_PROMPT:
        eff_system = HARD_SYSTEM_PROMPT

    messages = [{"role": "system", "content": eff_system}] + base_messages

    # 4) الاستدعاء مع تصعيد تلقائي للموديل القوي إذا فشل الاقتصادي
    attempts = [model_chosen] if model_chosen == strong_model else [model_chosen, strong_model]
    response = None
    for m in attempts:
        try:
            response = await acompletion(
                model=m, messages=messages,
                api_key=api_key, drop_params=True, timeout=REQUEST_TIMEOUT,
                num_retries=1, **(extra or {})
            )
            model_chosen = m
            break
        except Exception as e:
            logger.error("completion failed (%s): %s", m, type(e).__name__)
    if response is None:
        return {"status": "error", "error": "فشل الطلب إلى المزود. تحقق من المفتاح أو الموديل."}

    answer = response.choices[0].message.content or ""
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens

    if ckey and answer:
        cache_set(ckey, (answer, tokens_in, tokens_out, model_chosen, route_type))

    # مقارنة: نفس التوكنز لو استُخدم أقوى موديل متاح لمفتاح المستخدم
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
        "cache_hit": False,
        "history_dropped": history_dropped,
        # الاسم القديم محفوظ للتوافق مع الواجهة: يمثل تكلفة أقوى موديل لمزود المستخدم
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
    return await process_routing_logic(request.prompt, request.api_key.strip(), history=history,
                                       use_cache=not request.no_cache)

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
