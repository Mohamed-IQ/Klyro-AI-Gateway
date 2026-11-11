from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import uvicorn
from litellm import completion, get_model_info, get_llm_provider

app = FastAPI()

# إعدادات CORS للسماح للواجهة بالاتصال
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

# --- 1. خريطة النماذج القوية والاقتصادية لكل مزود ---
PROVIDER_MODEL_MAP = {
    "groq": {
        "strong": "groq/openai/gpt-oss-120b",
        "economy": "groq/openai/gpt-oss-20b",
    },
    "openai": {
        "strong": "gpt-4o",
        "economy": "gpt-4o-mini",
    },
    "anthropic": {
        "strong": "anthropic/claude-3-5-sonnet-20240620",
        "economy": "anthropic/claude-3-haiku-20240307",
    },
    "openrouter": {
        "strong": "openrouter/anthropic/claude-3.5-sonnet",
        "economy": "openrouter/meta-llama/llama-3-8b-instruct",
    },
    "deepseek": {
        "strong": "deepseek/deepseek-chat",
        "economy": "deepseek/deepseek-chat",
    },
}

def detect_provider_and_models(api_key: str):
    """كشف المزود والنماذج بناءً على مفتاح API"""
    try:
        # استخدام LiteLLM لكشف المزود
        model, provider, _, _ = get_llm_provider(model="gpt-3.5-turbo", api_key=api_key)
        provider = provider.lower()
    except Exception:
        # اكتشاف يدوي احتياطي بناءً على البادئة
        if api_key.startswith("gsk_"):
            provider = "groq"
        elif api_key.startswith("sk-ant-"):
            provider = "anthropic"
        elif api_key.startswith("sk-or-"):
            provider = "openrouter"
        else:
            provider = "openai"  # الافتراضي

    models = PROVIDER_MODEL_MAP.get(provider, PROVIDER_MODEL_MAP["openai"])
    return provider, models

def is_simple_prompt(prompt: str) -> bool:
    """فلتر سريع جداً (أقل من 1 مللي ثانية) لتحديد الأسئلة البسيطة جداً"""
    text = prompt.strip()
    # إذا كان السؤال قصيراً جداً (أقل من 10 كلمات)
    if len(text.split()) <= 10:
        return True
    # إذا كان يحتوي على تحية أو سؤال عام
    simple_patterns = [
        r"^(مرحبا|أهلا|السلام عليكم|هاي|hello|hi|hey)",
        r"(كيف حالك|شنو أخبارك|من أنت|ما اسمك)",
        r"(شكرا|thanks|thank you)",
    ]
    for pattern in simple_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    # إذا كان لا يحتوي على كلمات تدل على التعقيد
    complex_keywords = ["كود", "code", "دالة", "function", "خطأ", "error", "تحليل", "analysis", "خطة", "plan", "اشرح", "explain"]
    if not any(keyword in text.lower() for keyword in complex_keywords):
        return True

    return False

@app.post("/route")
async def route_chat(request: ChatRequest):
    provider, models = detect_provider_and_models(request.api_key)
    print(f"--- 1. المزود المكتشف: {provider} ---")

    # --- 2. الفلتر السريع (Heuristic Bypass) للأسئلة الواضحة جداً ---
    if is_simple_prompt(request.prompt):
        model_chosen = models["economy"]
        model_name = f"ECONOMY (Fast Track) - {provider}"
        print(f"--- 2. توجيه مباشر للاقتصادي (بدون موزع): {model_chosen} ---")
    else:
        # --- 3. الموزع الخبير (Expert Router) للأسئلة المتوسطة والمعقدة ---
        try:
            # تعليمات احترافية للموزع (Few-Shot Prompting)
            prompt_router = """
أنت خبير تصنيف مهام (Task Router) في شركة تقنية.
مهمتك: تصنيف سؤال المستخدم إلى EASY أو HARD. اكتب كلمة واحدة فقط.

قواعد التصنيف:
- EASY: التحيات، الأسئلة العامة، الترجمات البسيطة، التعريفات القصيرة، الأسئلة الحسابية المباشرة، المحادثات العادية.
  أمثلة: "مرحبا"، "شنو عاصمة فرنسا؟"، "ترجم لي هذه الكلمة"، "2+2 كم؟".

- HARD: كتابة أكواد برمجية، تصحيح أخطاء (Debugging)، تحليل بيانات، كتابة مقالات طويلة، وضع خطط عمل، حل مسائل رياضية معقدة، أو أي سؤال يتطلب تفكيراً منطقياً متسلسلاً.
  أمثلة: "اكتب لي سكربت بايثون لسحب بيانات تويتر"، "حلل لي أداء هذا الكود"، "ضع خطة تسويقية لمشروع"، "اشرح لي نظرية النسبية بالتفصيل".

إذا كان السؤال غامضاً، صنفه HARD لضمان الجودة.
اكتب EASY أو HARD فقط. لا تكتب أي شيء آخر.
"""
            # نستخدم النموذج الاقتصادي نفسه كموزع (لأنه الأرخص)
            route_decision = completion(
                model=models["economy"],
                messages=[{"role": "system", "content": prompt_router}, {"role": "user", "content": request.prompt}],
                temperature=0,
                api_key=request.api_key,
                drop_params=True
            )
            route_type = route_decision.choices[0].message.content.strip().upper()
            print(f"--- 3. قرار الموزع: {route_type} ---")
        except Exception as e:
            return {"error": f"فشل الاتصال بالموزع: {str(e)}", "status": "error"}

        if "HARD" in route_type:
            model_chosen = models["strong"]
            model_name = f"STRONG (Heavy Model) - {provider}"
        else:
            model_chosen = models["economy"]
            model_name = f"ECONOMY (Light Model) - {provider}"
        print(f"--- 4. التوجيه النهائي: {model_chosen} ---")

    # --- 4. توليد الإجابة ---
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

    # --- 5. حسابات التكلفة والوفر ---
    try:
        baseline_info = get_model_info(model="gpt-4o")
        baseline_in = baseline_info.get("input_cost_per_token", 2.50 / 1_000_000)
        baseline_out = baseline_info.get("output_cost_per_token", 10.00 / 1_000_000)
    except Exception:
        baseline_in, baseline_out = 2.50 / 1_000_000, 10.00 / 1_000_000

    old_cost = (tokens_in * baseline_in) + (tokens_out * baseline_out)

    try:
        model_info = get_model_info(model=model_chosen)
        price_in = model_info.get("input_cost_per_token", 0.0)
        price_out = model_info.get("output_cost_per_token", 0.0)
        real_cost_new = (tokens_in * price_in) + (tokens_out * price_out)
    except Exception:
        # خطة احتياطية لأسعار Groq إذا فشل جلب السعر
        real_cost_new = (tokens_in * 0.075 / 1_000_000) + (tokens_out * 0.30 / 1_000_000)

    saved = max(0, old_cost - real_cost_new)

    # --- 6. إرجاع النتيجة النهائية ---
    return {
        "status": "success",
        "provider": provider,
        "route": "HARD" if "STRONG" in model_name else "EASY",
        "model_used": model_name,
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "old_cost": old_cost,
        "new_cost": real_cost_new,
        "saved": saved
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
