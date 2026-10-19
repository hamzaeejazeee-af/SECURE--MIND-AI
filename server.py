"""SecureMind AI backend: keeps API keys on the server, never in the browser."""
import os, re, requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__, static_folder="static")
CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_VISION = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")


GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
XAI_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4")
_oa = {}  # remembered model fixes
OA = {"groq": ("https://api.groq.com/openai/v1", lambda: GROQ_KEY, lambda: GROQ_MODEL, "Groq"),
      "xai": ("https://api.x.ai/v1", lambda: XAI_KEY, lambda: XAI_MODEL, "Grok (xAI)")}


def _oa_pick(which):
    base, key, _, _ = OA[which]
    ids = [m["id"] for m in requests.get(base + "/models", headers={"Authorization": "Bearer " + key()}, timeout=20).json().get("data", [])]
    if which == "groq":
        for want in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"):
            if want in ids: return want
        ids = [i for i in ids if not re.search(r"whisper|tts|guard|embed|orpheus", i)]
    else:
        ids = [i for i in ids if "grok" in i and not re.search(r"image|imagine|video", i)]
    return ids[0] if ids else None


def call_openai_compat(which, messages, max_tokens=4000, image=None):
    """Groq / xAI Grok (OpenAI-style). messages = [{'role','content'}...]"""
    base, key, model_of, name = OA[which]
    msgs = [dict(m) for m in messages]
    model = _oa.get(which) or model_of()
    if image and msgs:
        if which == "xai":
            msgs[-1]["content"] = [{"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (image["type"], image["data"])}},
                                   {"type": "text", "text": msgs[-1]["content"]}]
        else:
            model = GROQ_VISION
            msgs[-1]["content"] = [{"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (image["type"], image["data"])}},
                                   {"type": "text", "text": msgs[-1]["content"]}]
    r = None
    for attempt in (0, 1):
        r = requests.post(base + "/chat/completions", headers={"Authorization": "Bearer " + key(), "content-type": "application/json"},
                          json={"model": model, "messages": msgs, "max_tokens": max_tokens}, timeout=180)
        if r.status_code in (400, 404) and attempt == 0 and not image:
            alt = _oa_pick(which)
            if alt and alt != model:
                model = _oa[which] = alt
                continue
        break
    if r.status_code == 401: raise RuntimeError("%s refused the key. Paste it again in .env (Groq keys start gsk_, xAI keys start xai-)." % name)
    if r.status_code in (402, 403) and which == "xai": raise RuntimeError("xAI says this key has no credit - Grok is paid. Use a free Groq key instead (console.groq.com/keys).")
    if r.status_code == 429: raise RuntimeError("%s limit reached for now. Wait a minute and try again." % name)
    if r.status_code == 413: raise RuntimeError("That was too much text for the %s free limit. Try shorter input or fewer tests." % name)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
_gm = {"name": None}
NO_AI_MSG = ("No AI is set up yet. Easiest fix: get a FREE Groq key at https://console.groq.com/keys , "
             "open the .env file and write GROQ_API_KEY=your_key , then restart. (Or start Ollama for a no-key option.)")


def ollama_up():
    try:
        return requests.get(OLLAMA_URL + "/api/tags", timeout=1.2).ok
    except Exception:
        return False


def resolve_provider(p):
    """auto / missing-key choices fall back to whatever is actually available."""
    keys = {"groq": GROQ_KEY, "xai": XAI_KEY, "gemini": GEMINI_KEY, "claude": CLAUDE_KEY}
    if p == "ollama": return "ollama"
    if p in keys and keys[p]: return p
    for name in ("groq", "xai", "gemini", "claude"):
        if keys[name]: return name
    if ollama_up(): return "ollama"
    return None


def _gemini_pick_model():
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                     headers={"x-goog-api-key": GEMINI_KEY}, timeout=20)
    names = [m["name"].replace("models/", "") for m in r.json().get("models", [])
             if "flash" in m["name"] and "generateContent" in m.get("supportedGenerationMethods", [])
             and not re.search(r"image|tts|live|audio|robotics|computer|embedding|exp", m["name"])]
    names.sort(key=lambda n: "preview" in n)
    return names[0] if names else None


def call_gemini(system, contents, max_tokens=8000, json_mode=False):
    """contents = [{'role': 'user'|'model', 'parts': [{'text': ..}, {'inline_data': ..}]}]"""
    body = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
    if system: body["system_instruction"] = {"parts": [{"text": system}]}
    if json_mode: body["generationConfig"]["responseMimeType"] = "application/json"
    r = None
    for attempt in (0, 1):
        model = _gm["name"] or GEMINI_MODEL
        r = requests.post("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model,
                          headers={"x-goog-api-key": GEMINI_KEY, "content-type": "application/json"}, json=body, timeout=180)
        if r.status_code == 404 and attempt == 0:
            alt = _gemini_pick_model()
            if alt and alt != model:
                _gm["name"] = alt
                continue
        break
    if r.status_code == 429: raise RuntimeError("Gemini free limit reached for now. Wait a minute and try again.")
    if r.status_code in (400, 403):
        raise RuntimeError("Gemini refused the key. Create a fresh one at https://aistudio.google.com/apikey and put it in .env (%s)" % r.text[:120])
    r.raise_for_status()
    j = r.json()
    cand = (j.get("candidates") or [{}])[0]
    return "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
@app.get("/")
def home():
    return send_from_directory("static", "index.html")

def ask_claude(msgs, img):
    if not CLAUDE_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
    if img:
        last = msgs[-1]
        last["content"] = [{"type": "image", "source": {"type": "base64", "media_type": img["type"], "data": img["data"]}},
                           {"type": "text", "text": last["content"]}]
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": CLAUDE_MODEL, "max_tokens": 4096, "messages": msgs}, timeout=90)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])

def ask_gemini(msgs, img):
    system = "\n".join(m["content"] for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str))
    contents = []
    for m in msgs:
        if m.get("role") == "system": continue
        c = m.get("content")
        parts = [{"text": c}] if isinstance(c, str) else [{"text": str(c)}]
        contents.append({"role": "model" if m.get("role") == "assistant" else "user", "parts": parts})
    if img and contents:
        contents[-1]["parts"].insert(0, {"inline_data": {"mime_type": img["type"], "data": img["data"]}})
    return call_gemini(system, contents, 4096)


def ask_ollama(msgs, img):
    if img:
        msgs[-1]["images"] = [img["data"]]
    r = requests.post(OLLAMA_URL + "/api/chat",
                      json={"model": OLLAMA_VISION if img else OLLAMA_MODEL, "messages": msgs, "stream": False}, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

@app.post("/api/chat")
def chat():
    d = request.get_json(force=True)
    try:
        prov = resolve_provider(d.get("provider"))
        if not prov:
            raise RuntimeError(NO_AI_MSG)
        if prov in OA:
            return jsonify(text=call_openai_compat(prov, d["messages"], 4096, d.get("image")))
        fn = {"ollama": ask_ollama, "gemini": ask_gemini}.get(prov, ask_claude)
        return jsonify(text=fn(d["messages"], d.get("image")))
    except Exception as e:
        return jsonify(error=str(e)), 500

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _save_env(name, value):
    lines = open(ENV_PATH, encoding="utf-8").read().splitlines() if os.path.exists(ENV_PATH) else []
    for i, l in enumerate(lines):
        if re.match(r"\s*%s\s*=" % name, l):
            lines[i] = "%s=%s" % (name, value); break
    else:
        lines.append("%s=%s" % (name, value))
    open(ENV_PATH, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def _is_local():
    # Only the person at the computer running the app may use AI Setup. Off when online (proxy headers or DISABLE_SETUP=1).
    if os.getenv("DISABLE_SETUP") == "1" or request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded"):
        return False
    return request.remote_addr in ("127.0.0.1", "::1")


@app.get("/api/health")
def health():
    return jsonify(groq=bool(GROQ_KEY), xai=bool(XAI_KEY), gemini=bool(GEMINI_KEY), claude=bool(CLAUDE_KEY),
                   ollama=ollama_up(), can_setup=_is_local())


@app.post("/api/setup")
def setup():
    """AI Setup box: saves a Groq (gsk_) or Grok (xai-) key into .env. Only works on the computer running the app."""
    global GROQ_KEY, XAI_KEY
    if not _is_local():
        return jsonify(error="AI Setup only works on the computer that runs the app. On a live website, add GROQ_API_KEY in the host's Environment settings instead."), 403
    key = str((request.get_json(force=True, silent=True) or {}).get("key", "")).strip()
    if key.startswith("gsk_"): which, env = "groq", "GROQ_API_KEY"
    elif key.startswith("xai-"): which, env = "xai", "XAI_API_KEY"
    else:
        return jsonify(error="That does not look right. A Groq key starts with gsk_ and a Grok (xAI) key starts with xai-."), 400
    note = ""
    try:
        r = requests.get(OA[which][0] + "/models", headers={"Authorization": "Bearer " + key}, timeout=15)
        if r.status_code == 401:
            return jsonify(error="That key was rejected. Copy it again from the website (no spaces)."), 400
    except Exception:
        note = " (could not double-check it online right now, but it is saved)"
    _save_env(env, key); os.environ[env] = key
    if which == "groq": GROQ_KEY = key
    else: XAI_KEY = key
    return jsonify(ok=True, provider=which, message="Saved. AI is ready" + note + ".")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
