"""One-click starter. Double-click (Windows: start.bat) or run:  python run.py
It installs what is missing, helps you paste a free AI key once, and opens the app."""
import os, re, subprocess, sys, threading, time, urllib.request, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def ensure_packages():
    try:
        import flask, requests, dotenv  # noqa: F401
    except ImportError:
        print("Installing required packages (one time, needs internet)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def read_env():
    vals = {}
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
            if m: vals[m.group(1)] = m.group(2)
    return vals


def set_env(key, value):
    lines = open(".env", encoding="utf-8").read().splitlines() if os.path.exists(".env") else []
    for i, l in enumerate(lines):
        if re.match(r"\s*%s\s*=" % key, l):
            lines[i] = "%s=%s" % (key, value); break
    else:
        lines.append("%s=%s" % (key, value))
    open(".env", "w", encoding="utf-8").write("\n".join(lines) + "\n")


def ollama_up():
    try:
        return urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5).status == 200
    except Exception:
        return False


def main():
    ensure_packages()
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        open(".env", "w", encoding="utf-8").write(open(".env.example", encoding="utf-8").read())
    env = read_env()
    has_key = any(env.get(k) for k in ("GROQ_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"))
    if not has_key and ollama_up():
        print("No key found, but Ollama is running - the app will use Ollama (free, no key).")
    elif not has_key:
        print("\n=== One-time AI setup ===")
        print("Easiest FREE key (Groq): open  https://console.groq.com/keys  -> sign up -> 'Create API Key' -> copy it.")
        print("(Starts with gsk_ ; no card needed. A Grok/xAI key starting xai- also works but is paid.)")
        print("Press Enter to skip and use the built-in offline mode.\n")
        try:
            key = input("Paste your key here: ").strip()
        except EOFError:
            key = ""
        if key:
            set_env("XAI_API_KEY" if key.startswith("xai-") else "ANTHROPIC_API_KEY" if key.startswith("sk-ant-") else "GEMINI_API_KEY" if key.startswith("AIza") else "GROQ_API_KEY", key)
            print("Saved to .env")
    from dotenv import load_dotenv
    load_dotenv(override=True)
    port = int(os.getenv("PORT", "5000"))
    threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open("http://localhost:%d" % port)), daemon=True).start()
    print("\nStarting on http://localhost:%d   (press Ctrl+C to stop)\n" % port)
    import server
    server.app.run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
