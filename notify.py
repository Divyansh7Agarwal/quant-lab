"""Cross-platform notification. Prefers Telegram (reaches your phone from anywhere,
incl. cloud CI); falls back to a macOS pop-up locally; else prints. Never raises."""
import os, sys, subprocess, urllib.request, urllib.parse


def notify(title, message):
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": f"{title}\n{message}"}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage",
                                   data=data, timeout=10)
            return "telegram"
        except Exception:                         # noqa: BLE001
            pass
    if sys.platform == "darwin":
        try:
            subprocess.run(["osascript", "-e",
                            f'display notification "{message}" with title "{title}" sound name "Glass"'],
                           check=False, timeout=10)
            return "macos"
        except Exception:                         # noqa: BLE001
            pass
    print(f"[NOTIFY] {title}: {message}")
    return "print"
