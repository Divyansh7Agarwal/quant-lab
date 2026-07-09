"""Bake the live backend into a static snapshot for GitHub Pages.

Runs each API function once, writes dashboard/site/data.json + a copy of index.html.
The same index.html works both ways: locally it hits the live server; on Pages it
finds data.json and serves the daily snapshot. Never raises on a single bad endpoint.
"""
import os, sys, json, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import server


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:                        # noqa: BLE001
        return {"error": str(e)}


ops = safe(server.ops)
if isinstance(ops, dict):
    ops["hosting"] = "GitHub Actions (cloud) — runs weekdays 11:00 UTC, no Mac required"

data = {
    "/api/overview": safe(server.overview),
    "/api/tilts": {"tilts": (server.tilts() if callable(server.tilts) else [])},
    "/api/score": safe(server.score),
    "/api/plan": safe(server.plan),
    "/api/paper": safe(server.paper),
    "/api/funding/live": safe(server.funding_live),
    "/api/graveyard": {"ideas": server.GRAVEYARD},
    "/api/ops": ops,
}

site = os.path.join(HERE, "site")
os.makedirs(site, exist_ok=True)
shutil.copy(os.path.join(HERE, "index.html"), os.path.join(site, "index.html"))
json.dump(data, open(os.path.join(site, "data.json"), "w"))
open(os.path.join(site, ".nojekyll"), "w").close()   # serve files as-is
print(f"built static site -> {site} ({len(json.dumps(data))} bytes of data)")
