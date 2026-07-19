#!/usr/bin/env python3
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
REPORT_RES = ROOT / "ax" / "utils" / "report" / "resources"

# ---- Stub ax.plot.render so render.py can import without importing full ax package ----
ax_mod = types.ModuleType("ax")
ax_plot_mod = types.ModuleType("ax.plot")
ax_plot_render_mod = types.ModuleType("ax.plot.render")

ax_plot_render_mod._js_requires = lambda *a, **k: ""
ax_plot_render_mod._load_css_resource = lambda *a, **k: ""

sys.modules["ax"] = ax_mod
sys.modules["ax.plot"] = ax_plot_mod
sys.modules["ax.plot.render"] = ax_plot_render_mod

# ---- Load ax/utils/report/render.py by file path ----
render_path = (ROOT / "ax" / "utils" / "report" / "render.py").resolve()
spec = importlib.util.spec_from_file_location("ax_utils_report_render", render_path)
assert spec and spec.loader, "Failed to load module spec"
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---- Monkeypatch ALL pkgutil-based loaders so we don't need package resources ----
mod._load_css_resource = lambda: ""
mod._load_plot_css_resource = lambda: ""
mod._load_html_template = lambda name: (REPORT_RES / name).read_text(encoding="utf-8")

# ---- Now test escaping ----
payload = "<img src=x onerror=alert('AX_XSS_TEST')>"
html = mod.render_report_elements(payload, [mod.p_html("hello")])

assert "<img" not in html, "FAIL: raw HTML injection still present"
assert "&lt;img" in html, "FAIL: expected escaped payload not found"

print("[OK] experiment_name is escaped (autoescape enabled)")
