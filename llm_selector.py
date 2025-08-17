#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 13:07:22 2025

@author: habbas
"""

# llm_selector.py
import os, json, math, textwrap
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# ---- Optional providers -------------------------------------------------------
# Use OpenAI if OPENAI_API_KEY is present (official SDK); else try Ollama (local).
# OpenAI "Responses API" is the current recommended path and supports structured outputs. 
# Docs: https://platform.openai.com/docs/api-reference/responses  :contentReference[oaicite:0]{index=0}
# Official SDK install: pip install openai  (Libraries page) :contentReference[oaicite:1]{index=1}
OPENAI_OK = False
try:
    from openai import OpenAI  # official SDK (2024+)  :contentReference[oaicite:2]{index=2}
    OPENAI_OK = True
except Exception:
    pass

OLLAMA_OK = False
try:
    import ollama  # official Ollama Python SDK  :contentReference[oaicite:3]{index=3}
    OLLAMA_OK = True
except Exception:
    # fallback to REST later if needed
    pass

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass



import re
from pathlib import Path

# -------------------- helpers (architectures + normalization) --------------------

_ALLOWED_MODELS = [
    "cnn_features_1d", "cnn_features_1d_sa", "cnn_openmax",
    "WideResNet", "WideResNet_sa", "WideResNet_edited"
]

_ALLOWED_ARCH = [
    "cnn_1d", "cnn_1d_sa", "cnn_openmax",
    "wideresnet", "wideresnet_sa", "wideresnet_edited"
]

def _normalize_arch(s: str) -> str:
    if not s:
        return ""
    t = s.strip().lower().replace("-", "_")
    aliases = {
        "cnn": "cnn_1d",
        "cnn1d": "cnn_1d",
        "cnn_features_1d": "cnn_1d",
        "cnn_features_1d_sa": "cnn_1d_sa",
        "wrn": "wideresnet",
        "wideresnet_sa": "wideresnet_sa",
        "wideresnet_edited": "wideresnet_edited",
    }
    if t in aliases:
        t = aliases[t]
    # keep only our known set; otherwise empty
    return t if t in _ALLOWED_ARCH else ""

def _normalize_model_name(s: str) -> str:
    if not s:
        return ""
    t = s.strip().replace("-", "_")
    # match allowed names case-insensitively
    for m in _ALLOWED_MODELS:
        if t.lower() == m.lower():
            return m
    # common aliases
    alias = {
        "cnn": "cnn_features_1d",
        "cnn_1d": "cnn_features_1d",
        "cnn_sa": "cnn_features_1d_sa",
        "openmax": "cnn_openmax",
        "wrn": "WideResNet",
        "wrn_sa": "WideResNet_sa",
        "wideresnet": "WideResNet",
        "wideresnetsa": "WideResNet_sa",
        "wideresnet_edited": "WideResNet_edited",
    }
    t2 = t.lower().replace("_", "")
    return alias.get(t2, "")

def _arch_to_model_name(arch: str, self_attention: bool, openmax: bool) -> str:
    """Map architecture + toggles to one of your concrete model_name strings."""
    a = (arch or "").lower()
    if a.startswith("cnn"):
        if openmax:
            return "cnn_openmax"
        return "cnn_features_1d_sa" if self_attention else "cnn_features_1d"
    if a.startswith("wideresnet_edited"):
        return "WideResNet_edited"
    if a.startswith("wideresnet"):
        return "WideResNet_sa" if self_attention else "WideResNet"
    # unknown arch -> fallback (lightweight default)
    return "cnn_features_1d_sa"

def _clamp(v, lo, hi, default):
    try:
        if isinstance(default, int):
            return int(max(lo, min(hi, int(v))))
        return float(max(lo, min(hi, float(v))))
    except Exception:
        return default

def _autofill_rationale(cfg: dict, num_summary: dict, provider_reason: str = "") -> str:
    ch = num_summary.get("channels")
    sl = num_summary.get("seq_len")
    notes = str(num_summary.get("notes", "")).lower()
    parts = []
    if provider_reason:
        parts.append(provider_reason.strip())
    if ch is not None and sl is not None:
        parts.append(f"Input has {ch} channels and sequence length {sl}.")
    if "label_inconsistent" in notes or "open" in notes:
        parts.append("Label inconsistency/open-set risk detected; prefer calibrated heads (SNGP/OpenMax) when appropriate.")
    arch = cfg.get("architecture", "")
    if arch:
        parts.append(f"Architecture chosen: {arch}.")
    if cfg.get("self_attention", False):
        parts.append("Self-attention enabled for long-range temporal dependencies.")
    if cfg.get("sngp", False):
        parts.append("SNGP enabled for calibrated uncertainty under domain shift.")
    if cfg.get("openmax", False):
        parts.append("OpenMax enabled for explicit unknown rejection.")
    return " ".join(parts) if parts else "Configuration selected from data shape and task setup."
# -------------------------------------------------------------------------------



def _list_model_architectures() -> list[str]:
    """Return available model architectures by parsing models/__init__.py."""
    init_py = Path(__file__).resolve().parent / "models" / "__init__.py"
    names: list[str] = []
    try:
        with init_py.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("from") or "import" not in line:
                    continue
                target = line.split("import", 1)[1].split("#", 1)[0]
                for part in target.split(","):
                    part = part.strip()
                    if " as " in part:
                        part = part.split(" as ", 1)[1].strip()
                    if re.search(
                        r"Adversarial|sngp|spectral|attention|classifier|auxiliary",
                        part,
                        re.IGNORECASE,
                    ):
                        continue
                    names.append(part)
    except Exception:
        pass
    fallback = [
        "cnn_features_1d",
        "cnn_features_1d_sa",
        "cnn_openmax",
        "WideResNet",
        "WideResNet_sa",
        "WideResNet_edited",
    ]
    return sorted(set(names)) or fallback


MODEL_ARCHS = _list_model_architectures()


# ------------------------------------------------------------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "enum": MODEL_ARCHS},
        "self_attention": {"type": "boolean"},
        "sngp": {"type": "boolean"},
        "openmax": {"type": "boolean"},
        "use_unknown_head": {"type": "boolean"},
        "bottleneck": {"type": "integer", "minimum": 16, "maximum": 1024},
        "dropout": {"type": "number", "minimum": 0.0, "maximum": 0.8},
        "learning_rate": {"type": "number", "minimum": 1e-5, "maximum": 1e-2},
        "batch_size": {"type": "integer", "minimum": 4, "maximum": 256},
        "lambda_src": {"type": "number", "minimum": 0.0, "maximum": 5.0},
        "rationale": {"type": "string"}
    },
    "required": ["model_name"],
    "additionalProperties": False
}

_ARCH_LIST_STR = ", ".join(MODEL_ARCHS)

SYSTEM_PROMPT = """\
You are a model-selection assistant for time-series fault diagnosis / battery health and CWRU vibration data.
Given a SHORT dataset description and SMALL numeric summary, recommend ONE configuration likely to perform best.

Return STRICT JSON with these fields (no prose outside JSON):
- architecture: one of [cnn_1d, cnn_1d_sa, cnn_openmax, wideresnet, wideresnet_sa, wideresnet_edited]
- model_name: one of [cnn_features_1d, cnn_features_1d_sa, cnn_openmax, WideResNet, WideResNet_sa, WideResNet_edited]
- self_attention: boolean
- sngp: boolean
- openmax: boolean
- use_unknown_head: boolean
- bottleneck: integer (16..1024)
- dropout: float (0..0.8)
- learning_rate: float (1e-5..1e-2)
- batch_size: integer (4..256)
- lambda_src: float (0..5)
- rationale: 2–4 sentences referencing channels, seq_len, label consistency/open-set, and trade-offs among SA/SNGP/OpenMax.

Return VALID JSON ONLY (no extra text).
"""


def _summarize_numeric(num_summary: Dict[str, Any]) -> str:
    # Keep payload tiny: just stats, a few class counts, lengths
    lines = []
    for k, v in num_summary.items():
        if isinstance(v, (int, float, str)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, dict):
            # include up to 6 items
            items = list(v.items())[:6]
            kv = ", ".join([f"{ik}={iv}" for ik, iv in items])
            lines.append(f"{k}: {{{kv}}}")
        else:
            lines.append(f"{k}: {str(v)[:120]}")
    return "\n".join(lines)

def _build_user_prompt(text_context: str, num_summary: Dict[str, Any]) -> str:
    summary = _summarize_numeric(num_summary)
    schema_str = json.dumps(JSON_SCHEMA, indent=2)
    return f"""\
DATASET CONTEXT
---------------
{textwrap.shorten(text_context.strip(), width=2000, placeholder='...')}

NUMERIC SUMMARY
---------------
{summary}

REQUIRED JSON SCHEMA
--------------------
{schema_str}

Return ONLY a JSON object that matches the schema.
"""

def _validate_or_default(payload, num_summary=None) -> dict:
    """
    Tolerant validator: parse provider JSON (or text), normalize fields,
    honor 'architecture' if provided, and compose a specific rationale.
    """
    import json
    if num_summary is None:
        num_summary = {}

    # Parse payload -> dict
    obj = {}
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(obj, dict):
            obj = {}
    except Exception:
        obj = {}

    # 1) read raw fields
    raw_arch = str(obj.get("architecture", obj.get("arch", "")) or "").strip()
    raw_model = str(obj.get("model_name", "")).strip()
    raw_self_att = obj.get("self_attention", None)
    raw_sngp = obj.get("sngp", None)
    raw_openmax = obj.get("openmax", None)
    raw_use_unk = obj.get("use_unknown_head", None)

    # 2) normalize architecture and model_name
    arch = _normalize_arch(raw_arch)
    model_name = _normalize_model_name(raw_model)

    # 3) heuristic defaults if missing
    ch = num_summary.get("channels", None)
    sl = num_summary.get("seq_len", None)

    if not arch and model_name:
        # infer arch from model_name
        mn = model_name.lower()
        if "wideresnet_edited" in mn:
            arch = "wideresnet_edited"
        elif "wideresnet" in mn:
            arch = "wideresnet_sa" if "sa" in mn else "wideresnet"
        elif "openmax" in mn:
            arch = "cnn_openmax"
        elif "cnn_features_1d_sa" in mn:
            arch = "cnn_1d_sa"
        else:
            arch = "cnn_1d"

    if not arch and not model_name:
        # choose a sensible default arch from data shape
        if (sl and sl >= 256) or (ch and ch >= 16):
            arch = "wideresnet_sa"
        else:
            arch = "cnn_1d_sa"

    # 4) toggles (prefer provider values; else infer from arch/model)
    self_attention = bool(raw_self_att) if raw_self_att is not None else (
        arch.endswith("_sa") or "sa" in model_name.lower()
    )
    sngp = bool(raw_sngp) if raw_sngp is not None else False
    openmax = bool(raw_openmax) if raw_openmax is not None else (arch == "cnn_openmax")
    use_unknown_head = bool(raw_use_unk) if raw_use_unk is not None else False

    # 5) model_name resolution (architecture + toggles -> concrete model name)
    if not model_name:
        model_name = _arch_to_model_name(arch, self_attention, openmax)

    # 6) numeric hyperparams (with safe clamps)
    bottleneck = _clamp(obj.get("bottleneck", 256), 16, 1024, 256)
    dropout = _clamp(obj.get("dropout", 0.3), 0.0, 0.8, 0.3)
    lr = _clamp(obj.get("learning_rate", 3e-4), 1e-5, 1e-2, 3e-4)
    bs = _clamp(obj.get("batch_size", 64), 4, 256, 64)
    lam_src = _clamp(obj.get("lambda_src", 1.0), 0.0, 5.0, 1.0)

    # 7) rationale
    prov_rat = str(obj.get("rationale", "")).strip()
    if not prov_rat or prov_rat.lower().startswith("fallback"):
        prov_rat = ""

    cfg = {
        "architecture": arch,
        "model_name": model_name,
        "self_attention": bool(self_attention),
        "sngp": bool(sngp),
        "openmax": bool(openmax),
        "use_unknown_head": bool(use_unknown_head),
        "bottleneck": bottleneck,
        "dropout": dropout,
        "learning_rate": lr,
        "batch_size": bs,
        "lambda_src": lam_src,
    }
    cfg["rationale"] = prov_rat or _autofill_rationale(cfg, num_summary, provider_reason="")

    return cfg


# ---------------------------- Provider adapters --------------------------------

def call_openai(text_context: str,
                num_summary: Dict[str, Any],
                model: str = "gpt-4o-mini",
                debug_dir: Optional[str] = None) -> Dict[str, Any]:
    """OpenAI backend using Chat Completions (v1 SDK)."""
    import os, json
    from openai import OpenAI

    # Build prompts
    sys = SYSTEM_PROMPT
    user = _build_user_prompt(text_context, num_summary)

    # Init client (picks up OPENAI_API_KEY / OPENAI_PROJECT / OPENAI_ORG_ID)
    client = OpenAI()

    # Try JSON mode; if the model/SDK doesn’t support response_format, retry without it
    content = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
    except TypeError:
        # No JSON mode → ask for JSON in the system prompt instead
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys + "\nReturn strictly JSON."},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""

    # Debug dumps
    if debug_dir:
        _safe_write(f"{debug_dir}/openai_request_user.txt", user)
        _safe_write(f"{debug_dir}/openai_raw.txt", content)

    # Robust JSON parse (exact → fallback extract)
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        try:
            s, e = content.find("{"), content.rfind("}")
            if s != -1 and e != -1 and e > s:
                obj = json.loads(content[s:e+1])
        except Exception:
            obj = None

    cfg = _validate_or_default(obj if obj is not None else content, num_summary=num_summary)
    cfg["_provider"] = "openai"
    cfg["_raw"] = content
    return cfg




# Save text/JSON safely; ignore errors in debug mode
def _safe_write(path: str, data) -> None:
    try:
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2)
            else:
                f.write(data if isinstance(data, str) else str(data))
    except Exception:
        # Debug writes should never crash the run
        pass


def call_ollama(text_context: str,
                num_summary: Dict[str, Any],
                model: str = "llama3.1:8b",
                debug_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Use local Ollama with JSON-only mode. Saves raw request/response if debug_dir is given.
    """
    import json, os, requests
    sys = SYSTEM_PROMPT
    user = _build_user_prompt(text_context, num_summary)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "format": "json",     # enforce JSON output
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = host.rstrip("/") + "/api/chat"
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        content = r.json()["message"]["content"]
    except Exception as e:
        content = ""
        if debug_dir:
            _safe_write(f"{debug_dir}/ollama_error.txt", f"{type(e).__name__}: {e}")

    if debug_dir:
        _safe_write(f"{debug_dir}/ollama_request_user.txt", user)
        _safe_write(f"{debug_dir}/ollama_request_payload.json", payload)
        _safe_write(f"{debug_dir}/ollama_raw.txt", content)

    # Parse JSON (provider should already be JSON because of format='json')
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        # crude extraction of first {...}
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(content[start:end+1])
        except Exception:
            obj = None

    parsed = _validate_or_default(json.dumps(obj) if obj is not None else content)
    parsed["_provider"] = "ollama"
    parsed["_raw"] = content
    return parsed



# ------------------------------ Public API ------------------------------------

def select_config(text_context: str,
                  num_summary: Dict[str, Any],
                  backend: str = "auto",
                  model: Optional[str] = None,
                  debug_dir: Optional[str] = None) -> Dict[str, Any]:
    if backend == "openai" or (backend == "auto" and os.getenv("OPENAI_API_KEY")):
        return call_openai(text_context, num_summary, model or "gpt-4.1-mini", debug_dir=debug_dir)
    if backend == "ollama" or backend == "auto":
        return call_ollama(text_context, num_summary, model or "llama3.1:8b", debug_dir=debug_dir)
    raise RuntimeError("No LLM backend available. Set OPENAI_API_KEY or run Ollama locally.")