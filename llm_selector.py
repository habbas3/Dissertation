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

# ------------------------------------------------------------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "enum": [
            "cnn_features_1d", "cnn_features_1d_sa", "cnn_openmax",
            "WideResNet", "WideResNet_sa", "WideResNet_edited"
        ]},
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

SYSTEM_PROMPT = """\
You are a model-selection assistant for time-series fault diagnosis / battery health and CWRU vibration data.
Given a SHORT dataset description and SMALL numeric summary, recommend ONE configuration that is likely to perform best:
- choose architecture: one of [cnn_features_1d, cnn_features_1d_sa, cnn_openmax, WideResNet, WideResNet_sa, WideResNet_edited]
- toggles: self_attention, sngp, openmax, use_unknown_head
- training knobs: bottleneck (16..1024), dropout (0..0.8), learning_rate (1e-5..1e-2), batch_size (4..256), lambda_src (0..5)

Constraints:
- Prefer self_attention for long sequences (>512) and many channels (>=16).
- Prefer SNGP for label-inconsistent or open-set validation (unknowns/outliers exist) to improve calibration.
- Prefer cnn_openmax ONLY if explicit open-set classification is required and downstream expects OpenMax.
- If data is small (<2k samples), favor lower learning_rate (e.g., 3e-4..1e-4) and higher dropout (0.2..0.5).
- For CWRU 32x1024 style, CNN or WRN variants with SA often perform well. For Battery time-series (7 channels), start with CNN; add SA when sequence_length >= 256.
- Return VALID JSON ONLY, matching the provided schema, with NO extra text.
- Include a short "rationale" string (1–3 sentences) that explains why this configuration fits the input.
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

def _validate_or_default(payload: str) -> Dict[str, Any]:
    try:
        obj = json.loads(payload)
        # Minimal validation against schema keys/ranges
        mn = obj.get("model_name", "")
        if mn not in [ "cnn_features_1d", "cnn_features_1d_sa", "cnn_openmax",
                       "WideResNet", "WideResNet_sa", "WideResNet_edited" ]:
            raise ValueError("invalid model_name")
        # Clamp/Defaults
        obj.setdefault("self_attention", mn.endswith("_sa") or mn == "WideResNet_sa")
        obj.setdefault("sngp", False)
        obj.setdefault("openmax", (mn == "cnn_openmax"))
        obj.setdefault("use_unknown_head", False)
        obj["bottleneck"] = int(max(16, min(1024, int(obj.get("bottleneck", 256)))))
        obj["dropout"] = float(max(0.0, min(0.8, float(obj.get("dropout", 0.3)))))
        obj["learning_rate"] = float(max(1e-5, min(1e-2, float(obj.get("learning_rate", 3e-4)))))
        obj["batch_size"] = int(max(4, min(256, int(obj.get("batch_size", 64)))))
        obj["lambda_src"] = float(max(0.0, min(5.0, float(obj.get("lambda_src", 1.0)))))
        obj["rationale"] = str(obj.get("rationale", "Heuristic fit based on channels, sequence length, and label inconsistency."))
        return obj
    except Exception:
        return {
            "model_name": "cnn_features_1d_sa",
            "self_attention": True,
            "sngp": False,
            "openmax": False,
            "use_unknown_head": False,
            "bottleneck": 256,
            "dropout": 0.3,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "lambda_src": 1.0,
            "rationale": "Fallback selection for robust performance on multi-channel sequences."
        }

# ---------------------------- Provider adapters --------------------------------

def call_openai(text_context: str,
                num_summary: Dict[str, Any],
                model: str = "gpt-4.1-mini",
                debug_dir: Optional[str] = None) -> Dict[str, Any]:
    """Use official OpenAI SDK + Responses API with JSON output."""
    if not OPENAI_OK:
        raise RuntimeError("openai SDK not installed. pip install openai")
    import os
    if os.getenv("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") is None:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")
    client = OpenAI()
    prompt = _build_user_prompt(text_context, num_summary)
    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_output_tokens=600,
        temperature=0.2,
    )
    txt = resp.output_text if hasattr(resp, "output_text") else (
        resp.choices[0].message["content"] if hasattr(resp, "choices") else ""
    )
    if debug_dir:
        _safe_write(f"{debug_dir}/openai_request_user.txt", prompt)
        _safe_write(f"{debug_dir}/openai_raw.json", txt)
    obj = _validate_or_default(txt)
    obj["_provider"] = "openai"
    obj["_raw"] = txt
    return obj



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
    if backend == "openai" or (backend == "auto" and os.getenv("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")):
        return call_openai(text_context, num_summary, model or "gpt-4.1-mini", debug_dir=debug_dir)
    if backend == "ollama" or backend == "auto":
        return call_ollama(text_context, num_summary, model or "llama3.1:8b", debug_dir=debug_dir)
    raise RuntimeError("No LLM backend available. Set OPENAI_API_KEY or run Ollama locally.")