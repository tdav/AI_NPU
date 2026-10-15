"""Адаптеры архитектур: model_type из config.json -> класс, который строит статические части."""
from .gemma4 import MODEL_TYPES as GEMMA4_TYPES, Gemma4Text
from .lfm2 import MODEL_TYPES as LFM2_TYPES, Lfm2

REGISTRY = {**{t: Gemma4Text for t in GEMMA4_TYPES}, **{t: Lfm2 for t in LFM2_TYPES}}


def adapter_for(model_type):
    if model_type not in REGISTRY:
        raise SystemExit(f"Архитектура «{model_type}» пока не поддерживается. Поддержаны: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[model_type]
