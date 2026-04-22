"""persona_lib — abstracted API for the Persona Vectors course project.

Students should only need these imports:

    from persona_lib import (
        ActivationModel, ChatPrompt,
        steering,
        run_extraction,
        score_trait, score_coherence,
        finetune,
        score_samples,
    )

All heavy ML/tokenizer/hook logic lives behind this surface.
"""

from persona_lib.data import ChatPrompt
from persona_lib.model import ActivationModel
from persona_lib.steering import steering
from persona_lib.judge import score_trait, score_coherence
from persona_lib.pipeline import (
    ExtractionResult,
    get_cached_rollouts,
    load_extraction,
    run_extraction,
    save_extraction,
)
from persona_lib.training import finetune
from persona_lib.screening import rank_by_projection, score_samples
from persona_lib.traits import list_traits, load_trait

__all__ = [
    "ActivationModel",
    "ChatPrompt",
    "ExtractionResult",
    "finetune",
    "get_cached_rollouts",
    "list_traits",
    "load_extraction",
    "load_trait",
    "rank_by_projection",
    "run_extraction",
    "save_extraction",
    "score_coherence",
    "score_samples",
    "score_trait",
    "steering",
]
