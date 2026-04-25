"""The `ActivationModel` — a thin, student-facing wrapper around an HF causal LM.

Responsibilities:
- Load a model + tokenizer with sensible defaults (bfloat16, device_map="auto").
- Render `ChatPrompt`s through the tokenizer's chat template.
- Generate completions.
- Collect residual-stream activations at one of two positions:
    "last_prompt"   — the last token of the rendered prompt (before generation).
    "response_mean" — mean over assistant-response tokens (requires `ChatPrompt.assistant`).

Returns NumPy arrays with shape `(n_prompts, n_layers, hidden_dim)`; students never see
a `torch.Tensor` or a tokenizer call.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_lib.data import ChatPrompt
from persona_lib.hooks import collecting_hooks, get_decoder_layers

Position = Literal["last_prompt", "response_mean"]


def cuda_info() -> dict:
    """Environment summary for the env-check cell in notebook 00.

    Keeps `import torch` out of student notebooks — they should only ever import
    from `persona_lib`.
    """
    info: dict = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


_DEFAULT_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n"
    "{% else %}<|assistant|>\n{{ m['content'] }}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


class ActivationModel:
    """Chat model with first-class activation access."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.chat_template is None:
            # Fallback template for base (non-chat) models so tests + debugging work.
            # Real instruct models ship their own template; this is never reached for Qwen/Llama chat.
            self.tokenizer.chat_template = _DEFAULT_CHAT_TEMPLATE
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        self.hf_model.eval()
        self._decoder_layers = get_decoder_layers(self.hf_model)

    @property
    def num_layers(self) -> int:
        return len(self._decoder_layers)

    @property
    def hidden_dim(self) -> int:
        return self.hf_model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.hf_model.parameters()).device

    def _render(self, prompt: ChatPrompt, *, add_generation_prompt: bool) -> str:
        messages: list[dict] = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        if prompt.assistant is not None:
            messages.append({"role": "assistant", "content": prompt.assistant})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and prompt.assistant is None,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: ChatPrompt | list[ChatPrompt],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        n: int = 1,
        batch_size: int = 8,
    ) -> list[str] | list[list[str]]:
        """Sample completions. Returns a list of strings (or list-of-lists if `n > 1`)."""
        single = isinstance(prompts, ChatPrompt)
        if single:
            prompts = [prompts]
        rendered = [self._render(p, add_generation_prompt=True) for p in prompts]

        all_texts: list[str] = []
        for start in range(0, len(rendered), batch_size):
            chunk = rendered[start : start + batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, padding_side="left").to(
                self.device
            )
            out = self.hf_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            input_len = enc["input_ids"].shape[1]
            gen_ids = out[:, input_len:]
            all_texts.extend(self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

        if n > 1:
            grouped: list[list[str]] = [all_texts[i * n : (i + 1) * n] for i in range(len(prompts))]
            return grouped[0] if single else grouped
        return all_texts

    @torch.no_grad()
    def collect_activations(
        self,
        prompts: ChatPrompt | list[ChatPrompt],
        *,
        position: Position,
        layers: list[int] | None = None,
    ) -> np.ndarray:
        """Return residual-stream activations. Shape: `(n_prompts, n_layers_selected, hidden_dim)`.

        - `position="last_prompt"`: the hidden state at the final prompt token (pre-generation).
          Ignores `prompt.assistant` if set.
        - `position="response_mean"`: the mean hidden state over the assistant-response
          tokens. Requires `prompt.assistant is not None` for every prompt.
        """
        single = isinstance(prompts, ChatPrompt)
        if single:
            prompts = [prompts]
        if layers is None:
            layers = list(range(self.num_layers))
        layers = sorted(set(layers))

        out = np.zeros((len(prompts), len(layers), self.hidden_dim), dtype=np.float32)

        for i, prompt in enumerate(prompts):
            if position == "last_prompt":
                rendered = self._render(prompt, add_generation_prompt=True)
                ids = self.tokenizer(rendered, return_tensors="pt").to(self.device)
                with collecting_hooks(self.hf_model, layers) as collected:
                    self.hf_model(**ids)
                for j, L in enumerate(layers):
                    h = collected[L][0]  # (1, seq, d)
                    out[i, j] = h[0, -1].float().numpy()

            elif position == "response_mean":
                if prompt.assistant is None:
                    raise ValueError(
                        "collect_activations(position='response_mean') requires "
                        "ChatPrompt.assistant to define the response span."
                    )
                prompt_only = ChatPrompt(user=prompt.user, system=prompt.system)
                prompt_text = self._render(prompt_only, add_generation_prompt=True)
                full_text = self._render(prompt, add_generation_prompt=False)
                prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
                full_ids = self.tokenizer(full_text, return_tensors="pt").input_ids
                prompt_len = prompt_ids.shape[1]
                if full_ids.shape[1] <= prompt_len:
                    raise ValueError("assistant turn tokenized to zero tokens")

                with collecting_hooks(self.hf_model, layers) as collected:
                    self.hf_model(input_ids=full_ids.to(self.device))
                for j, L in enumerate(layers):
                    h = collected[L][0]
                    resp = h[0, prompt_len : full_ids.shape[1]]
                    out[i, j] = resp.mean(dim=0).float().numpy()

            else:
                raise ValueError(f"unknown position {position!r}")

        return out
