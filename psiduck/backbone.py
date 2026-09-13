"""The shared language backbone and its latent-communication primitives.

One model is loaded once and shared by every agent, so all agents live in the
same latent space. The backbone exposes high-level operations so that the
orchestration code (``pipeline.py``) never touches tensors directly and can be
unit-tested with a dependency-free fake:

* ``encode(text)``            -> LatentState      (1 forward pass, 0 generated tokens)
* ``delegate_message(ctx, role)`` -> LatentState  (1 forward pass, 0 generated tokens)
* ``decode_answer(states, n)``    -> str           (1 short generation)
* ``baseline_generate(prompt, n)`` -> str          (1 long English generation)

The crucial idea: inter-agent messages are *latent vectors* (pooled hidden
states) injected into the next stage as "soft tokens" via ``inputs_embeds``.
No English is decoded between agents, only for the final integer answer.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

from .metrics import ComputeMeter


class LatentState:
    """An opaque carrier for latent messages exchanged between agents.

    For the real model this wraps a torch tensor of shape ``(num_soft_tokens,
    hidden)``. The orchestration layer treats it as opaque.
    """

    __slots__ = ("data", "num_tokens")

    def __init__(self, data, num_tokens: int) -> None:
        self.data = data
        self.num_tokens = num_tokens


@runtime_checkable
class Backbone(Protocol):
    meter: ComputeMeter

    @property
    def name(self) -> str: ...

    def encode(self, text: str) -> LatentState: ...

    def delegate_message(self, context: LatentState, role: str) -> LatentState: ...

    def combine(self, states: Sequence[LatentState]) -> LatentState: ...

    def decode_answer(self, context: LatentState, max_new_tokens: int) -> str: ...

    def baseline_generate(self, prompt: str, max_new_tokens: int) -> str: ...


class SharedModel:
    """A single Hugging Face causal LM shared by all agents (one latent space).

    torch/transformers are imported lazily so that importing this module (and
    running the fast unit tests with a fake backbone) needs neither installed.
    """

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.0,
        seed: int = 0,
        torch_threads: int = 0,
    ) -> None:
        from .hf_env import ensure_hf_home

        ensure_hf_home()

        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._model_name = model_name
        self._temperature = temperature
        self.meter = ComputeMeter()

        if torch_threads and torch_threads > 0:
            torch.set_num_threads(torch_threads)
        torch.manual_seed(seed)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self._model.eval()
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._embed = self._model.get_input_embeddings()
        self._hidden = self._model.config.hidden_size

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def hidden_size(self) -> int:
        return self._hidden

    # --- low-level helpers -------------------------------------------------

    def _text_embeds(self, text: str):
        torch = self._torch
        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            emb = self._embed(ids)  # (1, seq, d)
        return emb

    def _forward_last_hidden(self, inputs_embeds):
        """Run one forward pass over soft tokens; return last-layer hidden states."""
        torch = self._torch
        attn = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=self._device
        )
        with torch.no_grad():
            out = self._model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                output_hidden_states=True,
            )
        self.meter.add_forward(processed=inputs_embeds.shape[1])
        return out.hidden_states[-1]  # (1, seq, d)

    @staticmethod
    def _pool(hidden):
        return hidden.mean(dim=1, keepdim=True)  # (1, 1, d)

    # --- latent-communication API -----------------------------------------

    def encode(self, text: str) -> LatentState:
        emb = self._text_embeds(text)
        hidden = self._forward_last_hidden(emb)
        return LatentState(self._pool(hidden), num_tokens=1)

    def delegate_message(self, context: LatentState, role: str) -> LatentState:
        torch = self._torch
        role_emb = self._text_embeds(role)  # (1, r, d)
        inputs = torch.cat([role_emb, context.data], dim=1)  # role + context soft token
        hidden = self._forward_last_hidden(inputs)
        return LatentState(self._pool(hidden), num_tokens=1)

    def combine(self, states: Sequence[LatentState]) -> LatentState:
        torch = self._torch
        data = torch.cat([s.data for s in states], dim=1)  # (1, k, d)
        return LatentState(data, num_tokens=data.shape[1])

    def decode_answer(self, context: LatentState, max_new_tokens: int) -> str:
        torch = self._torch
        # A tiny English instruction prefix, then the latent messages as soft tokens.
        instr = self._text_embeds("The final integer answer (0-999) is: ")
        inputs = torch.cat([instr, context.data], dim=1)
        attn = torch.ones(inputs.shape[:2], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self._model.generate(
                inputs_embeds=inputs,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=self._temperature > 0.0,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        generated = out[0]  # generate(inputs_embeds=...) returns only new tokens
        self.meter.add_generation(prompt_len=inputs.shape[1], generated=len(generated))
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def baseline_generate(self, prompt: str, max_new_tokens: int) -> str:
        torch = self._torch
        messages = [
            {
                "role": "system",
                "content": "You are an expert competition mathematician. Solve the "
                "AIME problem. Reason step by step, then end with \\boxed{N} where N "
                "is an integer from 0 to 999.",
            },
            {"role": "user", "content": prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        ids = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=self._temperature > 0.0,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][ids["input_ids"].shape[1]:]
        self.meter.add_generation(
            prompt_len=ids["input_ids"].shape[1], generated=len(new_tokens)
        )
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def latent_vector(self, text: str) -> List[float]:
        """Expose a pooled hidden-state vector (for the shared-latent demo)."""
        state = self.encode(text)
        return state.data.squeeze(0).squeeze(0).tolist()
