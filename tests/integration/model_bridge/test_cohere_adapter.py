"""Integration tests for the Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~2M params, CPU-safe, ungated
  - tie_word_embeddings=True
  - logit_scale=0.125 (canonical Command-R is 0.0625; the tiny checkpoint deliberately
    differs so a silent fallback to the adapter default is detectable here)
  - use_qk_norm=False (a second, from-config fixture below turns it on)

Why this file looks the way it does
-----------------------------------
The June 2026 Gemma-3 miss (#1121) shipped through a parity test that compared logits on
one short prompt at 1e-4. The eps-placement error sat just under threshold on that prompt
and over it on a longer one. This file is written so that class of bug cannot pass:

  1. Parity is asserted on tokenized prompts, including one long enough for per-position
     error to accumulate, not on four hand-typed token IDs.
  2. Intermediate activations (layer-0 resid_pre, layer-0 attention pattern, final
     residual) are compared against the HF reference at a tighter tolerance than the
     logits, so compensating errors cannot wash out at the output.
  3. Parity is asserted on BOTH numerics paths: raw boot and enable_compatibility_mode().
     The processed path is where fold_ln / logit_scale folding / centering can diverge.
  4. RoPE is checked directly: bridge hook_rot_q vs Cohere's own apply_rotary_pos_emb
     (adjacent-pair rotate_half + repeat_interleave'd cos/sin), at positions >= 1. A
     Llama-style half-split rotation is asserted to be distinguishable on this
     checkpoint so the check is not vacuous.
  5. Batched left-padded input with an attention mask, and greedy generation through
     the KV cache, are compared to HF.
  6. A tautology guard proves the bridge's own attention path runs, so parity tests
     are not HF-vs-HF.

All fixtures are fp32. The bridge loads its HF model with attn_implementation="eager" by
default (sources/transformers.py); the independent HF reference is loaded eager explicitly.
"""

from __future__ import annotations

import platform
from typing import Any, Callable

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.cohere import modeling_cohere

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"

# Policy threshold (acceptance_test_notes.md): float32 max-abs logit delta <= 1e-4.
LOGIT_TOL = 1e-4
# Intermediate activations sit before the unembed's amplification, so hold them tighter.
ACT_TOL = 1e-5
# Wider fp32 op-order noise floor on GH Actions macOS-arm64 (see test_bridge_vs_hf_eager_parity).
_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"
if _MACOS_ARM64:
    LOGIT_TOL, ACT_TOL = 1e-2, 1e-3

SHORT_PROMPT = "Hello, world!"
# Long enough that a per-position error which is sub-threshold at 4 tokens accumulates.
LONG_PROMPT = (
    "The Gemma-3 adapter shipped in June with a norm-epsilon placement bug that stayed "
    "under the acceptance threshold on a short prompt and only showed up on a longer one. "
    "This sentence exists so that the Cohere adapter is measured on a prompt long enough "
    "to expose the same class of accumulating per-position error before release, not after."
)
PROMPTS = [SHORT_PROMPT, LONG_PROMPT]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def hf_eager() -> Any:
    """Independent HF reference, fp32 + eager, never touched by the bridge."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def bridge_raw() -> TransformerBridge:
    """Raw boot: HF-faithful numerics, no weight processing."""
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def bridge_compat() -> TransformerBridge:
    """Separate boot with full compatibility-mode processing (fold_ln, centering, folds).

    Compatibility mode mutates weights in place and is one-shot, so it gets its own
    instance rather than sharing bridge_raw.
    """
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge


@pytest.fixture(scope="module")
def bridge_fold_only() -> TransformerBridge:
    """process_weights with every ProcessWeights option off: only the adapter's
    preprocess_weights (logit_scale fold + embed/unembed untie) runs."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture
def tokenize(tokenizer) -> Callable[[str], torch.Tensor]:
    def _tok(prompt: str) -> torch.Tensor:
        return tokenizer(prompt, return_tensors="pt").input_ids

    return _tok


def _capture_hf(model: Any, module: torch.nn.Module, tokens: torch.Tensor, *, pre: bool):
    """Run HF once and return the captured input (pre=True) or output of ``module``."""
    box: dict[str, torch.Tensor] = {}

    def _pre(_m, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        box["v"] = hs.detach()

    def _post(_m, _i, out):
        box["v"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = (
        module.register_forward_pre_hook(_pre, with_kwargs=True)
        if pre
        else module.register_forward_hook(_post)
    )
    try:
        with torch.inference_mode():
            model(tokens)
    finally:
        handle.remove()
    return box["v"]


def _capture_bridge(bridge: TransformerBridge, hook_name: str, tokens: torch.Tensor):
    box: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        bridge.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, lambda v, hook: box.__setitem__("v", v.detach()))]
        )
    assert "v" in box, f"{hook_name} did not fire"
    return box["v"]


# ---------------------------------------------------------------------------
# 1. Bridge creation and config propagation
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    def test_boot_and_structure(self, bridge_raw: TransformerBridge) -> None:
        assert len(bridge_raw.blocks) == 2
        for attr in ("embed", "unembed", "ln_final", "rotary_emb"):
            assert hasattr(bridge_raw, attr), attr
        for block in bridge_raw.blocks:
            assert not hasattr(block, "ln2"), "parallel block must not have ln2"

    def test_cfg_flags(self, bridge_raw: TransformerBridge) -> None:
        cfg = bridge_raw.cfg
        assert cfg.normalization_type == "LN"
        assert cfg.uses_rms_norm is False
        assert cfg.parallel_attn_mlp is True
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is True

    def test_logit_scale_propagates_and_is_not_the_fallback(
        self, bridge_raw: TransformerBridge, hf_eager: Any
    ) -> None:
        """cfg.logit_scale must come from HF config, not the adapter's 0.0625 default.

        The tiny checkpoint uses 0.125 precisely so this is distinguishable. On real
        Command-R the fallback equals the true value and a passthrough regression would
        be invisible, so this assertion is the only place it can be caught.
        """
        scale = getattr(bridge_raw.cfg, "logit_scale")
        assert scale == pytest.approx(hf_eager.config.logit_scale)
        assert hf_eager.config.logit_scale != 0.0625, (
            "test checkpoint's logit_scale equals the adapter fallback; "
            "the passthrough regression check is vacuous on this model"
        )

    def test_rope_parameters_propagate(self, bridge_raw: TransformerBridge, hf_eager: Any):
        assert getattr(bridge_raw.cfg, "rope_parameters") == hf_eager.config.rope_parameters

    def test_to_tokens_matches_hf_tokenizer(
        self, bridge_raw: TransformerBridge, tokenizer
    ) -> None:
        """default_prepend_bos=True must reproduce HF's add_special_tokens behaviour."""
        for prompt in PROMPTS:
            bridge_ids = bridge_raw.to_tokens(prompt)
            hf_ids = tokenizer(prompt, return_tensors="pt").input_ids
            assert torch.equal(bridge_ids, hf_ids), (
                f"to_tokens diverges from HF tokenizer on {prompt[:30]!r}: "
                f"{bridge_ids[0, :5].tolist()} vs {hf_ids[0, :5].tolist()}"
            )


# ---------------------------------------------------------------------------
# 2. Tautology guard — the bridge's own attention path must actually run
# ---------------------------------------------------------------------------


class TestCohereNotTautological:
    def test_bridge_attention_reconstruction_runs(
        self, bridge_raw: TransformerBridge, tokenize
    ) -> None:
        fired: list[bool] = []
        bridge_raw.run_with_hooks(
            tokenize(SHORT_PROMPT),
            fwd_hooks=[("blocks.0.attn.hook_attn_scores", lambda v, hook: fired.append(True))],
        )
        assert fired, (
            "blocks.0.attn.hook_attn_scores did not fire; the bridge is delegating "
            "attention to HF and every parity test below is HF-vs-HF."
        )

    def test_hf_delegation_targets(self, bridge_raw: TransformerBridge) -> None:
        ln1 = bridge_raw.blocks[0].ln1
        attn = bridge_raw.blocks[0].attn
        assert isinstance(ln1, NormalizationBridge)
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert hasattr(ln1.original_component, "variance_epsilon")  # CohereLayerNorm
        assert hasattr(attn.original_component, "q_proj")  # CohereAttention
        assert hasattr(bridge_raw.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 3. Forward parity, raw path, short + long tokenized prompts
# ---------------------------------------------------------------------------


class TestCohereForwardParityRaw:
    @pytest.mark.parametrize("prompt", PROMPTS, ids=["short", "long"])
    def test_logits_match_hf_eager(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize, prompt: str
    ) -> None:
        tokens = tokenize(prompt)
        assert tokens.shape[1] >= 2
        with torch.inference_mode():
            bridge_logits = bridge_raw(tokens)
            hf_logits = hf_eager(tokens).logits
        assert bridge_logits.shape == hf_logits.shape
        assert torch.isfinite(bridge_logits).all()
        diff = (bridge_logits - hf_logits).abs()
        max_diff = diff.max().item()
        worst_pos = diff.amax(dim=-1)[0].argmax().item()
        assert max_diff < LOGIT_TOL, (
            f"raw bridge vs HF eager max diff {max_diff:.2e} > {LOGIT_TOL:.0e} on "
            f"{'short' if prompt is SHORT_PROMPT else 'long'} prompt "
            f"({tokens.shape[1]} tokens); worst position {worst_pos}"
        )

    def test_error_does_not_grow_with_position(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        """The Gemma-3 signature: sub-threshold early, over threshold late.

        Compare the per-position max error in the last quarter of the long prompt to
        the first quarter. fp32 noise is flat in position; an accumulating conversion
        error is not.
        """
        tokens = tokenize(LONG_PROMPT)
        with torch.inference_mode():
            diff = (bridge_raw(tokens) - hf_eager(tokens).logits).abs().amax(dim=-1)[0]
        n = diff.shape[0]
        q = max(n // 4, 2)
        head, tail = diff[:q].max().item(), diff[-q:].max().item()
        assert tail < LOGIT_TOL, f"late-position error {tail:.2e} exceeds {LOGIT_TOL:.0e}"
        assert tail <= max(10 * head, ACT_TOL), (
            f"per-position error grows along the sequence (first-quarter max {head:.2e}, "
            f"last-quarter max {tail:.2e}); this is the accumulating-error signature."
        )


# ---------------------------------------------------------------------------
# 4. Intermediate activations vs HF, tighter tolerance
# ---------------------------------------------------------------------------


class TestCohereIntermediateActivations:
    """The methodology's secondary check: resid_pre@0, attn pattern@0, final resid."""

    def test_layer0_resid_pre(self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize):
        tokens = tokenize(LONG_PROMPT)
        hf_v = _capture_hf(hf_eager, hf_eager.model.layers[0], tokens, pre=True)
        br_v = _capture_bridge(bridge_raw, "blocks.0.hook_resid_pre", tokens)
        d = (hf_v - br_v).abs().max().item()
        assert d < ACT_TOL, f"blocks.0.hook_resid_pre drift {d:.2e} (embedding path)"

    def test_layer0_attention_pattern(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        """Post-softmax pattern at layer 0. A RoPE-convention error shows up here first."""
        tokens = tokenize(LONG_PROMPT)
        with torch.inference_mode():
            hf_attn = hf_eager(tokens, output_attentions=True).attentions[0]  # [B,H,S,S]
        br_attn = _capture_bridge(bridge_raw, "blocks.0.attn.hook_pattern", tokens)
        assert br_attn.shape == hf_attn.shape, (br_attn.shape, hf_attn.shape)
        d = (hf_attn - br_attn).abs().max().item()
        assert d < ACT_TOL, f"blocks.0.attn.hook_pattern drift {d:.2e}; check RoPE / QK path"

    @pytest.mark.parametrize("layer", [0, 1])
    def test_resid_post_every_layer(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize, layer: int
    ) -> None:
        tokens = tokenize(LONG_PROMPT)
        hf_v = _capture_hf(hf_eager, hf_eager.model.layers[layer], tokens, pre=False)
        br_v = _capture_bridge(bridge_raw, f"blocks.{layer}.hook_resid_post", tokens)
        d = (hf_v - br_v).abs().max().item()
        assert d < ACT_TOL, f"blocks.{layer}.hook_resid_post drift {d:.2e}"

    def test_final_norm_input_and_output(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        """ln_final in and out: catches eps-placement and mean-subtraction errors directly."""
        tokens = tokenize(LONG_PROMPT)
        hf_in = _capture_hf(hf_eager, hf_eager.model.norm, tokens, pre=True)
        hf_out = _capture_hf(hf_eager, hf_eager.model.norm, tokens, pre=False)
        br_in = _capture_bridge(bridge_raw, "ln_final.hook_in", tokens)
        br_out = _capture_bridge(bridge_raw, "ln_final.hook_out", tokens)
        d_in = (hf_in - br_in).abs().max().item()
        d_out = (hf_out - br_out).abs().max().item()
        assert d_in < ACT_TOL, f"ln_final input drift {d_in:.2e}"
        assert d_out < ACT_TOL, f"ln_final output drift {d_out:.2e} (eps / mean-subtraction)"


# ---------------------------------------------------------------------------
# 5. RoPE convention, checked directly against Cohere's own rotation
# ---------------------------------------------------------------------------


def _llama_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TestCohereRoPEConvention:
    def test_hook_rot_q_matches_cohere_apply_rotary(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        """bridge hook_rot_q must equal Cohere's adjacent-pair rotation, not Llama's.

        Reference is computed from the independent HF model: capture layer-0 attention
        input + position_embeddings, project with q_proj, apply Cohere's own
        apply_rotary_pos_emb. Then assert (a) bridge == Cohere and (b) Cohere != Llama
        on this checkpoint, so the test cannot pass by accident.
        """
        tokens = tokenize(LONG_PROMPT)
        attn = hf_eager.model.layers[0].self_attn
        cfg = hf_eager.config
        box: dict[str, Any] = {}

        def _pre(_m, args, kwargs):
            box["hs"] = kwargs.get("hidden_states", args[0] if args else None).detach()
            box["pe"] = kwargs["position_embeddings"]

        h = attn.register_forward_pre_hook(_pre, with_kwargs=True)
        try:
            with torch.inference_mode():
                hf_eager(tokens)
        finally:
            h.remove()

        hs, (cos, sin) = box["hs"], box["pe"]
        bsz, seq, _ = hs.shape
        head_dim = attn.head_dim
        with torch.inference_mode():
            q = attn.q_proj(hs).view(bsz, seq, -1, head_dim)
            if getattr(cfg, "use_qk_norm", False):
                q = attn.q_norm(q)
            q = q.transpose(1, 2)  # [B,H,S,D]
            q_cohere, _ = modeling_cohere.apply_rotary_pos_emb(q, q, cos, sin)
            q_llama = (q * cos.unsqueeze(1)) + (_llama_rotate_half(q) * sin.unsqueeze(1))

        br_q = _capture_bridge(bridge_raw, "blocks.0.attn.hook_rot_q", tokens)
        if br_q.shape != q_cohere.shape and br_q.transpose(1, 2).shape == q_cohere.shape:
            br_q = br_q.transpose(1, 2)
        assert br_q.shape == q_cohere.shape, (br_q.shape, q_cohere.shape)

        # (b) discriminability: the two conventions must differ at positions >= 1.
        convention_gap = (q_cohere[:, :, 1:] - q_llama[:, :, 1:]).abs().max().item()
        assert convention_gap > 100 * ACT_TOL, (
            f"Cohere and Llama rotations agree to {convention_gap:.2e} on this checkpoint; "
            "the RoPE check cannot discriminate here"
        )

        # (a) the bridge must match Cohere's rotation, including at positions >= 1.
        d_pos0 = (br_q[:, :, :1] - q_cohere[:, :, :1]).abs().max().item()
        d_rest = (br_q[:, :, 1:] - q_cohere[:, :, 1:]).abs().max().item()
        assert d_pos0 < ACT_TOL, f"hook_rot_q drift at position 0: {d_pos0:.2e}"
        assert d_rest < ACT_TOL, (
            f"hook_rot_q drift at positions >= 1: {d_rest:.2e}. Position 0 matched "
            f"({d_pos0:.2e}), which is the signature of a rotate_half convention mismatch "
            "(half-split vs adjacent-pair)."
        )


# ---------------------------------------------------------------------------
# 6. Compatibility mode (processed weights) parity
# ---------------------------------------------------------------------------


class TestCohereCompatibilityModeParity:
    """fold_ln + center_writing_weights + center_unembed + fold_value_biases, vs HF.

    center_unembed shifts every logit vector by a per-position constant, so compare
    log-probabilities, which are invariant to that shift, at the policy tolerance.
    """

    @pytest.mark.parametrize("prompt", PROMPTS, ids=["short", "long"])
    def test_log_probs_match_hf(
        self, bridge_compat: TransformerBridge, hf_eager: Any, tokenize, prompt: str
    ) -> None:
        tokens = tokenize(prompt)
        with torch.inference_mode():
            br_lp = torch.log_softmax(bridge_compat(tokens).float(), dim=-1)
            hf_lp = torch.log_softmax(hf_eager(tokens).logits.float(), dim=-1)
        d = (br_lp - hf_lp).abs().max().item()
        assert d < LOGIT_TOL, (
            f"compatibility-mode log-prob drift {d:.2e} > {LOGIT_TOL:.0e} on "
            f"{tokens.shape[1]}-token prompt; processed path (fold_ln / logit_scale fold / "
            "centering) diverges from HF"
        )

    def test_logit_scale_applied_exactly_once(
        self, bridge_compat: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        """Ratio test on raw (uncentered-per-row) logits: a double or missing fold shows
        as a global factor of logit_scale or 1/logit_scale on the centered logits."""
        tokens = tokenize(SHORT_PROMPT)
        with torch.inference_mode():
            br = bridge_compat(tokens).float()
            hf = hf_eager(tokens).logits.float()
        # Remove the per-position centering constant before comparing scale.
        br_c = br - br.mean(dim=-1, keepdim=True)
        hf_c = hf - hf.mean(dim=-1, keepdim=True)
        ratio = (br_c.norm() / hf_c.norm()).item()
        assert abs(ratio - 1.0) < 1e-3, (
            f"centered-logit norm ratio bridge/HF = {ratio:.4f}; a value near "
            f"{hf_eager.config.logit_scale} or {1 / hf_eager.config.logit_scale:.1f} means "
            "logit_scale was folded twice or not at all"
        )



class TestCohereLogitScaleFold:
    """Fold-only path: logit_scale baked into unembed, HF forward no longer scales.

    This is the processed-path check the original test lacked: after the fold the
    bridge's forward must still match HF logits exactly. A double application shows
    as a factor of logit_scale; a dropped fold shows as 1/logit_scale.
    """

    def test_unembed_scaled_embed_untouched(
        self, bridge_fold_only: TransformerBridge, hf_eager: Any
    ) -> None:
        scale = hf_eager.config.logit_scale
        tl_embed = bridge_fold_only.embed.W_E.detach()
        hf_embed = hf_eager.model.embed_tokens.weight.detach()
        assert (tl_embed - hf_embed).abs().max().item() < 1e-6, "embed.W_E was scaled"
        tl_unembed = bridge_fold_only.unembed.original_component.weight.detach()
        expected = hf_eager.lm_head.weight.detach() * scale
        assert (tl_unembed - expected).abs().max().item() < 1e-5, "unembed not scaled once"
        assert not torch.allclose(tl_embed, tl_unembed), "embed/unembed still tied after fold"

    @pytest.mark.parametrize("prompt", PROMPTS, ids=["short", "long"])
    def test_forward_after_fold_matches_hf(
        self, bridge_fold_only: TransformerBridge, hf_eager: Any, tokenize, prompt: str
    ) -> None:
        tokens = tokenize(prompt)
        with torch.inference_mode():
            br = bridge_fold_only(tokens).float()
            hf = hf_eager(tokens).logits.float()
        d = (br - hf).abs().max().item()
        scale = hf_eager.config.logit_scale
        ratio = (br.norm() / hf.norm()).item()
        assert d < LOGIT_TOL, (
            f"fold-only bridge vs HF max diff {d:.2e} (norm ratio {ratio:.3f}); a ratio near "
            f"{scale} means logit_scale was applied twice, near {1 / scale:.1f} means the "
            "fold was dropped"
        )


# ---------------------------------------------------------------------------
# 7. Batched, left-padded input with an attention mask
# ---------------------------------------------------------------------------


class TestCohereBatchedPadding:
    def test_left_padded_batch_matches_hf_and_unbatched(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenizer
    ) -> None:
        tok = tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        try:
            batch = tok([SHORT_PROMPT, LONG_PROMPT], return_tensors="pt", padding=True)
        finally:
            tok.padding_side = prev_side
        ids, mask = batch.input_ids, batch.attention_mask

        with torch.inference_mode():
            hf_b = hf_eager(input_ids=ids, attention_mask=mask).logits
            br_b = bridge_raw(ids, attention_mask=mask)
            hf_long = hf_eager(tok(LONG_PROMPT, return_tensors="pt").input_ids).logits[0]

        valid = mask.bool()
        d_batch = (hf_b[valid] - br_b[valid]).abs().max().item()
        assert d_batch < LOGIT_TOL, (
            f"batched+masked logits drift {d_batch:.2e}; attention mask not honoured the way "
            "HF honours it"
        )

        # Row 1 is the long prompt and carries no padding, so its logits must equal the
        # unbatched run: batching itself must not perturb an unpadded sequence.
        assert bool(mask[1].all())
        d_row = (br_b[1] - hf_long).abs().max().item()
        assert d_row < LOGIT_TOL, f"unpadded row in a batch diverges from its unbatched run by {d_row:.2e}"


# ---------------------------------------------------------------------------
# 8. Greedy generation through the KV cache
# ---------------------------------------------------------------------------


class TestCohereKVCacheGeneration:
    def test_greedy_generate_matches_hf(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        tokens = tokenize(SHORT_PROMPT)
        n_new = 12
        with torch.inference_mode():
            hf_out = hf_eager.generate(
                tokens, max_new_tokens=n_new, do_sample=False, use_cache=True,
                pad_token_id=hf_eager.config.eos_token_id,
            )
            br_out = bridge_raw.generate(
                tokens, max_new_tokens=n_new, do_sample=False, use_past_kv_cache=True,
                stop_at_eos=False, verbose=False,
            )
        hf_new = hf_out[0, tokens.shape[1]:]
        br_new = br_out[0, tokens.shape[1]:]
        assert torch.equal(hf_new[: len(br_new)], br_new[: len(hf_new)]), (
            f"greedy continuation diverges: HF {hf_new.tolist()} vs bridge {br_new.tolist()}"
        )

    def test_cached_and_uncached_logits_agree(
        self, bridge_raw: TransformerBridge, hf_eager: Any, tokenize
    ) -> None:
        """Incremental decoding must reproduce full-sequence logits at the new position."""
        tokens = tokenize(LONG_PROMPT)
        prefix, last = tokens[:, :-1], tokens[:, -1:]
        with torch.inference_mode():
            full = hf_eager(tokens).logits[:, -1]
            out = hf_eager(prefix, use_cache=True)
            step = hf_eager(last, past_key_values=out.past_key_values, use_cache=True).logits[:, -1]
            br_full = bridge_raw(tokens)[:, -1]
        assert (full - step).abs().max().item() < LOGIT_TOL, "HF cache sanity failed"
        assert (br_full - full).abs().max().item() < LOGIT_TOL


# ---------------------------------------------------------------------------
# 9. QK-norm path, exercised via a from-config tiny model with use_qk_norm=True
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qk_norm_model_dir(tmp_path_factory, tokenizer):
    """Seeded random-init CohereForCausalLM with use_qk_norm=True, saved to disk.

    Real Command-R / Command-R+ checkpoints enable QK-norm; the public tiny checkpoint
    does not, so the per-head LayerNorm path would otherwise be untested.
    """
    from transformers import CohereConfig, CohereForCausalLM

    cfg = CohereConfig.from_pretrained(MODEL)
    cfg.use_qk_norm = True
    torch.manual_seed(1234)
    model = CohereForCausalLM(cfg).eval()
    out = tmp_path_factory.mktemp("cohere_qknorm")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return str(out)


class TestCohereQKNorm:
    def test_qk_norm_logits_and_pattern_match_hf(self, qk_norm_model_dir: str, tokenize):
        hf = AutoModelForCausalLM.from_pretrained(
            qk_norm_model_dir, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        assert hf.config.use_qk_norm is True
        assert hasattr(hf.model.layers[0].self_attn, "q_norm")
        bridge = TransformerBridge.boot_transformers(
            qk_norm_model_dir, device="cpu", dtype=torch.float32
        )
        tokens = tokenize(LONG_PROMPT)
        with torch.inference_mode():
            hf_out = hf(tokens, output_attentions=True)
            br_logits = bridge(tokens)
        d_logits = (br_logits - hf_out.logits).abs().max().item()
        assert d_logits < LOGIT_TOL, f"use_qk_norm=True logits drift {d_logits:.2e}"
        br_attn = _capture_bridge(bridge, "blocks.0.attn.hook_pattern", tokens)
        d_attn = (br_attn - hf_out.attentions[0]).abs().max().item()
        assert d_attn < ACT_TOL, f"use_qk_norm=True layer-0 pattern drift {d_attn:.2e}"


# ---------------------------------------------------------------------------
# 10. Hook surface for the parallel block
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    def test_hooks_present_and_resid_mid_absent(self, bridge_raw: TransformerBridge, tokenize):
        _, cache = bridge_raw.run_with_cache(tokenize(SHORT_PROMPT))
        assert not any("hook_resid_mid" in k for k in cache.keys())
        for i in range(2):
            for name in (
                "hook_resid_pre", "hook_resid_post",
                "attn.hook_in", "attn.hook_out", "mlp.hook_in", "mlp.hook_out",
                "attn.hook_pattern", "attn.hook_rot_q",
            ):
                assert f"blocks.{i}.{name}" in cache, f"blocks.{i}.{name} missing"

    def test_attn_and_mlp_share_ln1_output(self, bridge_raw: TransformerBridge, tokenize):
        """Parallel block: attn.hook_in and mlp.hook_in must be the same tensor values."""
        _, cache = bridge_raw.run_with_cache(tokenize(SHORT_PROMPT))
        d = (cache["blocks.0.attn.hook_in"] - cache["blocks.0.mlp.hook_in"]).abs().max().item()
        assert d == 0.0, f"attn and mlp inputs differ by {d:.2e}; block is not parallel"
