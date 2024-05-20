"""Utilities to get activations and replace them."""

import random
from sae_lens import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import download_sae_from_hf
from sae_lens.training.load_model import load_model
from transformer_lens import HookedTransformer
import torch
from typing import Any, Optional, TypedDict
from functools import partial
from datasets import Dataset
import os


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations


def load_sae_and_model(
    sae_hf_repo: str = "jbloom/GPT2-Small-SAEs-Reformatted",
    sae_name: str = "blocks.6.hook_resid_pre",
    device: str = "cuda",
) -> tuple[SparseAutoencoder, HookedTransformer]:
    (_, SAE_WEIGHTS_PATH, _) = download_sae_from_hf(sae_hf_repo, sae_name)

    SAE_PATH = os.path.dirname(SAE_WEIGHTS_PATH)

    sae = SparseAutoencoder.load_from_pretrained(SAE_PATH, device=device)
    sae.eval()

    model = load_model(
        sae.cfg.model_class_name,
        sae.cfg.model_name,
        device=sae.cfg.device,
        model_from_pretrained_kwargs=sae.cfg.model_from_pretrained_kwargs,
    )
    model.eval()

    return sae, model


@torch.no_grad()
def get_sae_activations(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    strs: list[str],
) -> tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[float]]]]:
    """Returns the SAE activations for each sequence, for each token in the sequence (including the BOS token).

    Returns pre encoding act, post encoding acts, and post decoding acts, all of shape [seq_id][seq_pos][dim]"""

    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        raise NotImplementedError("head_index not supported yet")

    hook_point = sparse_autoencoder.cfg.hook_point

    pre_acts = []
    inner_acts = []
    post_acts = []

    for s in strs:
        eval_tokens = model.to_tokens(
            [s],
            truncate=True,
            move_to_device=True,
            prepend_bos=sparse_autoencoder.cfg.prepend_bos,
        )

        # get cache
        _, cache = model.run_with_cache(
            eval_tokens,
            prepend_bos=False,
            names_filter=[hook_point],
            **sparse_autoencoder.cfg.model_kwargs,
        )
        pre_act: torch.Tensor = cache[hook_point]
        inner_act = sparse_autoencoder.encode(pre_act)
        post_act = sparse_autoencoder.decode(inner_act)

        pre_acts.extend(pre_act.tolist())
        inner_acts.extend(inner_act.tolist())
        post_acts.extend(post_act.tolist())

    return pre_acts, inner_acts, post_acts


@torch.no_grad()
def get_recons_loss_from_predicted_values(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    strs: list[str],
    # [seq_id][seq_pos][feature_id], naive predictions *after* seing the token, **INCLUDING BOS TOKEN**
    activation_predictions: list[list[list[float]]],
) -> list[list[float]]:
    """Returns the loss for each sequence, for each token in the sequence (excluding the BOS token)."""

    if not sparse_autoencoder.cfg.prepend_bos:
        raise NotImplementedError("SAEs with prepend_bos=False not supported yet")

    # running sequences one by one since it's so cheap and simple
    # improve if that becomes a bottleneck
    # (note: the sae_lens code below doesn't support padding by default, be careful with it)

    losses = []

    for s, relevant_act_predictions in zip(strs, activation_predictions, strict=True):
        tokens = model.to_tokens(
            [s],
            truncate=True,
            move_to_device=True,
            prepend_bos=sparse_autoencoder.cfg.prepend_bos,
        )

        assert len(relevant_act_predictions) == len(tokens[0]), (
            "Incorrect length. "
            "Did you forget to include the prediction for the BOS token? "
            f"{len(relevant_act_predictions)} != {len(tokens[0])}"
        )

        precomputed_features = torch.tensor(
            relevant_act_predictions, device=sparse_autoencoder.device, dtype=sparse_autoencoder.dtype
        ).unsqueeze(0)

        expected_shape = (1, len(tokens[0]), len(relevant_act_predictions[0]))
        assert precomputed_features.shape == expected_shape, f"{precomputed_features.shape} != {expected_shape}"

        loss_per_token = get_recons_loss_batch(model, sparse_autoencoder, tokens, precomputed_features)
        losses.extend(loss_per_token)

    return losses


def get_vanilla_loss(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    strs: list[str],
    # with_sae_replacement=False means loss without SAE, True means loss with activations replaced by SAE decode(encode(x))
    with_sae_replacement: bool = False,
) -> list[list[float]]:
    """Returns the loss for each sequence, for each token in the sequence (excluding the BOS token)."""

    losses = []

    for s in strs:
        tokens = model.to_tokens(
            [s],
            truncate=True,
            move_to_device=True,
            prepend_bos=sparse_autoencoder.cfg.prepend_bos,
        )

        if with_sae_replacement:
            loss_per_token = get_recons_loss_batch(model, sparse_autoencoder, tokens)
        else:
            loss_per_token = get_loss_batch(model, sparse_autoencoder, tokens)
        losses.extend(loss_per_token)

    return losses


def get_recons_loss_batch(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    batch_tokens: torch.Tensor,
    precomputed_features: Optional[torch.Tensor] = None,
) -> float:
    """compute reconstruction loss for one batch

    for CE experiments, use precomputed features, usually of shape (batch_size, seq_len, n_features)"""
    # adapted from sae_lens/training/evals.py

    hook_point = sparse_autoencoder.cfg.hook_point
    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        raise NotImplementedError("head_index not supported yet")

    def replacement_hook(activations: torch.Tensor, hook: Any):
        if precomputed_features is not None:
            features_acts = precomputed_features
        else:
            features_acts = sparse_autoencoder.encode(activations)
        return sparse_autoencoder.decode(features_acts).to(activations.dtype)

    recons_loss: torch.Tensor = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        loss_per_token=True,
        fwd_hooks=[(hook_point, partial(replacement_hook))],
        **sparse_autoencoder.cfg.model_kwargs,
    )

    return recons_loss.tolist()


def get_loss_batch(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    batch_tokens: torch.Tensor,
):
    losses: torch.Tensor = model(
        batch_tokens,
        return_type="loss",
        loss_per_token=True,
        **sparse_autoencoder.cfg.model_kwargs,
    )
    return losses.tolist()


def pretty_losses_fmt(name: str, strings: list[str], losses: list[list[float]]):
    averages = [sum(seq_losses) / len(seq_losses) for seq_losses in losses]
    return f"{name} losses:\n" + "\n".join(
        [
            f"{s} ({avg:.4f}): " + " ".join([f"{loss:.4f}" for loss in seq_losses])
            for s, avg, seq_losses in zip(strings, averages, losses)
        ]
    )


if __name__ == "__main__":
    # some sanity checks

    strings = ["Hello, world!", "What a wonderful day! I love it!"]

    sae, model = load_sae_and_model()
    pre_acts, inner_acts, post_acts = get_sae_activations(model, sae, strings)

    regular_losses = get_vanilla_loss(model, sae, strings)
    print(pretty_losses_fmt("Regular", strings, regular_losses))

    sae_losses = get_vanilla_loss(model, sae, strings, with_sae_replacement=True)
    print(pretty_losses_fmt("SAE", strings, sae_losses))

    # sae predictions
    sae_losses_from_precomputed = get_recons_loss_from_predicted_values(model, sae, strings, inner_acts)
    print(pretty_losses_fmt("SAE from precomputed", strings, sae_losses_from_precomputed))

    # check that sae_losses and sae_losses_from_precomputed are the same
    for l1, l2 in zip(sae_losses, sae_losses_from_precomputed):
        for e1, e2 in zip(l1, l2):
            assert abs(e1 - e2) < 1e-5, f"{e1} != {e2}"

    # zero ablations
    precomputed_zeros = [[[0.0] * len(l) for l in seq] for seq in inner_acts]
    zeros_losses = get_recons_loss_from_predicted_values(model, sae, strings, precomputed_zeros)
    print(pretty_losses_fmt("Zeros", strings, zeros_losses))

    # shuffle ablations
    def shuffle(l: list):
        l = l.copy()
        random.shuffle(l)
        return l

    shuffled_precomputed = [[shuffle(l) for l in seq] for seq in inner_acts]
    shuffled_losses = get_recons_loss_from_predicted_values(model, sae, strings, shuffled_precomputed)
    print(pretty_losses_fmt("Shuffled", strings, shuffled_losses))
