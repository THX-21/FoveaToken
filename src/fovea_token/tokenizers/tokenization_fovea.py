"""Token helpers for fixed fovea retrieval."""

FOVEA_TOKEN = "<fovea>"


def _ensure_fovea_token_in_vocab(tokenizer, model=None):
    """Add <fovea> to the tokenizer vocabulary if it is not already present
    and resize the model's embedding and output heads accordingly."""

    token_id = tokenizer.convert_tokens_to_ids(FOVEA_TOKEN)
    if token_id != tokenizer.unk_token_id:
        return token_id

    num_added = tokenizer.add_tokens([FOVEA_TOKEN], special_tokens=True)
    token_id = tokenizer.convert_tokens_to_ids(FOVEA_TOKEN)
    if model is not None and num_added > 0:
        old_vocab = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        if model.lm_head.weight.shape[0] != len(tokenizer):
            new_vocab = len(tokenizer)
            old_dim = model.lm_head.weight.shape[1]
            import torch
            old_lm_head = model.lm_head.weight.data
            model.lm_head = torch.nn.Linear(old_dim, new_vocab, bias=False)
            model.lm_head.weight.data[:old_vocab] = old_lm_head
            if model.lm_head.weight.device != old_lm_head.device:
                model.lm_head = model.lm_head.to(old_lm_head.device)
        if model.fovea_aux_lm_head.weight.shape[0] != len(tokenizer):
            new_vocab = len(tokenizer)
            old_dim = model.fovea_aux_lm_head.weight.shape[1]
            import torch
            old_aux = model.fovea_aux_lm_head.weight.data
            model.fovea_aux_lm_head = torch.nn.Linear(old_dim, new_vocab, bias=False)
            model.fovea_aux_lm_head.weight.data[:old_vocab] = old_aux
            if model.fovea_aux_lm_head.weight.device != old_aux.device:
                model.fovea_aux_lm_head = model.fovea_aux_lm_head.to(old_aux.device)
        repair_rows = getattr(model, "_repair_fovea_token_rows", None)
        if callable(repair_rows):
            repair_rows(token_id=token_id, force=True)
    return token_id


def sync_fovea_token_ids(config, tokenizer, model=None) -> None:
    """Mirror fovea token ids onto the model config."""

    token_id = _ensure_fovea_token_in_vocab(tokenizer, model)
    config.fovea_token_id = token_id


__all__ = [
    "FOVEA_TOKEN",
    "sync_fovea_token_ids",
]
