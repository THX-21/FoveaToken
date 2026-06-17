"""Token helpers for fixed fovea retrieval."""

FOVEA_TOKEN = "<|vision_start|>"


def sync_fovea_token_ids(config, tokenizer) -> None:
    """Mirror fovea token ids onto the model config."""

    config.fovea_token_id = tokenizer.convert_tokens_to_ids(FOVEA_TOKEN)


__all__ = [
    "FOVEA_TOKEN",
    "sync_fovea_token_ids",
]
