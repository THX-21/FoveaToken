"""Special-token helpers for fixed fovea retrieval."""

FOVEA_TOKEN = "<fovea>"


def add_fovea_tokens(tokenizer) -> int:
    """Add the fixed fovea trigger token if it is missing."""

    return tokenizer.add_special_tokens({"additional_special_tokens": [FOVEA_TOKEN]})


def sync_fovea_token_ids(config, tokenizer) -> None:
    """Mirror fovea token ids onto the model config."""

    config.fovea_token_id = tokenizer.convert_tokens_to_ids(FOVEA_TOKEN)


__all__ = [
    "FOVEA_TOKEN",
    "add_fovea_tokens",
    "sync_fovea_token_ids",
]
