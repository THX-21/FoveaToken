"""Special-token helpers for fixed fovea retrieval."""

FOVEA_TOKEN = "<fovea>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
FOVEA_SPECIAL_TOKENS = [FOVEA_TOKEN, THINK_START_TOKEN, THINK_END_TOKEN]


def add_fovea_tokens(tokenizer) -> int:
    """Add fixed fovea/reasoning marker tokens if they are missing."""

    try:
        return tokenizer.add_special_tokens(
            {"additional_special_tokens": FOVEA_SPECIAL_TOKENS},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        return tokenizer.add_special_tokens({"additional_special_tokens": FOVEA_SPECIAL_TOKENS})


def sync_fovea_token_ids(config, tokenizer) -> None:
    """Mirror fovea token ids onto the model config."""

    config.fovea_token_id = tokenizer.convert_tokens_to_ids(FOVEA_TOKEN)


__all__ = [
    "FOVEA_TOKEN",
    "THINK_START_TOKEN",
    "THINK_END_TOKEN",
    "FOVEA_SPECIAL_TOKENS",
    "add_fovea_tokens",
    "sync_fovea_token_ids",
]
