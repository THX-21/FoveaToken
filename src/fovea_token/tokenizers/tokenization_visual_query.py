"""Special-token helpers for visual-query code generation."""

VQ_START_TOKEN = "<vq>"
VQ_END_TOKEN = "</vq>"
MASK_VIS_TOKEN = "<mask_vis>"
REPLAY_TOKEN = "<|replay_pad|>"
VIS_TOKEN_PREFIX = "<vis_"
VIS_TOKEN_SUFFIX = ">"
DEFAULT_VISUAL_CODEBOOK_SIZE = 16384


def vis_token(code_id: int) -> str:
    return f"{VIS_TOKEN_PREFIX}{int(code_id)}{VIS_TOKEN_SUFFIX}"


def vis_tokens(codebook_size: int = DEFAULT_VISUAL_CODEBOOK_SIZE) -> list[str]:
    return [vis_token(i) for i in range(int(codebook_size))]


def visual_query_special_tokens(codebook_size: int = DEFAULT_VISUAL_CODEBOOK_SIZE) -> list[str]:
    return [
        VQ_START_TOKEN,
        VQ_END_TOKEN,
        MASK_VIS_TOKEN,
        REPLAY_TOKEN,
        *vis_tokens(codebook_size),
    ]


def add_visual_query_tokens(tokenizer, codebook_size: int = DEFAULT_VISUAL_CODEBOOK_SIZE) -> int:
    """Add the full visual-query vocabulary if it is missing."""

    tokens = visual_query_special_tokens(codebook_size)
    return tokenizer.add_special_tokens({"additional_special_tokens": tokens})


def sync_visual_query_token_ids(config, tokenizer, codebook_size: int = DEFAULT_VISUAL_CODEBOOK_SIZE) -> None:
    """Mirror visual-query token ids onto the model config."""

    config.visual_codebook_size = int(codebook_size)
    config.vq_start_token_id = tokenizer.convert_tokens_to_ids(VQ_START_TOKEN)
    config.vq_end_token_id = tokenizer.convert_tokens_to_ids(VQ_END_TOKEN)
    config.mask_vis_token_id = tokenizer.convert_tokens_to_ids(MASK_VIS_TOKEN)
    config.replay_token_id = tokenizer.convert_tokens_to_ids(REPLAY_TOKEN)
    config.vis_token_start_id = tokenizer.convert_tokens_to_ids(vis_token(0))
    config.vis_token_end_id = tokenizer.convert_tokens_to_ids(vis_token(int(codebook_size) - 1))


__all__ = [
    "DEFAULT_VISUAL_CODEBOOK_SIZE",
    "MASK_VIS_TOKEN",
    "REPLAY_TOKEN",
    "VIS_TOKEN_PREFIX",
    "VIS_TOKEN_SUFFIX",
    "VQ_END_TOKEN",
    "VQ_START_TOKEN",
    "add_visual_query_tokens",
    "sync_visual_query_token_ids",
    "vis_token",
    "vis_tokens",
    "visual_query_special_tokens",
]
