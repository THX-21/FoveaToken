"""Token helpers for the Fovea JSON tool call."""

FOVEA_TOOL_CALL = '{"fovea"}'


def sync_fovea_trigger_ids(config, tokenizer) -> None:
    """Record existing tokenizer ids used by the Fovea JSON tool call."""

    trigger_ids = tokenizer(FOVEA_TOOL_CALL + "\n", add_special_tokens=False).input_ids
    if not trigger_ids:
        raise ValueError("Fovea tool call must tokenize to at least one token.")
    config.fovea_trigger_token_ids = [int(token_id) for token_id in trigger_ids]


__all__ = [
    "FOVEA_TOOL_CALL",
    "sync_fovea_trigger_ids",
]
