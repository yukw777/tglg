from typing import TypedDict

GenerationConfig = TypedDict(
    "GenerationConfig",
    {
        "max_new_tokens": int,
        "do_sample": bool,
        "num_beams": int,
        "temperature": float,
        "top_k": int,
        "top_p": float,
    },
    total=False,
)
