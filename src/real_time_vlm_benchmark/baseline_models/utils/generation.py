from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """
    Closely mirrors transformers.GenerationConfig
    """

    max_new_tokens: int | None = None
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
