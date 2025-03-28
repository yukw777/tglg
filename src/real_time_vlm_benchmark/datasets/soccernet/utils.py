def convert_pbp_annotated(pbp_annotated: dict, tolerance: float = 5) -> list[dict]:
    """
    Figure out play-by-play segments with the given tolerance, and mark them for evaluation.
    """
    segments = pbp_annotated["segments"]
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg["end"] < seg["start"]:
            # WhisperX seems to have a bug when a single number is transcribed,
            # the start and end times are flipped. Manual inspection shows that
            # these should be merged with the previous segment.
            segments[i - 1]["text"] += f" {seg['text']}"
            segments[i - 1]["end"] = seg["start"]
            segments = segments[:i] + segments[i + 1 :]
            continue
        i += 1

    # first copy over all the segments
    anns: list[dict] = []
    for seg in segments:
        anns.append(
            {
                "role": "assistant",
                "content": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "eval": False,
            }
        )

    # identify play-by-play segments and mark them for evaluation
    i = 0
    while i < len(segments):
        if segments[i]["is_pbp"]:
            # first look past and include segments within tolerance
            start = i
            while start - 1 > 0:
                if segments[i]["start"] - tolerance <= segments[start - 1]["end"]:
                    start -= 1
                else:
                    break
            # next move forward until it's no longer pbp
            # NOTE: i points to the last pbp segment after this loop
            while i + 1 < len(segments):
                if segments[i + 1]["is_pbp"]:
                    i += 1
                else:
                    break
            # finally look forward and include segments within tolerance
            end = i
            while end < len(segments):
                if segments[i]["end"] + tolerance >= segments[end]["start"]:
                    end += 1
                else:
                    break
            # we've found a segment mark them for evaluation
            for j in range(start, end):
                anns[j]["eval"] = True
        i += 1

    # Filter out non eval utterances since we're only evaluating the model for
    # play-by-play commentaries
    anns = [ann for ann in anns if ann["eval"]]

    return anns
