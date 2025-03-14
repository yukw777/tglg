import json
from pathlib import Path

import whisperx
from jsonargparse import auto_cli
from tqdm import tqdm


def main(
    soccernet_dir: str,
    output_dir: str,
    whisperx_model: str = "large-v2",
    device: str = "cuda",
    batch_size: int = 32,
    hf_auth_token: str | None = None,
) -> None:
    # load models
    model = whisperx.load_model(whisperx_model, device=device)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_auth_token, device=device
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Figure out the progress
    completed = set(
        (trancribed_file.parts[-2], trancribed_file.stem)
        for trancribed_file in output_dir_path.glob("**/*.json")
    )
    vid_paths = []
    for vid_path in Path(soccernet_dir).glob("**/*.mkv"):
        if (vid_path.parts[-2], vid_path.stem) not in completed:
            vid_paths.append(vid_path)

    for vid_path in tqdm(vid_paths):
        audio = whisperx.load_audio(str(vid_path))
        language = model.detect_language(audio)
        if language == "en":
            # transcribe
            result = model.transcribe(audio, batch_size=batch_size, language=language)

            # align
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device
            )

            # diarize
            diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # remove word level alignment for now
            for seg in result["segments"]:
                del seg["words"]
        else:
            result = {"language": language, "segments": []}

        transcript_dir = output_dir_path / vid_path.parts[-2]
        transcript_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_dir / f"{vid_path.stem}.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
