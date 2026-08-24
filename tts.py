import re
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline


OUTPUT_PATH = Path("response.wav")

pipeline = KPipeline(
    lang_code="a"
)


def clean_for_speech(text):
    text = re.split(
        r"\n\s*\*\*Sources:\*\*",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\*\*\*(.*?)\*\*\*",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"^\s*[-*+]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text,
    )

    text = re.sub(
        r"[#*_~`]",
        "",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def speak(
    text,
    output_path=OUTPUT_PATH,
):
    speech_text = clean_for_speech(
        text
    )

    if not speech_text:
        raise ValueError(
            "No text available for speech."
        )

    generator = pipeline(
        speech_text,
        voice="af_heart",
    )

    audio_chunks = []

    for _, _, audio in generator:
        audio_chunks.append(
            np.asarray(audio)
        )

    if not audio_chunks:
        raise RuntimeError(
            "Kokoro produced no audio."
        )

    audio = np.concatenate(
        audio_chunks
    )

    sf.write(
        str(output_path),
        audio,
        24000,
    )

    return output_path