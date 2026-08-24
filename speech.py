import whisper


MODEL_NAME = "base"

model = whisper.load_model(
    MODEL_NAME
)


def transcribe(audio_path):
    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,
    )

    return result.get(
        "text",
        "",
    ).strip()


