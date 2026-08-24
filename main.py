from ollama import chat

from speech import record_audio, transcribe
from tts import speak
from rag import build_index, search

from tools import get_learning_progress


MODEL = "qwen3:8b"


def search_knowledge(query, index, chunks):
    results = search(
        query,
        index,
        chunks,
        top_k=4,
    )

    return results


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the uploaded study documents. "
                "Use this for questions about the user's "
                "uploaded documents or study material."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The question or topic to search "
                            "for in the documents."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_learning_progress",
            "description": (
                "Get the learner's stored learning progress "
                "from PostgreSQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


print("Building knowledge base...")

index, chunks = build_index()

print(
    f"Loaded {len(chunks)} knowledge chunks."
)

print("Voice RAG learning agent ready.\n")


messages = [
    {
        "role": "system",
        "content": """
You are a voice-first AI learning assistant.

Your job is to answer questions using the user's
uploaded study documents.

RULES:

1. For questions about uploaded documents or study
   material, ALWAYS use search_knowledge first.

2. Answer using ONLY the retrieved document context.

3. Do not invent information that is not supported
   by the retrieved context.

4. If the documents do not contain enough information,
   clearly say that the information was not found.

5. For questions about the learner's progress,
   use get_learning_progress.

6. Keep answers concise and natural because the
   answer will be spoken aloud.

7. Do not use Markdown, bullet points, emojis,
   or unnecessary formatting.

8. When document context is available, mention the
   source document and page in the final response.

9. Do not expose internal tool calls or implementation
   details to the learner.
""",
    }
]


while True:

    if not record_audio():
        continue

    text = transcribe().strip()

    if not text:
        print("No speech detected.")
        continue

    print(f"\nYou: {text}")

    command = (
        text.lower()
        .strip()
        .rstrip(".!?")
    )

    if command in {
        "exit",
        "quit",
        "stop",
        "goodbye",
    }:
        speak("Goodbye!")
        print("Goodbye!")
        break

    messages.append(
        {
            "role": "user",
            "content": text,
        }
    )

    response = chat(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    while response.message.tool_calls:

        messages.append(response.message)

        for tool_call in response.message.tool_calls:

            name = tool_call.function.name
            args = tool_call.function.arguments

            print(
                f"\n[Agent] Tool: {name}"
            )

            print(
                f"[Agent] Arguments: {args}"
            )

            if name == "search_knowledge":

                results = search_knowledge(
                    args["query"],
                    index,
                    chunks,
                )

                result_text = []

                print(
                    "\nRetrieved sources:"
                )

                for result in results:

                    source = result.get(
                        "source",
                        "Unknown",
                    )

                    page = result.get(
                        "page",
                        "?",
                    )

                    text_content = result.get(
                        "text",
                        "",
                    )

                    print(
                        f"- {source} "
                        f"(Page {page})"
                    )

                    result_text.append(
                        (
                            f"SOURCE: {source}\n"
                            f"PAGE: {page}\n"
                            f"CONTENT:\n"
                            f"{text_content}"
                        )
                    )

                result = "\n\n".join(
                    result_text
                )

            elif name == "get_learning_progress":

                result = get_learning_progress()

            else:

                result = {
                    "error": (
                        f"Unknown tool: {name}"
                    )
                }

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                }
            )

        response = chat(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

    answer = (
        response.message.content
        .strip()
    )

    if answer:

        messages.append(
            response.message
        )

        print(
            f"\nMentor:\n{answer}"
        )

        speak(answer)

    print(
        "\nListening..."
    )