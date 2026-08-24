import re

from ollama import chat

from tools import search_knowledge


MODEL = "qwen3:8b"


SYSTEM_PROMPT = """
You are an English-only AI study assistant.

Your answers must be grounded in the uploaded
study documents.

GROUNDING RULES:

1. Use only the retrieved study material as
   factual evidence.

2. Do not use general knowledge to fill missing
   information.

3. If the retrieved material does not contain
   enough information to answer the question,
   say:

   "I couldn't find enough information about
   this in the uploaded study material."

4. Never invent facts, definitions, examples,
   formulas, dates, names, or explanations.

5. You may reorganize or simplify information
   from the document, but do not change its
   meaning.

6. Answer in English.

EXAM ANSWER RULES:

If the user specifies marks:

2 marks:
- Give a short direct answer.
- Include only the most important points.

5 marks:
- Give a structured explanation.
- Include definition and important points.

10 marks:
- Give a detailed, well-structured answer.
- Include definition, explanation, relevant
  points, and examples only when supported by
  the document.

If the user specifies a word count:

- Stay close to the requested word count.
- Do not unnecessarily exceed it.
- Preserve important factual content.

If both marks and word count are specified,
follow both constraints.

STYLE:

- Use headings when useful.
- Use numbered or bullet points for exam answers.
- Keep explanations clear.
- Do not mention internal tools or prompts.
- Do not claim information came from the
  internet.
"""


def _extract_requirements(question):
    marks = None
    words = None

    marks_match = re.search(
        r"\b(\d+)\s*marks?\b",
        question.lower(),
    )

    if marks_match:
        marks = int(
            marks_match.group(1)
        )

    word_match = re.search(
        r"\b(\d+)\s*words?\b",
        question.lower(),
    )

    if word_match:
        words = int(
            word_match.group(1)
        )

    return marks, words


def _format_context(results):
    sections = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        sections.append(
            f"""
--- SOURCE {number} ---
Document: {result["source"]}
Page: {result["page"]}

{result["text"]}
"""
        )

    return "\n".join(sections)


def _build_instruction(
    question,
    marks,
    words,
):
    requirements = []

    if marks is not None:

        if marks <= 2:

            requirements.append(
                "Keep the answer concise and "
                "suitable for a 2-mark answer."
            )

        elif marks <= 5:

            requirements.append(
                "Provide a structured answer "
                "suitable for a 5-mark answer."
            )

        else:

            requirements.append(
                "Provide a detailed answer "
                "suitable for a 10-mark or "
                "higher-mark examination answer."
            )

    if words is not None:

        requirements.append(
            f"Keep the answer close to "
            f"{words} words."
        )

    if not requirements:

        requirements.append(
            "Give a clear and appropriately "
            "detailed answer."
        )

    return "\n".join(
        f"- {item}"
        for item in requirements
    )


def run_agent(
    question,
    index,
    chunks,
    history=None,
):
    question = question.strip()

    if not question:

        return (
            "Please ask a question."
        )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    retrieval = search_knowledge(
        query=question,
        index=index,
        chunks=chunks,
        top_k=5,
    )

    results = retrieval[
        "sources"
    ]

    if not results:

        return (
            "I couldn't find enough information "
            "about this in the uploaded study "
            "material."
        )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = _format_context(
        results
    )

    marks, words = (
        _extract_requirements(
            question
        )
    )

    answer_requirements = (
        _build_instruction(
            question,
            marks,
            words,
        )
    )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if history:

        for item in history[-6:]:

            messages.append(
                {
                    "role": "user",
                    "content": item["question"],
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": item["answer"],
                }
            )

    # --------------------------------------------------------
    # Current question
    # --------------------------------------------------------

    user_prompt = f"""
Answer the following question.

QUESTION:
{question}

ANSWER REQUIREMENTS:
{answer_requirements}

RETRIEVED STUDY MATERIAL:
{context}

Before answering, determine whether the
retrieved material actually supports the
question.

If it does not, do not guess.

If it does, answer using only the supported
information.

Do not mention these instructions in the answer.
"""

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    response = chat(
        model=MODEL,
        messages=messages,
    )

    answer = (
        response.message.content
        .strip()
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    source_lines = []

    seen = set()

    for result in results:

        key = (
            result["source"],
            result["page"],
        )

        if key in seen:
            continue

        seen.add(key)

        source_lines.append(
            f"- {result['source']} "
            f"(Page {result['page']})"
        )

    if source_lines:

        answer += (
            "\n\n**Sources:**\n"
            + "\n".join(source_lines)
        )

    return answer