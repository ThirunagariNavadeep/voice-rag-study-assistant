import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from agent import run_agent
from rag import add_uploaded_pdf, build_index
from speech import transcribe
from tts import speak

MAX_HISTORY = 6

st.set_page_config(
    page_title="Voice RAG Study Assistant",
    page_icon="🎙️",
    layout="wide",
)


DEFAULT_STATE = {
    "index": None,
    "chunks": [],
    "knowledge_base_files": [],
    "pending_question": None,
    "pending_audio_hash": None,
    "recording_version": 0,
    "conversation": [],
    "response_audio": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_recording():
    st.session_state["pending_question"] = None
    st.session_state["pending_audio_hash"] = None
    st.session_state["recording_version"] += 1


def clear_conversation():
    st.session_state["conversation"] = []


def clear_knowledge_base():
    st.session_state["index"] = None
    st.session_state["chunks"] = []
    st.session_state["knowledge_base_files"] = []
    st.session_state["response_audio"] = None

    reset_recording()
    clear_conversation()


def clear_response_audio():
    st.session_state["response_audio"] = None

st.title(
    "🎙️ Voice RAG Study Assistant"
)

st.caption(
    "Upload PDF study material, ask questions "
    "by voice or text, and receive grounded "
    "answers on screen and through voice."
)

st.header(
    "📚 Upload Study Material"
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)


if uploaded_files:

    st.write(
        f"{len(uploaded_files)} PDF(s) selected."
    )

    for uploaded_file in uploaded_files:
        st.write(
            f"📄 {uploaded_file.name}"
        )


    if st.button(
        "Build Knowledge Base",
        type="primary",
    ):

        uploaded_paths = []

        try:

            with st.spinner(
                "Saving PDF files..."
            ):

                for uploaded_file in uploaded_files:

                    path = add_uploaded_pdf(
                        uploaded_file
                    )

                    uploaded_paths.append(
                        path
                    )


            with st.spinner(
                "Building knowledge base..."
            ):

                index, chunks = build_index(
                    uploaded_paths
                )


            st.session_state["index"] = index
            st.session_state["chunks"] = chunks

            st.session_state[
                "knowledge_base_files"
            ] = [
                path.name
                for path in uploaded_paths
            ]

            clear_conversation()
            clear_response_audio()
            reset_recording()

            st.success(
                f"Knowledge base built successfully. "
                f"{len(chunks)} chunks indexed."
            )

        except Exception as error:

            st.error(
                f"Knowledge base error: {error}"
            )

st.subheader(
    "Knowledge Base"
)

files = st.session_state[
    "knowledge_base_files"
]


if files:

    for filename in files:
        st.write(
            f"✓ {filename}"
        )


    if st.button(
        "🗑️ Clear Knowledge Base"
    ):

        clear_knowledge_base()

        st.rerun()

else:

    st.info(
        "No knowledge base loaded."
    )

st.divider()

st.header(
    "🎤 Ask a Question"
)

st.caption(
    "Record your question and review the "
    "transcription before submitting it."
)


audio = st.audio_input(
    "Record your question",
    key=(
        "voice_question_"
        f"{st.session_state['recording_version']}"
    ),
)

if audio is not None:

    audio_bytes = audio.getvalue()

    audio_hash = hashlib.sha256(
        audio_bytes
    ).hexdigest()


    if (
        st.session_state[
            "pending_audio_hash"
        ]
        != audio_hash
    ):

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            audio_path = temp_file.name


        try:

            with st.spinner(
                "Transcribing..."
            ):

                question = transcribe(
                    audio_path
                )

        except Exception as error:

            st.error(
                f"Speech recognition failed: "
                f"{error}"
            )

            question = ""

        finally:

            Path(
                audio_path
            ).unlink(
                missing_ok=True
            )


        st.session_state[
            "pending_audio_hash"
        ] = audio_hash

        st.session_state[
            "pending_question"
        ] = question

pending_question = st.session_state[
    "pending_question"
]


if pending_question is not None:

    st.subheader(
        "Recognized Question"
    )


    if pending_question:

        st.info(
            pending_question
        )

    else:

        st.warning(
            "No speech was recognized."
        )


    col1, col2, col3 = st.columns(
        3
    )


    with col1:

        answer_button = st.button(
            "✅ Answer",
            type="primary",
            disabled=not bool(
                pending_question
            ),
        )


    with col2:

        discard_button = st.button(
            "🗑️ Discard"
        )


    with col3:

        rerecord_button = st.button(
            "🎤 Re-record"
        )

    if discard_button:

        reset_recording()
        clear_response_audio()

        st.rerun()

    if rerecord_button:

        reset_recording()
        clear_response_audio()

        st.rerun()

     if answer_button:

        if st.session_state["index"] is None:

            st.warning(
                "Build the knowledge base "
                "before asking a question."
            )

            st.stop()


        question = pending_question

        clear_response_audio()


        # ----------------------------------------------------
        # RAG + LLM
        # ----------------------------------------------------

        with st.spinner(
            "Searching the study material..."
        ):

            try:

                answer = run_agent(
                    question=question,
                    index=st.session_state["index"],
                    chunks=st.session_state["chunks"],
                    history=st.session_state["conversation"],
                )

            except Exception as error:

                st.error(
                    f"Agent error: {error}"
                )

                st.stop()


        # ----------------------------------------------------
        # Screen answer
        # ----------------------------------------------------

        st.subheader(
            "🤖 Answer"
        )

        st.write(
            answer
        )


        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

        st.session_state[
            "conversation"
        ].append(
            {
                "question": question,
                "answer": answer,
            }
        )

        st.session_state[
            "conversation"
        ] = st.session_state[
            "conversation"
        ][-MAX_HISTORY:]


        # ----------------------------------------------------
        # Generate TTS
        # ----------------------------------------------------

        with st.spinner(
            "Generating voice answer..."
        ):

            try:

                response_path = speak(
                    answer
                )

                if response_path:

                    st.session_state[
                        "response_audio"
                    ] = str(
                        response_path
                    )

            except Exception as error:

                st.warning(
                    f"Voice generation failed: "
                    f"{error}"
                )


        # ----------------------------------------------------
        # Clear transcription
        # ----------------------------------------------------

        st.session_state[
            "pending_question"
        ] = None

        st.session_state[
            "pending_audio_hash"
        ] = None


# ============================================================
# Persistent Voice Answer
# ============================================================

response_audio = st.session_state[
    "response_audio"
]


if response_audio:

    response_path = Path(
        response_audio
    )

    if response_path.exists():

        st.subheader(
            "🔊 Voice Answer"
        )

        st.audio(
            str(response_path),
            format="audio/wav",
            autoplay=True,
        )


# ============================================================
# Text Question
# ============================================================

st.divider()

st.header(
    "⌨️ Text Question"
)

question_text = st.text_input(
    "Type your question",
    placeholder=(
        "Explain backpropagation for "
        "5 marks in 150 words."
    ),
)


if st.button(
    "Ask Question"
):

    if st.session_state["index"] is None:

        st.warning(
            "Build the knowledge base "
            "before asking a question."
        )

        st.stop()


    if not question_text.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # Clear previous audio.
    clear_response_audio()


    # --------------------------------------------------------
    # RAG + LLM
    # --------------------------------------------------------

    with st.spinner(
        "Searching the study material..."
    ):

        try:

            answer = run_agent(
                question=question_text,
                index=st.session_state["index"],
                chunks=st.session_state["chunks"],
                history=st.session_state["conversation"],
            )

        except Exception as error:

            st.error(
                f"Agent error: {error}"
            )

            st.stop()


    # --------------------------------------------------------
    # Screen answer
    # --------------------------------------------------------

    st.subheader(
        "🤖 Answer"
    )

    st.write(
        answer
    )


    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    st.session_state[
        "conversation"
    ].append(
        {
            "question": question_text,
            "answer": answer,
        }
    )

    st.session_state[
        "conversation"
    ] = st.session_state[
        "conversation"
    ][-MAX_HISTORY:]


    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    with st.spinner(
        "Generating voice answer..."
    ):

        try:

            response_path = speak(
                answer
            )

            if response_path:

                st.session_state[
                    "response_audio"
                ] = str(
                    response_path
                )

        except Exception as error:

            st.warning(
                f"Voice generation failed: "
                f"{error}"
            )


# ============================================================
# Persistent audio after text question
# ============================================================

response_audio = st.session_state[
    "response_audio"
]


if response_audio:

    response_path = Path(
        response_audio
    )

    if response_path.exists():

        # Avoid duplicating the player if it
        # was already rendered above.
        if not (
            pending_question is not None
            and st.session_state.get(
                "pending_question"
            )
        ):

            st.subheader(
                "🔊 Voice Answer"
            )

            st.audio(
                str(response_path),
                format="audio/wav",
                autoplay=True,
            )


# ============================================================
# Conversation History
# ============================================================

if st.session_state[
    "conversation"
]:

    st.divider()

    st.header(
        "💬 Conversation History"
    )

    for item in st.session_state[
        "conversation"
    ]:

        with st.expander(
            item["question"]
        ):

            st.write(
                item["answer"]
            )


    if st.button(
        "🗑️ Clear Conversation"
    ):

        clear_conversation()

        st.rerun()
