
"""Streamlit chat interface for The Librarian."""

import os
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv(
    "LIBRARIAN_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = 240

st.set_page_config(
    page_title="The Librarian | SMU",
    page_icon="📚",
    layout="centered",
)

st.title("📚 The Librarian")
st.caption(
    "Ask questions about South Mediterranean University "
    "using information retrieved from its public website."
)

st.info(
    "This assistant uses public SMU website content. "
    "It cannot access private student records or internal systems. "
    "Check the cited sources before relying on an answer."
)


def is_safe_source_url(url: str) -> bool:
    """Allow only HTTP(S) source links with a valid hostname."""
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and not parsed.username
            and not parsed.password
        )
    except (TypeError, ValueError):
        return False


def display_sources(sources: list[dict]) -> None:
    """Display citations and links to supporting source pages."""
    if not sources:
        st.caption("No supporting source links were returned.")
        return

    st.markdown("**Sources**")

    for index, source in enumerate(sources, start=1):
        title = str(source.get("title") or "SMU source")
        citation = str(source.get("citation") or f"S{index}")
        url = str(source.get("source_url") or "")
        chunk_id = str(source.get("chunk_id") or "")

        if is_safe_source_url(url):
            st.markdown(f"- **[{citation}] {title}** — [Open source]({url})")
        else:
            st.markdown(f"- **[{citation}] {title}**")

        if chunk_id:
            st.caption(f"Passage ID: {chunk_id}")


def display_timings(
    timings: dict,
    service_latency_ms: float | None = None,
) -> None:
    """Display end-to-end and model timing information."""
    if service_latency_ms is not None:
        st.caption(f"End-to-end service latency: {service_latency_ms / 1000:.2f} seconds")

    total = timings.get("total")
    generation = timings.get("generation")
    prompt_evaluation = timings.get("prompt_evaluation")

    if total is not None:
        st.caption(f"Pipeline timing: {total / 1000:.2f} seconds")

    if generation is not None:
        st.caption(f"Generation timing: {generation / 1000:.2f} seconds")

    if prompt_evaluation is not None:
        st.caption(
            f"Prompt evaluation timing: {prompt_evaluation / 1000:.2f} seconds"
        )


def submit_feedback(message_index: int, rating: str) -> None:
    """Save feedback for the selected assistant response in this session."""
    feedback = st.session_state.setdefault("feedback", {})
    feedback[str(message_index)] = rating


if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

with st.sidebar:
    st.subheader("Conversation")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    st.caption(f"API endpoint: {API_URL}")


# Display previous messages.
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))
            display_timings(
                message.get("timings_ms", {}),
                message.get("service_latency_ms"),
            )

            if message.get("request_succeeded", True):
                st.markdown("**Was this answer helpful?**")

                current_feedback = st.session_state.feedback.get(str(index))

                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button(
                        "👍 Helpful",
                        key=f"helpful_{index}",
                        disabled=current_feedback is not None,
                        use_container_width=True,
                    ):
                        submit_feedback(index, "helpful")
                        st.rerun()

                with col_no:
                    if st.button(
                        "👎 Not helpful",
                        key=f"not_helpful_{index}",
                        disabled=current_feedback is not None,
                        use_container_width=True,
                    ):
                        submit_feedback(index, "not_helpful")
                        st.rerun()

                if current_feedback is not None:
                    st.caption("Thank you for your feedback.")


question = st.chat_input("Ask a question about SMU...")

if question:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching SMU sources and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

                if response.ok:
                    result = response.json()
                    answer = result.get(
                        "answer",
                        "No answer was returned.",
                    )
                    sources = result.get("sources", [])
                    timings = result.get("timings_ms", {})
                    service_latency_ms = result.get("service_latency_ms")

                    st.markdown(answer)
                    display_sources(sources)
                    display_timings(timings, service_latency_ms)

                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timings_ms": timings,
                        "service_latency_ms": service_latency_ms,
                        "request_succeeded": True,
                    }
                    st.session_state.messages.append(assistant_message)

                    st.markdown("**Was this answer helpful?**")
                    col_yes, col_no = st.columns(2)

                    message_index = len(st.session_state.messages) - 1

                    with col_yes:
                        if st.button(
                            "👍 Helpful",
                            key=f"helpful_{message_index}",
                            use_container_width=True,
                        ):
                            submit_feedback(message_index, "helpful")
                            st.rerun()

                    with col_no:
                        if st.button(
                            "👎 Not helpful",
                            key=f"not_helpful_{message_index}",
                            use_container_width=True,
                        ):
                            submit_feedback(message_index, "not_helpful")
                            st.rerun()

                else:
                    try:
                        error_data = response.json()
                        detail = error_data.get("detail", "The request failed.")
                    except ValueError:
                        detail = "The API returned an unreadable error response."

                    st.error(
                        f"Request failed (HTTP {response.status_code}): {detail}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "Sorry, I couldn't process that question. "
                                "Please try again later."
                            ),
                            "sources": [],
                            "timings_ms": {},
                            "request_succeeded": False,
                        }
                    )

            except requests.Timeout:
                st.error(
                    "The request timed out. The model may still be processing "
                    "the question. Please try again."
                )

            except requests.ConnectionError:
                st.error(
                    "Could not connect to The Librarian API. "
                    "Check that FastAPI is running at "
                    f"{API_URL}."
                )

            except requests.RequestException:
                st.error(
                    "A network error occurred while contacting the API. "
                    "Please try again."
                )

            except (ValueError, TypeError, KeyError):
                st.error(
                    "The API returned an unexpected response. "
                    "Please check the backend logs."
                )
