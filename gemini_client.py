"""
gemini_client.py
----------------
Gemini API integration with:
  - Exponential backoff + jitter for 429 rate-limit errors
  - Model fallback chain: flash → flash-lite → 1.5-flash → (error)
  - Conversation history support (multi-turn Socratic dialogue)
  - Structured error types for clean FastAPI error handling
  - Australian Socratic tone + follow-up suggestions

Uses the current `google-genai` SDK (google.genai), which replaced
the deprecated `google-generativeai` package.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types
import google.genai.errors as genai_errors

from nlp_utils import PreprocessedInput

logger = logging.getLogger(__name__)

# Configuration

# Model fallback chain
MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",        # Primary: best balance of quality + rate limits
    "gemini-2.0-flash-lite",   # Fallback 1: higher RPM, lower quality
    "gemini-2.5-flash-lite",   # Fallback 2: separate quota bucket, very fast
]

MAX_RETRIES = 2
BASE_DELAY_SECONDS = 1.5
MAX_DELAY_SECONDS = 8.0
JITTER_FACTOR = 0.3

# Custom exception types

class GeminiRateLimitError(Exception):
    """Raised when all retries and model fallbacks are exhausted due to 429s."""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after ~{retry_after}s." if retry_after else "Rate limit exceeded.")

class GeminiAPIError(Exception):
    """Raised for non-rate-limit Gemini API failures."""

class GeminiSafetyError(Exception):
    """Raised when Gemini blocks a prompt due to safety filters."""

# Conversation history container

@dataclass
class ConversationHistory:
    """Stores multi-turn chat history."""
    turns: list[dict] = field(default_factory=list)
    max_turns: int = 20

    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "parts": [text]})
        self._trim()

    def add_model(self, text: str) -> None:
        self.turns.append({"role": "model", "parts": [text]})
        self._trim()

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def to_genai_history(self) -> list[types.Content]:
        """Convert to google-genai Content objects."""
        return [
            types.Content(
                role=turn["role"],
                parts=[types.Part(text=turn["parts"][0])]
            )
            for turn in self.turns
        ]

    def clear(self) -> None:
        self.turns.clear()

# Socratic system prompt 

SOCRATIC_SYSTEM_PROMPT = """You are Socra — someone who does a basic socratic dialogue.

Your job:
- Guide students (ages 7–18) to discover answers themselves — never give direct answers
- Ask ONE probing question at a time to help them think it through
- Use fun, everyday comparisons — sport, food, animals, weather, school life
- Keep responses SHORT and straightforward: 2–3 sentences, then your question
- End every main response with exactly ONE question

RESPONSE FORMAT — follow this exactly, every single time:
[Your Socratic response: 2–3 sentences, end with one question]
---EXPLORE---
• [Curious follow-up question, max 12 words]
• [Curious follow-up question, max 12 words]
• [Curious follow-up question, max 12 words]

Rules:
- The ---EXPLORE--- separator must appear on its own line, exactly as written
- Always include exactly 3 bullet points after it (starting with •)
- The 3 follow-up questions should spark curiosity and make the student want to dig deeper
- Match language complexity to how the user writes — simpler for younger, deeper for older
- Never open with "Great question!" or "That's interesting!" — just dive in"""

# Core client class

class GeminiClient:
    """
    Thread-safe Gemini API client with retry logic and model fallback.
    Instantiate once at app startup and reuse across requests.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            system_instruction=SOCRATIC_SYSTEM_PROMPT,
            temperature=0.85,
            top_p=0.92,
            max_output_tokens=400,
        )
        logger.info("GeminiClient initialised. Model chain: %s", MODEL_FALLBACK_CHAIN)

    # Public API
    def get_response(
        self,
        preprocessed: PreprocessedInput,
        history: ConversationHistory,
    ) -> tuple[str, list[str]]:
        """
        Generate a Socratic response + follow-up suggestions.

        Returns:
            (response_text, suggestions) tuple.
            suggestions is a list of up to 3 follow-up question strings.

        Raises:
            GeminiRateLimitError: All models and retries exhausted.
            GeminiAPIError: Non-recoverable API error.
            GeminiSafetyError: Safety filter blocked the prompt.
        """
        enriched_prompt = self._build_enriched_prompt(preprocessed)

        for model_name in MODEL_FALLBACK_CHAIN:
            logger.info("Trying model: %s", model_name)
            try:
                raw_response = self._call_with_retry(
                    model_name=model_name,
                    user_message=enriched_prompt,
                    history=history,
                )
                response_text, suggestions = self._parse_response(raw_response)
                # Update history with the clean response (no suggestions section)
                history.add_user(preprocessed.original)
                history.add_model(response_text)
                return response_text, suggestions

            except GeminiRateLimitError:
                logger.warning("Rate limit exhausted on model %s, trying next.", model_name)
                continue

            except GeminiSafetyError:
                raise

            except GeminiAPIError as e:
                logger.error("API error on model %s: %s", model_name, e)
                continue

        raise GeminiRateLimitError()

    # Internal helpers

    def _parse_response(self, raw: str) -> tuple[str, list[str]]:
        """
        Parse the Gemini response into (response_text, suggestions).
        Splits on ---EXPLORE--- separator.
        """
        parts = raw.split("---EXPLORE---")
        response_text = parts[0].strip()
        suggestions: list[str] = []

        if len(parts) > 1:
            for line in parts[1].strip().split("\n"):
                line = line.strip()
                if line.startswith("•"):
                    suggestion = line[1:].strip()
                    if suggestion:
                        suggestions.append(suggestion)

        return response_text, suggestions[:3]

    def _call_with_retry(
        self,
        model_name: str,
        user_message: str,
        history: ConversationHistory,
    ) -> str:
        """Retry loop with exponential backoff + jitter for a single model."""
        last_exception: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                chat = self._client.chats.create(
                    model=model_name,
                    history=history.to_genai_history(),
                    config=self._config,
                )
                response = chat.send_message(user_message)

                if not response.candidates:
                    raise GeminiSafetyError("Response blocked by safety filters.")

                candidate = response.candidates[0]
                finish_name = candidate.finish_reason.name if candidate.finish_reason else ""
                if finish_name == "SAFETY":
                    raise GeminiSafetyError("Content flagged by safety filters.")

                return response.text.strip()

            except GeminiSafetyError:
                raise

            except genai_errors.ClientError as e:
                err_str = str(e)
                status = getattr(e, "status_code", None) or getattr(e, "code", None)

                if status == 429 or "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower():
                    last_exception = e
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "429 on %s (attempt %d/%d). Waiting %.1fs...",
                        model_name, attempt + 1, MAX_RETRIES + 1, delay,
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(delay)
                    else:
                        raise GeminiRateLimitError(retry_after=int(delay))
                elif status == 403 or "permission" in err_str.lower() or "api key" in err_str.lower():
                    raise GeminiAPIError(f"API key invalid or lacks permissions: {e}") from e
                elif status == 400 or "invalid" in err_str.lower():
                    raise GeminiAPIError(f"Invalid request: {e}") from e
                else:
                    raise GeminiAPIError(f"Client error: {e}") from e

            except genai_errors.ServerError as e:
                last_exception = e
                delay = self._backoff_delay(attempt)
                logger.warning("Service unavailable (attempt %d). Waiting %.1fs.", attempt + 1, delay)
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                else:
                    raise GeminiAPIError(f"Service unavailable after {MAX_RETRIES} retries.") from e

            except Exception as e:
                raise GeminiAPIError(f"Unexpected error: {e}") from e

        raise GeminiAPIError(f"Failed after {MAX_RETRIES} retries: {last_exception}")

    def _build_enriched_prompt(self, p: PreprocessedInput) -> str:
        """Enrich the user message with NLP-derived metadata for better Socratic framing."""
        lines = [p.original]

        if p.key_concepts:
            lines.append(f"[Key concepts detected: {', '.join(p.key_concepts)}]")

        if p.sentiment != "neutral":
            lines.append(f"[User sentiment: {p.sentiment}]")

        if not p.is_question:
            lines.append("[Note: User made a statement, not a question — invite them to question it]")

        if p.word_count < 5:
            lines.append("[Note: Very brief input — ask them to elaborate]")

        return "\n".join(lines)

    @staticmethod
    def _backoff_delay(attempt: int, hint: Optional[int] = None) -> float:
        """Exponential backoff with jitter."""
        exponential = min(BASE_DELAY_SECONDS * (2 ** attempt), MAX_DELAY_SECONDS)
        jitter = exponential * JITTER_FACTOR * (random.random() * 2 - 1)
        delay = max(0.5, exponential + jitter)
        if hint:
            delay = max(delay, hint)
        return round(delay, 2)
