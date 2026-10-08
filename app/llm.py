from langchain_groq import ChatGroq

from app.config import get_settings


def get_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(model=s.llm_model, api_key=s.groq_api_key, temperature=0, max_retries=4)
from typing import TypeVar

from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import get_settings

T = TypeVar("T", bound=BaseModel)


def get_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(model=s.llm_model, api_key=s.groq_api_key, temperature=0, max_retries=4)


def invoke_structured(prompt, schema: type[T], inputs: dict, default: T) -> T:
    """Run `prompt | llm` and parse the result into `schema`, robustly.

    Some Groq models occasionally answer in plain text instead of calling the structured-output
    tool (HTTP 400 'tool_use_failed'). We try tool calling first, then JSON-schema mode, and fall
    back to a safe `default` so one flaky call never crashes a whole request or evaluation run.
    """
    llm = get_llm()
    for method in ("function_calling", "json_schema"):
        try:
            return (prompt | llm.with_structured_output(schema, method=method)).invoke(inputs)
        except Exception as e:  # noqa: BLE001
            print(f"[structured output via {method} failed: {type(e).__name__}]", flush=True)
    return default