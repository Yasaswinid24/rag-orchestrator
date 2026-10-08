"""Agentic RAG workflow built with LangGraph.

START -> route --(direct)--> direct_answer -> END
           |
           +--(retrieve)--> retrieve -> grade_documents
                                          |-- relevant docs --> generate -> check_grounding -> END
                                          |-- none, retries left --> rewrite_query -> retrieve
                                          |-- none, no retries --> generate (will say "not enough info")
"""
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import get_llm, invoke_structured


class GraphState(TypedDict, total=False):
    question: str
    query: str                 # possibly rewritten query used for retrieval
    documents: list[Document]
    generation: str
    rewrites: int
    route: str
    grounded: bool
    trace: list[str]


class RouteDecision(BaseModel):
    route: Literal["retrieve", "direct"] = Field(
        description="'direct' ONLY for greetings, thanks, goodbyes, or questions about the assistant itself. "
        "'retrieve' for every other question, including general-knowledge questions."
    )


class Grade(BaseModel):
    relevant_ids: list[int] = Field(
        description="Numbers of the passages that help answer the question; empty list if none."
    )


class Grounding(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is supported by the context.")


def _fmt(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source')}) {d.page_content}"
        for i, d in enumerate(docs)
    )


def _log(state: GraphState, msg: str) -> list[str]:
    return [*state.get("trace", []), msg]


def build_graph():
    from app.vectorstore import get_vectorstore

    s = get_settings()
    llm = get_llm()

    # ---- nodes -------------------------------------------------------
    def route(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You route messages for a document question-answering assistant. Choose 'direct' ONLY for greetings, thanks, goodbyes or questions about the assistant itself. Choose 'retrieve' for every other question, including general-knowledge questions, so answers come from the documents."),
            ("human", "{question}"),
        ])
        d = invoke_structured(prompt, RouteDecision, {"question": state["question"]},
                              default=RouteDecision(route="retrieve"))
        return {"route": d.route, "query": state["question"], "rewrites": 0,
                "trace": _log(state, f"route={d.route}")}

    def direct_answer(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a document question-answering assistant. Reply briefly and politely to "
                       "greetings or small talk, and say you answer questions about the indexed documents. "
                       "Do not answer factual questions."),
            ("human", "{question}"),
        ])
        out = (prompt | llm).invoke({"question": state["question"]}).content
        return {"generation": out, "documents": [], "grounded": False,
                "trace": _log(state, "direct_answer")}

    def retrieve(state: GraphState) -> GraphState:
        docs = get_vectorstore().similarity_search(state["query"], k=s.top_k)
        return {"documents": docs, "trace": _log(state, f"retrieve(k={len(docs)}, q={state['query']!r})")}

    def grade_documents(state: GraphState) -> GraphState:
        docs = state["documents"]
        if not docs:
            return {"trace": _log(state, "grade_documents(kept=0)")}
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You grade retrieved passages. Return the numbers of passages that contain "
                       "information useful for answering the question (empty list if none)."),
            ("human", "Question: {question}\n\nPassages:\n{passages}"),
        ])
        g = invoke_structured(
            prompt, Grade, {"question": state["question"], "passages": _fmt(docs)},
            default=Grade(relevant_ids=list(range(1, len(docs) + 1))))
        keep = set(g.relevant_ids)
        kept = [d for i, d in enumerate(docs, 1) if i in keep]
        return {"documents": kept, "trace": _log(state, f"grade_documents(kept={len(kept)})")}

    def rewrite_query(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the question into a better standalone search query for semantic "
                       "vector search. Return only the rewritten query."),
            ("human", "{question}"),
        ])
        q = (prompt | llm).invoke({"question": state["question"]}).content.strip()
        return {"query": q, "rewrites": state.get("rewrites", 0) + 1,
                "trace": _log(state, f"rewrite_query -> {q!r}")}

    def generate(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using ONLY the numbered context. Cite sources like [1], [2]. "
                       "If the context is insufficient, say you do not have enough information."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        ctx = _fmt(state.get("documents", [])) or "(no relevant context found)"
        out = (prompt | llm).invoke({"context": ctx, "question": state["question"]}).content
        return {"generation": out, "trace": _log(state, "generate")}

    def check_grounding(state: GraphState) -> GraphState:
        if not state.get("documents"):
            return {"grounded": False, "trace": _log(state, "check_grounding(no context)")}
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Check whether the answer is fully supported by the context."),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
        ])
        g = invoke_structured(
            prompt, Grounding, {"context": _fmt(state["documents"]), "answer": state["generation"]},
            default=Grounding(grounded=False))
        return {"grounded": g.grounded, "trace": _log(state, f"check_grounding={g.grounded}")}

    # ---- edges -------------------------------------------------------
    def after_route(state: GraphState) -> str:
        return "direct_answer" if state["route"] == "direct" else "retrieve"

    def after_grade(state: GraphState) -> str:
        if state["documents"]:
            return "generate"
        return "rewrite_query" if state.get("rewrites", 0) < s.max_rewrites else "generate"

    g = StateGraph(GraphState)
    for name, fn in [("route", route), ("direct_answer", direct_answer), ("retrieve", retrieve),
                     ("grade_documents", grade_documents), ("rewrite_query", rewrite_query),
                     ("generate", generate), ("check_grounding", check_grounding)]:
        g.add_node(name, fn)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["direct_answer", "retrieve"])
    g.add_edge("direct_answer", END)
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", after_grade, ["generate", "rewrite_query"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_grounding")
    g.add_edge("check_grounding", END)
    return g.compile()