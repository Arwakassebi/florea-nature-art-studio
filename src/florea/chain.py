"""Couche LangChain : modèle local (BaseChatModel), prompts LCEL, parseur Pydantic, outils @tool, agent ReAct.

    chaîne concept :   ChatPromptTemplate | LocalChat | ConceptParser(FloreaConcept)
    chaînes texte  :   ChatPromptTemplate | LocalChat | StrOutputParser
    agent          :   create_react_agent(LocalChat, [search_knowledge])   (repli : boucle ReAct manuelle)

`LocalChat` n'a besoin que d'une fonction  generate_fn(messages: list[dict], max_new_tokens, temperature, tools) -> str ,
fournie par le pipeline (Qwen) ou par les tests (réponses scriptées) : tout est testable sans GPU.
"""
import json
import re
import uuid
import warnings
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from florea.text import FloreaConcept, as_list, build_image_prompt, extract_json, normalize_colors


# ----------------------------------------------------------------------------- modèle local
def to_dicts(messages):
    """Messages LangChain -> format de chat Hugging Face (inclut les appels d'outils Qwen/Hermes)."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            d = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in m.tool_calls]
            out.append(d)
        else:
            out.append({"role": "user", "content": str(getattr(m, "content", m))})
    return out


_TOOL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text):
    """Texte du LLM -> (contenu, appels d'outils). Accepte <tool_call>{...}</tool_call> ou un JSON nu {"name", "arguments"}."""
    calls, content = [], text
    blocks = _TOOL_BLOCK.findall(text)
    if blocks:
        content = _TOOL_BLOCK.sub("", text).strip()
    else:
        j = extract_json(text)
        blocks = [json.dumps(j)] if j and "name" in j and "arguments" in j else []
        if blocks:
            content = ""
    for b in blocks:
        try:
            d = json.loads(b)
            args = d.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": d["name"], "args": args, "id": "call_" + uuid.uuid4().hex[:8], "type": "tool_call"})
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return content, calls


class LocalChat(BaseChatModel):
    """Enveloppe LangChain autour d'un modèle de chat local (fonction generate_fn)."""
    generate_fn: Any = None
    max_new_tokens: int = 400
    temperature: float = 0.7

    @property
    def _llm_type(self):
        return "florea-local-chat"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        tools = kwargs.get("tools")
        text = self.generate_fn(to_dicts(messages), self.max_new_tokens, self.temperature, tools)
        content, calls = parse_tool_calls(text) if tools else (text, [])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=calls))])

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)


# ----------------------------------------------------------------------------- concept : prompt LCEL + parseur Pydantic
class ConceptParser(PydanticOutputParser):
    """JSON tolérant (```, bavardage, virgules en trop) PUIS validation stricte par le schéma Pydantic FloreaConcept."""

    def parse_result(self, result, *, partial=False):
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        data = extract_json(text)
        if data is None:
            raise OutputParserException("Pas de JSON dans la réponse", llm_output=text)
        try:
            return self._parse_obj(data)
        except OutputParserException as e:
            raise OutputParserException(str(e), llm_output=text) from e


concept_parser = ConceptParser(pydantic_object=FloreaConcept)

CONCEPT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the creative director of an art studio. Answer with ONE valid JSON object and nothing else."),
    ("human", "{inputs}\n\nNATURE KNOWLEDGE:\n{knowledge}\n\nKNOWLEDGE GRAPH (relations):\n{graph}\n\n"
              "Design one artwork. Language of title, concept and symbols: {language}. "
              "Subject, palette and mood MUST be in English.\n{format_instructions}"),
]).partial(format_instructions=concept_parser.get_format_instructions())

POEM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a poet. Write only the poem, in {language}. Original free verse, 8 to 12 lines, concrete natural "
               "images, no title, no explanation, no markdown."),
    ("human", "{context}"),
])
STORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a storyteller. Write only a very short story of 3 or 4 sentences, in {language}, inspired by the "
               "notes. No title, no markdown."),
    ("human", "{context}"),
])


def build_chains(generate_fn):
    """Trois modèles (températures différentes) et les chaînes LCEL  prompt | modèle | parseur."""
    precise = LocalChat(generate_fn=generate_fn, max_new_tokens=450, temperature=0.6)
    poet = LocalChat(generate_fn=generate_fn, max_new_tokens=260, temperature=0.9)
    story = LocalChat(generate_fn=generate_fn, max_new_tokens=200, temperature=0.8)
    return {"concept": CONCEPT_PROMPT | precise | concept_parser,
            "concept_retry": CONCEPT_PROMPT | LocalChat(generate_fn=generate_fn, max_new_tokens=450, temperature=0.2) | concept_parser,
            "poem": POEM_PROMPT | poet | StrOutputParser(),
            "story": STORY_PROMPT | story | StrOutputParser()}


def salvage_concept(raw_text, fallback):
    """Réponse invalide : garde chaque champ qui passe la validation, remplace les autres par le concept de secours.

    Renvoie (FloreaConcept, liste des champs remplacés)."""
    data, fb = extract_json(raw_text or "") or {}, fallback.model_dump()
    merged, fixed = dict(fb), []
    for field in fb:
        if field in data:
            try:
                FloreaConcept.model_validate({**fb, field: data[field]})
                merged[field] = data[field]
                continue
            except ValidationError:
                pass
        fixed.append(field)
    return FloreaConcept.model_validate(merged), fixed


def generate_concept(chains, variables, fallback):
    """1) chaîne normale  2) chaîne à basse température  3) récupération champ par champ. Renvoie (concept, infos)."""
    last = ""
    for attempt, key in enumerate(("concept", "concept_retry"), 1):
        try:
            return chains[key].invoke(variables), {"attempts": attempt, "fixed": [], "llm_ok": True}
        except (OutputParserException, ValidationError, ValueError) as e:
            last = getattr(e, "llm_output", "") or last
    concept, fixed = salvage_concept(last, fallback)
    return concept, {"attempts": 2, "fixed": fixed, "llm_ok": len(fixed) < len(FloreaConcept.model_fields)}


# ----------------------------------------------------------------------------- outils @tool (partagés avec le serveur MCP)
def make_tools(kb, ctx):
    """Outils LangChain. ctx (dict) mémorise ce que l'agent a trouvé : ctx['retrieved'], ctx['calls'], ctx['style_name']."""

    @tool
    def search_knowledge(query: str) -> str:
        """Search the nature knowledge base (flowers, trees, landscapes, weather, sounds, Mediterranean culture).
        Call it with a short English query when you need facts, symbolism or atmosphere."""
        hits = kb.search(query, top_k=4)
        ctx.setdefault("retrieved", [])
        for h in hits:
            if h["document"] not in [d["document"] for d in ctx["retrieved"]]:
                ctx["retrieved"].append(h)
        ctx.setdefault("calls", []).append(f"search_knowledge({query!r}) -> {len(hits)} documents")
        return "\n".join(f"- {h['document']}" for h in hits) or "No result."

    @tool
    def generate_image_prompt(subject: str, palette: str, mood: str) -> str:
        """Build the final text-to-image prompt (style + subject + palette + mood). palette is a comma-separated list."""
        style = ctx.get("style_name")
        colors = normalize_colors(as_list(palette, 8)) or as_list(palette, 5)
        return build_image_prompt(style, subject, colors, mood)

    return {"search_knowledge": search_knowledge, "generate_image_prompt": generate_image_prompt}


AGENT_SYSTEM = ("You are a research assistant for an art studio. Read the request. If it mentions a plant, a landscape, a "
                "weather, a sound or a feeling, call the tool search_knowledge ONCE with a short English query. "
                "If the request is already fully clear you may skip the tool. When done, answer only: DONE.")


def run_research_agent(generate_fn, tools, request_text, max_steps=3):
    """Agent ReAct : le LLM DÉCIDE d'appeler (ou non) search_knowledge. Renvoie la liste des événements (trace)."""
    chat = LocalChat(generate_fn=generate_fn, max_new_tokens=160, temperature=0.2)
    use = [tools["search_knowledge"]]
    trace = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langgraph.prebuilt import create_react_agent
            agent = create_react_agent(chat, use, prompt=AGENT_SYSTEM)
            out = agent.invoke({"messages": [("user", request_text)]}, {"recursion_limit": 2 * max_steps + 2})
        n_tool = sum(1 for m in out["messages"] if isinstance(m, ToolMessage))
        return [f"agent(create_react_agent): {n_tool} appel(s) d'outil"]
    except Exception as e:                       # version de LangGraph différente, ou sortie inattendue du petit LLM
        trace.append(f"agent: create_react_agent indisponible ({type(e).__name__}) -> boucle manuelle")
    msgs = [SystemMessage(AGENT_SYSTEM), HumanMessage(request_text)]
    bound = chat.bind_tools(use)
    by_name = {t.name: t for t in use}
    for _ in range(max_steps):
        ai = bound.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            break
        for c in ai.tool_calls:
            try:
                res = by_name[c["name"]].invoke(c["args"])
            except Exception as e:
                res = f"Erreur: {e}"
            msgs.append(ToolMessage(content=res, tool_call_id=c["id"]))
    trace.append(f"agent(boucle manuelle): {sum(isinstance(m, ToolMessage) for m in msgs)} appel(s) d'outil")
    return trace
