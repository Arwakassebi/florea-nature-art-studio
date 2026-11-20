import json

import pytest

from florea.chain import (build_chains, generate_concept, make_tools, parse_tool_calls, run_research_agent,
                          salvage_concept)
from florea.kb import KnowledgeBase, hash_embed
from florea.text import FloreaConcept

GOOD = {"title": "Le murmure du violet", "concept": "Un champ de lavande où la mémoire revient doucement.",
        "subject": "rows of blooming purple lavender at sunset with small birds", "palette": ["violet", "gold", "cream"],
        "symbols": ["lavande", "oiseaux"], "mood": "gentle longing"}
VARS = {"inputs": "x", "knowledge": "-", "graph": "-", "language": "French"}


def fallback():
    return FloreaConcept(title="Lavande — Nostalgie", concept="Une œuvre inspirée par la lavande et la nostalgie.",
                         subject="a quiet natural landscape in soft light", palette=["soft green", "cream"],
                         symbols=["nature"], mood="nostalgia")


def scripted(*replies):
    it = iter(replies)
    return lambda messages, max_new_tokens, temperature, tools: next(it)


def test_concept_chain_parses_valid_json():
    chains = build_chains(scripted("```json\n" + json.dumps(GOOD) + "\n```"))
    c, info = generate_concept(chains, VARS, fallback())
    assert c.title == GOOD["title"] and info == {"attempts": 1, "fixed": [], "llm_ok": True}


def test_concept_retries_once_then_succeeds():
    chains = build_chains(scripted("désolé, pas de JSON", json.dumps(GOOD)))
    _, info = generate_concept(chains, VARS, fallback())
    assert info["attempts"] == 2 and info["llm_ok"]


def test_concept_salvages_valid_fields_only():
    bad = {**GOOD, "subject": "un champ de lavande au coucher du soleil"}        # sujet en français : refusé
    chains = build_chains(scripted(json.dumps(bad), json.dumps(bad)))
    c, info = generate_concept(chains, VARS, fallback())
    assert c.title == GOOD["title"] and c.subject == fallback().subject
    assert info["fixed"] == ["subject"] and info["llm_ok"]


def test_concept_falls_back_completely_when_llm_is_useless():
    chains = build_chains(scripted("rien", "rien"))
    c, info = generate_concept(chains, VARS, fallback())
    assert c == fallback() and not info["llm_ok"]


def test_salvage_ignores_garbage():
    c, fixed = salvage_concept("n'importe quoi", fallback())
    assert c == fallback() and len(fixed) == 6


def test_parse_tool_calls_both_formats():
    content, calls = parse_tool_calls('Je cherche.\n<tool_call>\n{"name": "search_knowledge", "arguments": {"query": "lavender"}}\n</tool_call>')
    assert content == "Je cherche." and calls[0]["name"] == "search_knowledge" and calls[0]["args"] == {"query": "lavender"}
    _, calls = parse_tool_calls('{"name": "search_knowledge", "arguments": {"query": "rain"}}')
    assert calls[0]["args"] == {"query": "rain"}
    assert parse_tool_calls("DONE") == ("DONE", [])


@pytest.fixture
def kb():
    return KnowledgeBase(embed_fn=hash_embed)


def test_search_tool_records_context(kb):
    ctx = {}
    out = make_tools(kb, ctx)["search_knowledge"].invoke({"query": "lavender purple flower"})
    assert "Lavender" in out and ctx["retrieved"] and ctx["calls"]


def test_image_prompt_tool(kb):
    t = make_tools(kb, {"style_name": "Hokusai — Ukiyo-e"})["generate_image_prompt"]
    p = t.invoke({"subject": "a blue wave under a cherry tree", "palette": "bleu, blanc", "mood": "awe"})
    assert "Hokusai" in p and "blue, white" in p and len(p.split()) <= 60


def test_agent_calls_tool_when_model_decides_to(kb):
    ctx = {}
    tools = make_tools(kb, ctx)
    calls = iter(['<tool_call>{"name": "search_knowledge", "arguments": {"query": "lavender"}}</tool_call>', "DONE"])
    trace = run_research_agent(lambda m, n, t, tools_: next(calls), tools, "Un champ de lavande")
    assert ctx["retrieved"], trace


def test_agent_can_skip_tool(kb):
    ctx = {}
    trace = run_research_agent(lambda m, n, t, tools_: "DONE", make_tools(kb, ctx), "Un champ de lavande")
    assert not ctx.get("retrieved") and trace
