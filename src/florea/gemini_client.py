"""Client Gemini pour FLOREA Lite — sans dépendance GPU."""

import json
import os

from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.8-flash"


def _get_client():
    """Initialise le client uniquement lorsque nécessaire."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY est absente. Configure la variable d'environnement."
        )

    return genai.Client(api_key=api_key)


def _extract_messages(messages):
    """Sépare les instructions système de l'historique conversationnel."""
    system_parts = []
    conversation = []

    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content") or "")

        if role == "system":
            system_parts.append(content)

        elif role == "assistant":
            conversation.append(f"Assistant: {content}")

            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                conversation.append(
                    "Assistant tool call: "
                    + function.get("name", "")
                    + " "
                    + json.dumps(function.get("arguments", {}), ensure_ascii=False)
                )

        elif role == "tool":
            conversation.append(f"Tool result: {content}")

        else:
            conversation.append(f"User: {content}")

    return "\n\n".join(system_parts), "\n\n".join(conversation)


def _convert_tools(tools):
    """Convertit les outils LangChain/OpenAI vers le format Gemini."""
    if not tools:
        return None

    declarations = []

    for item in tools:
        function = item.get("function", item)

        declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters_json_schema=function.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            )
        )

    return [types.Tool(function_declarations=declarations)]


def gemini_generate(
    messages,
    max_new_tokens=400,
    temperature=0.7,
    tools=None,
):
    """
    Fonction compatible avec LocalChat.generate_fn.

    Retourne du texte ou des appels d'outils au format compris par chain.py.
    """
    client = _get_client()

    system_instruction, prompt = _extract_messages(messages)

    if not prompt:
        prompt = "Continue."

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_new_tokens,
    )

    if system_instruction:
        config.system_instruction = system_instruction

    gemini_tools = _convert_tools(tools)

    if gemini_tools:
        config.tools = gemini_tools

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    # Convertir les appels de fonctions Gemini en format reconnu par chain.py.
    function_calls = []

    for candidate in response.candidates or []:
        if not candidate.content or not candidate.content.parts:
            continue

        for part in candidate.content.parts:
            call = part.function_call

            if call and call.name:
                function_calls.append(
                    "<tool_call>"
                    + json.dumps(
                        {
                            "name": call.name,
                            "arguments": dict(call.args or {}),
                        },
                        ensure_ascii=False,
                    )
                    + "</tool_call>"
                )

    if function_calls:
        return "\n".join(function_calls)

    text = response.text

    if not text:
        raise RuntimeError(
            "Gemini a renvoyé une réponse vide. Vérifie les limites et la configuration API."
        )

    return text
