"""Client MCP de démonstration : lance florea_server.py (stdio), liste les outils et en appelle deux."""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demo(server_path):
    params = StdioServerParameters(command=sys.executable, args=[server_path])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("Outils exposés :", names)
            r1 = await session.call_tool("recommend_styles", {"entities": "lavender", "emotion": "Sérénité", "season": "summer", "k": 3})
            print("recommend_styles ->", r1.content[0].text[:300])
            r2 = await session.call_tool("generate_image_prompt", {"style_name": "Hokusai — Ukiyo-e", "subject": "a blue wave under a cherry tree", "palette": "bleu, blanc", "mood": "awe"})
            print("generate_image_prompt ->", r2.content[0].text[:300])
            return names


def run(server_path):
    try:
        return asyncio.run(demo(server_path))
    except RuntimeError:                       # boucle déjà active (notebook)
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(demo(server_path))


if __name__ == "__main__":
    run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "florea_server.py"))
