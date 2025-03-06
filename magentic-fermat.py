import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import os

async def main() -> None:
    load_dotenv()

    api_key = os.getenv("API_KEY")
    api_version = os.getenv("API_VERSION")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")

    # Define a model client. You can use other model client that implements
    # the `ChatCompletionClient` interface.
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment="gpt-4o",
        model="gpt-4o-2024-05-13",
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )

    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(team.run_stream(task="Provide a different proof for Fermat's Last Theorem"))


asyncio.run(main())