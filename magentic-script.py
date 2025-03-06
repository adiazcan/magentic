import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import os

async def example_usage():

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

    m1 = MagenticOne(client=model_client)
    task = "Write a Python script to fetch data from an API."
    result = await Console(m1.run_stream(task=task))
    print(result)

if __name__ == "__main__":
    asyncio.run(example_usage())