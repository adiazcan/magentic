import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

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

    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )

    file_surfer = FileSurfer( "FileSurfer",model_client=model_client)
    coder = MagenticOneCoderAgent("Coder",model_client=model_client)
    terminal = CodeExecutorAgent("ComputerTerminal",code_executor=LocalCommandLineCodeExecutor())

    team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=model_client)
    await Console(team.run_stream(task="What is the UV index in Melbourne today?"))

asyncio.run(main())