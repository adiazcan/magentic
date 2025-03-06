import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",
    model="gpt-4o",
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())