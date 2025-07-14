from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage
from azure.identity import DefaultAzureCredential




model = AzureAIChatCompletionsModel(
    endpoint="https://fasazureaihub5942341532.services.ai.azure.com/openai/v1",
    credential=DefaultAzureCredential(),
    model_name="gpt-4o"
)

messages = [
    SystemMessage(
      content="Translate the following from English into Italian"
    ),
    HumanMessage(content="hi!"),
]

model.invoke(messages)

print(model.invoke(messages))