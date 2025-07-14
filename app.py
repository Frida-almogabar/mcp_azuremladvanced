import os
import azure.ai.agents as agentslib
from azure.ai.agents.models import (
    FunctionTool,
    ToolSet,
    MessageRole,
)
from azure.ai.agents import AgentsClient

from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


load_dotenv()
connection_string = os.getenv("PROJECT_CONNECTION_STRING")
deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
ENDPOINT = os.getenv("ENDPOINT")
instructions = "You are a helpful assistant that can answer my questions with help of your tools"
project_client = AgentsClient(
    credential=DefaultAzureCredential(),
    endpoint=ENDPOINT,
)

user_prompt= "use mcp tool and what are available github docs?"

agent_name = "Git_Agent"

found_agent = None
all_agents_list = project_client.list_agents()
for a in all_agents_list:
    if a.name == agent_name:
        found_agent = a
        break

if found_agent:
    print(f"reusing agent > {found_agent.name} (id: {found_agent.id})")
else:
    with project_client:
        agent = project_client.create_agent(
                model=deployment,
                name=agent_name,
                instructions=instructions,
                tools=[{
                    "type": "mcp",
                    "server_label": "product_info_mcp",
                    "server_url": "https://gitmcp.io/Azure/azure-rest-api-specs",
                    "require_approval": "never"
                }],
                tool_resources=None,
            )
        print(f"Created agent '{agent_name}' \nID: {agent.id}")
        print(f"Project client API version: {project_client._config.api_version}")
        print(f"Azure-ai-agents version: {agentslib.__version__}")



    thread = project_client.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")

    msg = project_client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=user_prompt
    )

    
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    print(f"Run ID: {run.id}")

    print("Messages in the thread:")