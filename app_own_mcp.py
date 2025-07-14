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
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZUREML_RESOURCE_GROUP = os.getenv("AZUREML_RESOURCE_GROUP")
AZUREML_WORKSPACE_NAME = os.getenv("AZUREML_WORKSPACE_NAME")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")


connection_string = os.getenv("PROJECT_CONNECTION_STRING")
deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
ENDPOINT = os.getenv("ENDPOINT")

# Updated instructions to focus on the new execute_code_job tool
instructions = f"""You are an Azure ML code executor assistant. Your main job is to use the execute_code_job tool to run Python code as Azure ML jobs.

Always use these parameters:
- subscription_id: {AZURE_SUBSCRIPTION_ID}
- resource_group: {AZUREML_RESOURCE_GROUP}  
- workspace_name: {AZUREML_WORKSPACE_NAME}

The MCP server will handle authentication automatically using managed identity or available credentials.
Use the available MCP tools to query Azure ML resources like models, datasets, and compute targets.

Always provide clear, helpful responses with job details and studio links."""

project_client = AgentsClient(
    credential=DefaultAzureCredential(),
    endpoint=ENDPOINT,
)

# Test the new execute_code_job tool with enhanced dataset discovery
user_prompt = f"""Please do the following:

1. First, use the list_azureml_datasets tool to show all available datasets in the workspace
2. Then, use the search_dataset tool to specifically search for "INS_OBJ" dataset
3. Finally, try to execute this Python code with the INS_OBJ dataset:

```python
print("🚀 Starting ML Training Job with INS_OBJ dataset...")
print("Dataset loaded and ready for processing!")
print("This is a test to verify INS_OBJ dataset access with enhanced discovery!")
import pandas as pd
print("Pandas imported successfully!")
print("Job completed successfully - INS_OBJ dataset access verified!")
```

Parameters to use:
- subscription_id: {AZURE_SUBSCRIPTION_ID}
- resource_group: {AZUREML_RESOURCE_GROUP}
- workspace_name: {AZUREML_WORKSPACE_NAME}
- compute_name: "FAS-cluster"
- dataset_name: "INS_OBJ"
- display_name: "MCP Test with INS_OBJ Dataset"

Please execute this code and return the Azure ML studio link to view the run."""

print(f"Using following parameters for Azure ML tools:\n{user_prompt}")

agent_name = "AzureMLAgentJobs_v2"

print(f"Looking for existing agent: {agent_name}")
found_agent = None

print("Listing existing agents...")
all_agents_list = list(project_client.list_agents())
print(f"Found {len(all_agents_list)} existing agents")
for a in all_agents_list:
    print(f"  - {a.name} (id: {a.id})")
    if a.name == agent_name:
        found_agent = a
        break

with project_client:
    if found_agent:
        print(f"reusing agent > {found_agent.name} (id: {found_agent.id})")
        agent = found_agent
    else:
        print("Creating new agent with MCP server...")
        print(f"MCP Server URL: https://azureml-mcp-server.mangopond-3e5c0fa1.eastus.azurecontainerapps.io/sse")
        agent = project_client.create_agent(
                model=deployment,
                name=agent_name,
                instructions=instructions,
                tools=[{
                    "type": "mcp",
                    "server_label": "azureml_mcp_https",
                    "server_url": "https://azureml-mcp-server.mangopond-3e5c0fa1.eastus.azurecontainerapps.io/sse",
                    "require_approval": "never"
                }],
                tool_resources=None,
            )
        print(f"Created agent '{agent_name}' \nID: {agent.id}")
        print(f"Project client API version: {project_client._config.api_version}")
        print(f"Azure-ai-agents version: {agentslib.__version__}")

    thread = project_client.threads.create()
    print(f"Created thread, ID: {thread.id}")

    msg = project_client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=user_prompt
    )

    run = project_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
        print("\nThis might be due to:")
        print("1. Network timeout between Azure AI and your MCP server")
        print("2. Response format issues") 
        print("3. Azure AI Agent service temporary issues")
        print("\nYour MCP server logs should show if the connection was made successfully.")
    elif run.status == "completed":
        print("✅ Run completed successfully!")

    print(f"Run ID: {run.id}")

    print("\nMessages in the thread:")
    messages = project_client.messages.list(thread_id=thread.id)
    for message in reversed(list(messages)):  # Show in chronological order
        print(f"\n{message.role}:")
        for content in message.content:
            if hasattr(content, 'text') and hasattr(content.text, 'value'):
                print(f"  {content.text.value}")
            else:
                print(f"  {content}")