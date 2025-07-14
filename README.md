# Azure ML MCP Server

A Model Context Protocol (MCP) server that provides tools for interacting with Azure Machine Learning resources. This server can be deployed to Azure with managed identity authentication.

## Features

- List Azure ML models, datasets, and compute targets
- Managed identity authentication for Azure deployments
- Fallback to Azure CLI and service principal authentication for local development
- SSE (Server-Sent Events) transport for Azure AI Agents integration
- Health check endpoint for monitoring

## Local Development

### Prerequisites

- Python 3.11+
- Azure CLI (for local authentication)
- Docker (optional, for containerized testing)

### Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the example environment file and configure it:
```bash
cp .env.example .env
# Edit .env with your Azure ML workspace details
```

3. Authenticate with Azure CLI:
```bash
az login
```

4. Run the MCP server:
```bash
python mcp_server.py
```

The server will be available at `http://localhost:8080/sse`

### Docker Development

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

## Azure Deployment

### Option 1: Azure Container Instances (Recommended)

1. Build and push your container image to Azure Container Registry:
```bash
az acr build --registry your-acr-name --image azureml-mcp-server:latest .
```

2. Deploy to ACI with managed identity:
```powershell
.\deploy-aci.ps1 -ResourceGroupName "your-rg" -ContainerGroupName "azureml-mcp" -SubscriptionId "your-sub-id" -AzureMLResourceGroup "your-ml-rg" -AzureMLWorkspaceName "your-workspace"
```

### Option 2: Azure App Service

1. Deploy to App Service:
```powershell
.\deploy-appservice.ps1 -ResourceGroupName "your-rg" -AppServiceName "azureml-mcp-app" -SubscriptionId "your-sub-id" -AzureMLResourceGroup "your-ml-rg" -AzureMLWorkspaceName "your-workspace"
```

2. Configure continuous deployment from your Git repository.

## Azure AI Agents Integration

Once deployed, you can use the MCP server with Azure AI Agents:

```python
agent = project_client.create_agent(
    model="gpt-4",
    name="azureml-agent",
    instructions="You are an Azure ML assistant...",
    tools=[{
        "type": "mcp",
        "server_label": "azureml_mcp",
        "server_url": "https://your-server-url/sse",
        "require_approval": "never"
    }]
)
```

## Managed Identity Setup

For the MCP server to access Azure ML resources using managed identity:

1. **Assign the managed identity** to your container/app service during deployment
2. **Grant permissions** to the managed identity:
```bash
# Get the managed identity principal ID
PRINCIPAL_ID=$(az container show --name azureml-mcp --resource-group your-rg --query identity.principalId -o tsv)

# Assign Contributor role to the Azure ML workspace
az role assignment create \
    --assignee $PRINCIPAL_ID \
    --role "Contributor" \
    --scope "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_ML_RG/providers/Microsoft.MachineLearningServices/workspaces/YOUR_WORKSPACE"
```

## Available Tools

The MCP server provides the following tools:

- `list_azureml_models`: List all models in an Azure ML workspace
- `list_azureml_datasets`: List all datasets in an Azure ML workspace  
- `list_azureml_computes`: List all compute targets in an Azure ML workspace

Each tool requires:
- `subscription_id`: Azure subscription ID
- `resource_group`: Resource group containing the ML workspace
- `workspace_name`: Name of the ML workspace

## Monitoring

- **Health Check**: `GET /health` - Returns server health status
- **Logs**: Check container logs for authentication and operation details

## Troubleshooting

### Authentication Issues

1. **Local Development**: Ensure `az login` is completed
2. **Azure Deployment**: Verify managed identity is assigned and has proper permissions
3. **Service Principal**: Set `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` environment variables

### Connection Issues

1. **Network**: Ensure the server is accessible from Azure AI Agents service
2. **Firewall**: Check that port 8080 is open
3. **SSL**: Use HTTPS endpoints for production deployments

### Common Error Messages

- `"No valid Azure credentials found"`: Authentication setup required
- `"ResourceNotFoundError"`: Check workspace name and permissions
- `"SSE connection error"`: Network connectivity issue

## Security Considerations

- Use managed identity in production
- Never commit service principal credentials to source control
- Run containers as non-root user (already configured)
- Use HTTPS in production deployments
- Regularly rotate any service principal credentials
