# Azure Container Instances Deployment Script
# This script deploys the MCP server to Azure Container Instances with managed identity

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$ContainerGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$true)]
    [string]$AzureMLResourceGroup,
    
    [Parameter(Mandatory=$true)]
    [string]$AzureMLWorkspaceName,
    
    [Parameter(Mandatory=$true)]
    [string]$Location = "East US",
    
    [Parameter(Mandatory=$false)]
    [string]$ImageName = "azureml-mcp-server:latest"
)

Write-Host "Deploying AzureML MCP Server to Azure Container Instances..."

# Build and push the Docker image to Azure Container Registry (if using ACR)
# Uncomment and modify the following lines if using ACR:
# $ACRName = "your-acr-name"
# az acr build --registry $ACRName --image $ImageName .

# Deploy to Azure Container Instances with managed identity
az container create `
    --resource-group $ResourceGroupName `
    --name $ContainerGroupName `
    --image $ImageName `
    --cpu 1 `
    --memory 1.5 `
    --restart-policy Always `
    --ip-address Public `
    --ports 8080 `
    --os-type Linux `
    --registry-login-server mcpdemoregistry1363.azurecr.io `
    --registry-username mcpdemoregistry1363 `
    --registry-password $(az acr credential show --name mcpdemoregistry1363 --query passwords[0].value --output tsv) `
    --environment-variables `
        AZURE_SUBSCRIPTION_ID=$SubscriptionId `
        AZUREML_RESOURCE_GROUP=$AzureMLResourceGroup `
        AZUREML_WORKSPACE_NAME=$AzureMLWorkspaceName `
    --assign-identity

Write-Host "Getting the public IP address..."
$publicIP = az container show --resource-group $ResourceGroupName --name $ContainerGroupName --query ipAddress.ip --output tsv

Write-Host "MCP Server deployed successfully!"
Write-Host "Public IP: $publicIP"
Write-Host "SSE Endpoint: http://$publicIP:8080/sse"
Write-Host "Health Check: http://$publicIP:8080/health"
