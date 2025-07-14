# Azure App Service Deployment Script
# This script deploys the MCP server to Azure App Service with managed identity

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$AppServiceName,
    
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$true)]
    [string]$AzureMLResourceGroup,
    
    [Parameter(Mandatory=$true)]
    [string]$AzureMLWorkspaceName,
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "East US",
    
    [Parameter(Mandatory=$false)]
    [string]$AppServicePlan = "$AppServiceName-plan"
)

Write-Host "Deploying AzureML MCP Server to Azure App Service..."

# Create App Service Plan
Write-Host "Creating App Service Plan..."
az appservice plan create `
    --name $AppServicePlan `
    --resource-group $ResourceGroupName `
    --location $Location `
    --sku B1 `
    --is-linux

# Create Web App
Write-Host "Creating Web App..."
az webapp create `
    --resource-group $ResourceGroupName `
    --plan $AppServicePlan `
    --name $AppServiceName `
    --deployment-container-image-name "python:3.11-slim"

# Enable managed identity
Write-Host "Enabling managed identity..."
az webapp identity assign --name $AppServiceName --resource-group $ResourceGroupName

# Set environment variables
Write-Host "Setting environment variables..."
az webapp config appsettings set `
    --name $AppServiceName `
    --resource-group $ResourceGroupName `
    --settings `
        AZURE_SUBSCRIPTION_ID=$SubscriptionId `
        AZUREML_RESOURCE_GROUP=$AzureMLResourceGroup `
        AZUREML_WORKSPACE_NAME=$AzureMLWorkspaceName `
        SCM_DO_BUILD_DURING_DEPLOYMENT=true `
        WEBSITES_PORT=8080

# Deploy code (requires git deployment or container registry)
Write-Host "To complete deployment, you need to:"
Write-Host "1. Push your code to a Git repository and configure continuous deployment, OR"
Write-Host "2. Build and push a container image to a registry and update the app"
Write-Host ""

$webAppUrl = az webapp show --name $AppServiceName --resource-group $ResourceGroupName --query defaultHostName --output tsv
Write-Host "Web App URL: https://$webAppUrl"
Write-Host "SSE Endpoint: https://$webAppUrl/sse"
Write-Host "Health Check: https://$webAppUrl/health"
