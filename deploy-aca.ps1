# Azure Container Apps Deployment Script
# This script deploys the MCP server to Azure Container Apps with HTTPS support

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,

    [Parameter(Mandatory=$true)]
    [string]$ContainerAppName,

    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,

    [Parameter(Mandatory=$true)]
    [string]$AzureMLResourceGroup,

    [Parameter(Mandatory=$true)]
    [string]$AzureMLWorkspaceName,

    [Parameter(Mandatory=$true)]
    [string]$Location = "East US",

    [Parameter(Mandatory=$false)]
    [string]$ImageName = "mcpdemoregistry1363.azurecr.io/mcpdemo:latest"
)

Write-Host "Deploying AzureML MCP Server to Azure Container Apps..."

# Create a Container Apps environment
$envName = "$ContainerAppName-env"
Write-Host "Creating Container Apps environment: $envName"

az containerapp env create `
    --name $envName `
    --resource-group $ResourceGroupName `
    --location $Location

# Deploy the container app with managed identity
Write-Host "Deploying container app: $ContainerAppName"

az containerapp create `
    --name $ContainerAppName `
    --resource-group $ResourceGroupName `
    --environment $envName `
    --image $ImageName `
    --target-port 8080 `
    --ingress external `
    --cpu 1.0 `
    --memory 2.0Gi `
    --registry-server mcpdemoregistry1363.azurecr.io `
    --registry-username mcpdemoregistry1363 `
    --registry-password $(az acr credential show --name mcpdemoregistry1363 --query passwords[0].value --output tsv) `
    --env-vars `
        AZURE_SUBSCRIPTION_ID=$SubscriptionId `
        AZUREML_RESOURCE_GROUP=$AzureMLResourceGroup `
        AZUREML_WORKSPACE_NAME=$AzureMLWorkspaceName `
    --system-assigned

Write-Host "Getting the FQDN..."
$fqdn = az containerapp show --name $ContainerAppName --resource-group $ResourceGroupName --query properties.configuration.ingress.fqdn --output tsv

Write-Host "MCP Server deployed successfully to Azure Container Apps!"
Write-Host "FQDN: $fqdn"
Write-Host "SSE Endpoint: https://$fqdn/sse"
Write-Host "Health Check: https://$fqdn/health"

# Get the managed identity principal ID for role assignment
$principalId = az containerapp show --name $ContainerAppName --resource-group $ResourceGroupName --query identity.principalId --output tsv
Write-Host "Managed Identity Principal ID: $principalId"

Write-Host "Assigning AzureML Data Scientist role to the managed identity..."
az role assignment create `
    --assignee $principalId `
    --role "AzureML Data Scientist" `
    --scope "/subscriptions/$SubscriptionId/resourceGroups/$AzureMLResourceGroup/providers/Microsoft.MachineLearningServices/workspaces/$AzureMLWorkspaceName"

Write-Host "✅ Deployment complete!"
