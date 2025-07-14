import os
import sys
import argparse
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential, ClientSecretCredential, ManagedIdentityCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment, CodeConfiguration, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command, Input, Output
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.server.fastmcp.prompts import base
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse
from dotenv import load_dotenv
import logging
import uvicorn
import tempfile
import json
from pathlib import Path

# Load environment variables
load_dotenv(override=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("AzureML MCP Server")

def get_azure_credential(client_id: Optional[str] = None, client_secret: Optional[str] = None, tenant_id: Optional[str] = None):
    """Get Azure credential based on provided parameters or environment."""
    logger.info("Starting credential detection process...")
    
    # Use provided credentials first, then fall back to environment variables
    client_id = client_id or os.getenv("AZURE_CLIENT_ID")
    client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET") 
    tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID")
    
    if client_id and client_secret and tenant_id:
        logger.info("Using ClientSecretCredential with provided credentials")
        try:
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            # Test the credential
            token = credential.get_token("https://management.azure.com/.default")
            logger.info("ClientSecretCredential authenticated successfully")
            return credential
        except Exception as e:
            logger.error(f"ClientSecretCredential failed: {e}")
    
    # Check if running in Azure environment (managed identity available)
    identity_endpoint = os.getenv("IDENTITY_ENDPOINT")
    msi_endpoint = os.getenv("MSI_ENDPOINT")
    website_site_name = os.getenv("WEBSITE_SITE_NAME")
    
    logger.info(f"Environment check - IDENTITY_ENDPOINT: {identity_endpoint}")
    logger.info(f"Environment check - MSI_ENDPOINT: {msi_endpoint}")
    logger.info(f"Environment check - WEBSITE_SITE_NAME: {website_site_name}")
    
    is_azure_environment = identity_endpoint or msi_endpoint or website_site_name
    
    if is_azure_environment:
        logger.info("Detected Azure environment, attempting ManagedIdentityCredential")
        try:
            credential = ManagedIdentityCredential()
            # Test the credential with different scopes
            try:
                token = credential.get_token("https://management.azure.com/.default")
                logger.info("ManagedIdentityCredential authenticated successfully for Azure Management")
                
                # Also test Azure ML specific scope
                ml_token = credential.get_token("https://ml.azure.com/.default")
                logger.info("ManagedIdentityCredential authenticated successfully for Azure ML")
                return credential
            except Exception as token_e:
                logger.error(f"ManagedIdentityCredential token test failed: {token_e}")
                raise token_e
        except Exception as e:
            logger.error(f"ManagedIdentityCredential failed: {e}")
    
    # For local development or fallback, try DefaultAzureCredential with detailed logging
    logger.info("Attempting DefaultAzureCredential as fallback...")
    try:
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=True,  # Don't prompt in server environment
            exclude_visual_studio_code_credential=False,
            exclude_azure_cli_credential=False,
            exclude_managed_identity_credential=False,
            logging_enable=True  # Enable detailed logging
        )
        # Test the credential by trying to get a token
        token = credential.get_token("https://management.azure.com/.default")
        logger.info("DefaultAzureCredential authenticated successfully")
        return credential
    except Exception as e:
        logger.error(f"DefaultAzureCredential failed: {e}")
        logger.error(f"All credential methods failed. Environment details:")
        logger.error(f"  - Running in container: {os.path.exists('/.dockerenv')}")
        logger.error(f"  - IDENTITY_ENDPOINT: {os.getenv('IDENTITY_ENDPOINT')}")
        logger.error(f"  - MSI_ENDPOINT: {os.getenv('MSI_ENDPOINT')}")
        logger.error(f"  - AZURE_CLIENT_ID: {os.getenv('AZURE_CLIENT_ID')}")
        raise Exception(f"No valid Azure credentials found. Last error: {e}")

def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str, 
                  client_id: Optional[str] = None, client_secret: Optional[str] = None, 
                  tenant_id: Optional[str] = None):
    credential = get_azure_credential(client_id, client_secret, tenant_id)
    return MLClient(credential, subscription_id, resource_group, workspace_name)

def _list_azureml_models(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> list[str]:
    """List all models in the specified AzureML workspace."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name, 
                                  client_id, client_secret, tenant_id)
        models = list(ml_client.models.list())
        return [f"{model.name} (v{model.version})" for model in models]
    except Exception as e:
        return [f"Error: {str(e)}"]

def _list_azureml_datasets(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> list[str]:
    """List all datasets in the specified AzureML workspace."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        logger.info("Attempting to list datasets...")
        datasets = list(ml_client.data.list())
        logger.info(f"Successfully listed {len(datasets)} datasets")
        
        result = []
        for dataset in datasets:
            # Handle cases where version might be None
            version_str = f"v{dataset.version}" if dataset.version else "v?"
            dataset_info = f"{dataset.name} ({version_str})"
            result.append(dataset_info)
            logger.info(f"Found dataset: {dataset_info}")
        
        # Also try to find specific known datasets that might not appear in the general list
        known_datasets = ["INS_OBJ", "ins_obj", "INS-OBJ", "ins-obj", "INSOBJ", "insobj"]
        for dataset_name in known_datasets:
            try:
                # Try different ways to get the dataset
                dataset = None
                # Try with latest label
                try:
                    dataset = ml_client.data.get(dataset_name, label="latest")
                except:
                    # Try without label
                    try:
                        dataset = ml_client.data.get(dataset_name)
                    except:
                        # Try with version 1
                        try:
                            dataset = ml_client.data.get(dataset_name, version="1")
                        except:
                            continue
                
                if dataset:
                    version_str = f"v{dataset.version}" if dataset.version else "v?"
                    dataset_info = f"{dataset.name} ({version_str})"
                    if dataset_info not in result:  # Avoid duplicates
                        result.append(dataset_info)
                        logger.info(f"Found additional dataset: {dataset_info}")
            except Exception as get_e:
                logger.debug(f"Could not find dataset {dataset_name}: {get_e}")
        
        return result
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        # Try to return some common dataset names if listing fails
        error_msg = f"Error: {str(e)}"
        logger.info("Attempting to get specific known datasets...")
        
        # Try to get some known datasets with multiple approaches
        known_datasets = ["INS_OBJ", "ins_obj", "INS-OBJ", "ins-obj", "INSOBJ", "insobj"]
        found_datasets = []
        
        try:
            ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                      client_id, client_secret, tenant_id)
            
            for dataset_name in known_datasets:
                try:
                    # Try different ways to get the dataset
                    dataset = None
                    # Try with latest label
                    try:
                        dataset = ml_client.data.get(dataset_name, label="latest")
                    except:
                        # Try without label
                        try:
                            dataset = ml_client.data.get(dataset_name)
                        except:
                            # Try with version 1
                            try:
                                dataset = ml_client.data.get(dataset_name, version="1")
                            except:
                                continue
                    
                    if dataset:
                        version_str = f"v{dataset.version}" if dataset.version else "v?"
                        dataset_info = f"{dataset.name} ({version_str})"
                        if dataset_info not in found_datasets:  # Avoid duplicates
                            found_datasets.append(dataset_info)
                            logger.info(f"Found known dataset: {dataset_info}")
                except Exception as get_e:
                    logger.debug(f"Could not find dataset {dataset_name}: {get_e}")
        except Exception as client_e:
            logger.error(f"Could not create ML client for fallback search: {client_e}")
        
        if found_datasets:
            return found_datasets + [f"Note: List operation failed, showing known datasets only. Error: {error_msg}"]
        else:
            return [error_msg]

def _list_azureml_computes(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> list[str]:
    """List all compute targets in the specified AzureML workspace."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        computes = list(ml_client.compute.list())
        return [f"{compute.name} (type: {compute.type})" for compute in computes]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.prompt()
def get_initial_prompts() -> list[base.Message]:
    return [
        base.UserMessage("You are a helpful assistant that can help with Azure ML operations. You can list models, datasets, and compute targets from Azure ML workspaces."),
    ]

@mcp.tool()
async def list_azureml_models(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """List all models in the specified AzureML workspace."""
    logger.info(f"Listing models for workspace: {workspace_name}")
    try:
        models = _list_azureml_models(subscription_id, resource_group, workspace_name, 
                                      client_id, client_secret, tenant_id)
        
        if not models:
            return "No models found in the workspace."
        elif models[0].startswith("Error:"):
            return f"Failed to list models: {models[0]}"
        else:
            result = f"Found {len(models)} models in AzureML workspace '{workspace_name}':\n"
            for i, model in enumerate(models[:10], 1):  # Limit to first 10 models
                result += f"{i}. {model}\n"
            if len(models) > 10:
                result += f"... and {len(models) - 10} more models"
            return result
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return f"Failed to list models: {str(e)}"

@mcp.tool()
async def list_azureml_datasets(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """List all datasets in the specified AzureML workspace."""
    logger.info(f"Listing datasets for workspace: {workspace_name}")
    try:
        datasets = _list_azureml_datasets(subscription_id, resource_group, workspace_name,
                                          client_id, client_secret, tenant_id)
        
        if not datasets:
            return "No datasets found in the workspace."
        elif datasets[0].startswith("Error:"):
            return f"Failed to list datasets: {datasets[0]}"
        else:
            result = f"Found {len(datasets)} datasets in AzureML workspace '{workspace_name}':\n"
            for i, dataset in enumerate(datasets[:10], 1):  # Limit to first 10 datasets
                result += f"{i}. {dataset}\n"
            if len(datasets) > 10:
                result += f"... and {len(datasets) - 10} more datasets"
            return result
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return f"Failed to list datasets: {str(e)}"

@mcp.tool()
async def list_azureml_computes(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """List all compute targets in the specified AzureML workspace."""
    logger.info(f"Listing compute targets for workspace: {workspace_name}")
    try:
        computes = _list_azureml_computes(subscription_id, resource_group, workspace_name,
                                          client_id, client_secret, tenant_id)
        
        if not computes:
            return "No compute targets found in the workspace."
        elif computes[0].startswith("Error:"):
            return f"Failed to list compute targets: {computes[0]}"
        else:
            result = f"Found {len(computes)} compute targets in AzureML workspace '{workspace_name}':\n"
            for i, compute in enumerate(computes[:10], 1):  # Limit to first 10 computes
                result += f"{i}. {compute}\n"
            if len(computes) > 10:
                result += f"... and {len(computes) - 10} more compute targets"
            return result
    except Exception as e:
        logger.error(f"Error listing compute targets: {e}")
        return f"Failed to list compute targets: {str(e)}"

# Helper functions for advanced operations
def _start_compute(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    compute_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """Start a compute instance or cluster."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        # Get compute details
        compute = ml_client.compute.get(compute_name)
        
        if hasattr(compute, 'state') and compute.state.lower() in ['stopped', 'deallocated']:
            ml_client.compute.begin_start(compute_name)
            return f"Started compute '{compute_name}'. It may take a few minutes to become available."
        else:
            return f"Compute '{compute_name}' is already running or in transition state: {getattr(compute, 'state', 'unknown')}"
            
    except Exception as e:
        return f"Error starting compute: {str(e)}"

def _stop_compute(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    compute_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """Stop a compute instance or cluster."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        # Get compute details
        compute = ml_client.compute.get(compute_name)
        
        if hasattr(compute, 'state') and compute.state.lower() in ['running', 'idle']:
            ml_client.compute.begin_stop(compute_name)
            return f"Stopped compute '{compute_name}'. This may take a few minutes."
        else:
            return f"Compute '{compute_name}' is not in a running state: {getattr(compute, 'state', 'unknown')}"
            
    except Exception as e:
        return f"Error stopping compute: {str(e)}"

def _get_compute_details(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    compute_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get detailed information about a specific compute."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        compute = ml_client.compute.get(compute_name)
        
        details = {
            "name": compute.name,
            "type": compute.type,
            "state": getattr(compute, 'state', 'unknown'),
            "size": getattr(compute, 'size', 'unknown'),
            "location": getattr(compute, 'location', 'unknown'),
            "provisioning_state": getattr(compute, 'provisioning_state', 'unknown')
        }
        
        # Add specific details based on compute type
        if hasattr(compute, 'min_instances'):
            details["min_instances"] = compute.min_instances
        if hasattr(compute, 'max_instances'):
            details["max_instances"] = compute.max_instances
        if hasattr(compute, 'idle_time_before_scale_down'):
            details["idle_time_before_scale_down"] = compute.idle_time_before_scale_down
            
        return details
        
    except Exception as e:
        return {"error": str(e)}

def _submit_command_job(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    compute_name: str,
    script_content: str,
    environment: str = "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    display_name: Optional[str] = None,
    experiment_name: str = "mcp-experiments",
    inputs: Optional[Dict[str, str]] = None,
    outputs: Optional[Dict[str, str]] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """Submit a command job with a Python script."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        # Create a temporary directory for the script
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "script.py"
            script_path.write_text(script_content)
            
            # Prepare inputs and outputs
            job_inputs = {}
            job_outputs = {}
            
            if inputs:
                for name, path in inputs.items():
                    job_inputs[name] = Input(type=AssetTypes.URI_FOLDER, path=path)
                    
            if outputs:
                for name, path in outputs.items():
                    job_outputs[name] = Output(type=AssetTypes.URI_FOLDER, path=path)
            
            # Create the command job
            job = command(
                code=temp_dir,
                command="python script.py",
                environment=environment,
                compute=compute_name,
                display_name=display_name or f"MCP Generated Job - {script_content[:50]}...",
                experiment_name=experiment_name,
                inputs=job_inputs if job_inputs else None,
                outputs=job_outputs if job_outputs else None
            )
            
            # Submit the job
            submitted_job = ml_client.create_or_update(job)
            
            return {
                "job_id": submitted_job.name,
                "status": submitted_job.status,
                "display_name": submitted_job.display_name,
                "experiment_name": submitted_job.experiment_name,
                "compute": submitted_job.compute,
                "studio_url": submitted_job.studio_url
            }
            
    except Exception as e:
        return {"error": str(e)}

def _get_job_status(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    job_id: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get the status and details of a job."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        job = ml_client.jobs.get(job_id)
        
        return {
            "job_id": job.name,
            "status": job.status,
            "display_name": job.display_name,
            "experiment_name": job.experiment_name,
            "compute": job.compute,
            "creation_context": {
                "created_at": str(job.creation_context.created_at) if job.creation_context else None,
                "created_by": job.creation_context.created_by if job.creation_context else None
            },
            "studio_url": job.studio_url
        }
        
    except Exception as e:
        return {"error": str(e)}

def _list_recent_jobs(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    limit: int = 10,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> list[Dict[str, Any]]:
    """List recent jobs in the workspace."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        jobs = list(ml_client.jobs.list(max_results=limit))
        
        result = []
        for job in jobs:
            result.append({
                "job_id": job.name,
                "status": job.status,
                "display_name": job.display_name,
                "experiment_name": job.experiment_name,
                "compute": job.compute,
                "created_at": str(job.creation_context.created_at) if job.creation_context else None
            })
            
        return result
        
    except Exception as e:
        return [{"error": str(e)}]

def _get_dataset_info(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    dataset_name: str,
    version: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get detailed information about a registered dataset."""
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        if version:
            dataset = ml_client.data.get(dataset_name, version=version)
        else:
            dataset = ml_client.data.get(dataset_name, label="latest")
        
        return {
            "name": dataset.name,
            "version": dataset.version,
            "type": dataset.type,
            "path": dataset.path,
            "description": dataset.description,
            "tags": dataset.tags,
            "properties": dataset.properties
        }
        
    except Exception as e:
        return {"error": str(e)}

def _create_training_script(
    task_type: str = "classification",
    dataset_type: str = "csv",
    model_type: str = "sklearn",
    include_preprocessing: bool = True,
    include_model_saving: bool = True
) -> str:
    """Generate a Python training script template."""
    script_parts = []
    
    # Imports
    script_parts.append("import pandas as pd")
    script_parts.append("import numpy as np")
    script_parts.append("from pathlib import Path")
    script_parts.append("import joblib")
    script_parts.append("import argparse")
    script_parts.append("")
    
    if model_type == "sklearn":
        if task_type == "classification":
            script_parts.append("from sklearn.ensemble import RandomForestClassifier")
            script_parts.append("from sklearn.metrics import accuracy_score, classification_report")
        else:  # regression
            script_parts.append("from sklearn.ensemble import RandomForestRegressor")
            script_parts.append("from sklearn.metrics import mean_squared_error, r2_score")
        
        if include_preprocessing:
            script_parts.append("from sklearn.preprocessing import StandardScaler, LabelEncoder")
            script_parts.append("from sklearn.model_selection import train_test_split")
    
    script_parts.append("")
    script_parts.append("def main():")
    script_parts.append("    parser = argparse.ArgumentParser()")
    script_parts.append("    parser.add_argument('--data-path', type=str, help='Path to the dataset')")
    script_parts.append("    parser.add_argument('--output-path', type=str, help='Path to save outputs')")
    script_parts.append("    args = parser.parse_args()")
    script_parts.append("")
    script_parts.append("    # Load data")
    
    if dataset_type == "csv":
        script_parts.append("    data = pd.read_csv(args.data_path)")
    else:
        script_parts.append("    # TODO: Implement data loading for your specific format")
        script_parts.append("    data = pd.read_csv(args.data_path)  # Modify as needed")
    
    script_parts.append("")
    script_parts.append("    # TODO: Update these column names to match your dataset")
    script_parts.append("    feature_columns = data.columns[:-1]  # All columns except last")
    script_parts.append("    target_column = data.columns[-1]     # Last column as target")
    script_parts.append("")
    script_parts.append("    X = data[feature_columns]")
    script_parts.append("    y = data[target_column]")
    script_parts.append("")
    
    if include_preprocessing:
        script_parts.append("    # Preprocessing")
        script_parts.append("    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
        script_parts.append("")
        script_parts.append("    # Scale features")
        script_parts.append("    scaler = StandardScaler()")
        script_parts.append("    X_train_scaled = scaler.fit_transform(X_train)")
        script_parts.append("    X_test_scaled = scaler.transform(X_test)")
        script_parts.append("")
    else:
        script_parts.append("    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
        script_parts.append("    X_train_scaled = X_train")
        script_parts.append("    X_test_scaled = X_test")
        script_parts.append("")
    
    # Model training
    if model_type == "sklearn":
        if task_type == "classification":
            script_parts.append("    # Train model")
            script_parts.append("    model = RandomForestClassifier(n_estimators=100, random_state=42)")
        else:
            script_parts.append("    # Train model")
            script_parts.append("    model = RandomForestRegressor(n_estimators=100, random_state=42)")
    
    script_parts.append("    model.fit(X_train_scaled, y_train)")
    script_parts.append("")
    script_parts.append("    # Make predictions")
    script_parts.append("    y_pred = model.predict(X_test_scaled)")
    script_parts.append("")
    script_parts.append("    # Evaluate model")
    
    if task_type == "classification":
        script_parts.append("    accuracy = accuracy_score(y_test, y_pred)")
        script_parts.append("    print(f'Accuracy: {accuracy:.4f}')")
        script_parts.append("    print('Classification Report:')")
        script_parts.append("    print(classification_report(y_test, y_pred))")
    else:
        script_parts.append("    mse = mean_squared_error(y_test, y_pred)")
        script_parts.append("    r2 = r2_score(y_test, y_pred)")
        script_parts.append("    print(f'Mean Squared Error: {mse:.4f}')")
        script_parts.append("    print(f'R² Score: {r2:.4f}')")
    
    if include_model_saving:
        script_parts.append("")
        script_parts.append("    # Save model")
        script_parts.append("    if args.output_path:")
        script_parts.append("        output_path = Path(args.output_path)")
        script_parts.append("        output_path.mkdir(parents=True, exist_ok=True)")
        script_parts.append("        model_path = output_path / 'model.joblib'")
        script_parts.append("        joblib.dump(model, model_path)")
        script_parts.append("        print(f'Model saved to: {model_path}')")
    
    script_parts.append("")
    script_parts.append("if __name__ == '__main__':")
    script_parts.append("    main()")
    
    return "\n".join(script_parts)

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages")

    async def handle_sse(request: Request):
        logger.info(f"SSE connection from {request.client.host if request.client else 'unknown'}")
        
        try:
            async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                logger.info("SSE connection established, starting MCP server")
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
        except Exception as e:
            logger.error(f"SSE connection error: {e}", exc_info=True)
            raise

    async def handle_messages(request: Request):
        """Handle POST messages to /messages"""
        logger.info("Handling POST message to /messages")
        return await sse.handle_post_message(request.scope, request.receive, request._send)

    async def health_check(request: Request):
        """Simple health check endpoint"""
        return JSONResponse({"status": "healthy", "service": "azureml-mcp-server"})

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
            Route("/health", endpoint=health_check, methods=["GET"]),
        ],
    )

# Simplified tool for executing code as Azure ML jobs
@mcp.tool()
async def execute_code_job(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    code: str,
    dataset_name: Optional[str] = None,
    compute_name: Optional[str] = None,
    environment: str = "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    display_name: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """Execute Python code as an Azure ML job with optional dataset.
    
    This is the main tool for running code in Azure ML. It will:
    1. Take your Python code
    2. Optionally use a specified dataset
    3. Submit it as a job to Azure ML
    4. Return the link to view the run
    """
    logger.info(f"Starting execute_code_job - workspace: {workspace_name}, dataset: {dataset_name}")
    
    try:
        # Test credentials first
        logger.info("Testing Azure credentials...")
        credential = get_azure_credential(client_id, client_secret, tenant_id)
        logger.info("Credentials obtained successfully")
        
        # Create ML client with detailed logging
        logger.info(f"Creating ML client for subscription: {subscription_id}")
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        
        # Test workspace access
        logger.info("Testing workspace access...")
        try:
            workspace = ml_client.workspaces.get(workspace_name)
            logger.info(f"Workspace access successful: {workspace.name}")
        except Exception as ws_e:
            logger.error(f"Workspace access failed: {ws_e}")
            return f"❌ Cannot access workspace '{workspace_name}': {str(ws_e)}"
        
        # Auto-select compute if not provided
        if not compute_name:
            logger.info("Auto-selecting compute target...")
            try:
                computes = list(ml_client.compute.list())
                logger.info(f"Found {len(computes)} compute targets")
                
                running_computes = [c for c in computes if hasattr(c, 'state') and c.state.lower() in ['running', 'idle']]
                
                if running_computes:
                    compute_name = running_computes[0].name
                    logger.info(f"Auto-selected running compute: {compute_name}")
                elif computes:
                    compute_name = computes[0].name
                    logger.info(f"Auto-selected compute: {compute_name}")
                else:
                    return "❌ No compute targets found in workspace. Please create a compute target first."
            except Exception as compute_e:
                logger.error(f"Error listing compute targets: {compute_e}")
                return f"❌ Cannot list compute targets: {str(compute_e)}"
        
        # Test compute access
        logger.info(f"Testing access to compute target: {compute_name}")
        try:
            compute = ml_client.compute.get(compute_name)
            logger.info(f"Compute access successful: {compute.name} (state: {getattr(compute, 'state', 'unknown')})")
        except Exception as comp_e:
            logger.error(f"Compute access failed: {comp_e}")
            return f"❌ Cannot access compute '{compute_name}': {str(comp_e)}"
        
        # Prepare inputs if dataset is specified
        inputs = {}
        script_with_data = code
        
        if dataset_name:
            logger.info(f"Processing dataset: {dataset_name}")
            try:
                dataset = ml_client.data.get(dataset_name, label="latest")
                logger.info(f"Dataset loaded successfully: {dataset.name} v{dataset.version}")
                logger.info(f"Dataset type: {dataset.type}, Dataset path: {dataset.path}")
                
                # Determine the correct input type based on dataset type
                if hasattr(dataset, 'type') and dataset.type:
                    if dataset.type.lower() == 'uri_file':
                        input_type = AssetTypes.URI_FILE
                        logger.info("Using URI_FILE type for dataset")
                    elif dataset.type.lower() == 'uri_folder':
                        input_type = AssetTypes.URI_FOLDER
                        logger.info("Using URI_FOLDER type for dataset")
                    else:
                        # Default to URI_FOLDER for backwards compatibility
                        input_type = AssetTypes.URI_FOLDER
                        logger.info(f"Unknown dataset type '{dataset.type}', defaulting to URI_FOLDER")
                else:
                    input_type = AssetTypes.URI_FOLDER
                    logger.info("Dataset type not specified, defaulting to URI_FOLDER")
                
                inputs["input_data"] = Input(type=input_type, path=dataset.path)
                
                # Modify the script to include data loading
                data_loading_code = f"""
# Auto-generated data loading for dataset: {dataset_name}
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, help='Path to input data')
args = parser.parse_args()

# Load dataset
import pandas as pd
from pathlib import Path
if args.input_data:
    data_path = Path(args.input_data)
    print(f"Dataset path: {{data_path}}")
    print(f"Dataset path type: {{type(data_path)}}")
    print(f"Dataset path exists: {{data_path.exists()}}")
    
    if data_path.is_dir():
        print("Dataset is a directory, looking for files...")
        csv_files = list(data_path.glob('*.csv'))
        parquet_files = list(data_path.glob('*.parquet'))
        json_files = list(data_path.glob('*.json'))
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            print(f"Loaded CSV dataset {{dataset_name}} with shape: {{df.shape}}")
        elif parquet_files:
            df = pd.read_parquet(parquet_files[0])
            print(f"Loaded Parquet dataset {{dataset_name}} with shape: {{df.shape}}")
        elif json_files:
            df = pd.read_json(json_files[0])
            print(f"Loaded JSON dataset {{dataset_name}} with shape: {{df.shape}}")
        else:
            print("No supported files found in dataset directory")
            # List all files for debugging
            all_files = list(data_path.rglob('*'))
            print(f"All files in dataset: {{[f.name for f in all_files[:10]]}}")
    elif data_path.is_file():
        print("Dataset is a file, loading directly...")
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
            print(f"Loaded CSV dataset {{dataset_name}} with shape: {{df.shape}}")
        elif data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
            print(f"Loaded Parquet dataset {{dataset_name}} with shape: {{df.shape}}")
        elif data_path.suffix.lower() == '.json':
            df = pd.read_json(data_path)
            print(f"Loaded JSON dataset {{dataset_name}} with shape: {{df.shape}}")
        else:
            print(f"Unsupported file type: {{data_path.suffix}}")
    else:
        print(f"Dataset path does not exist or is not accessible: {{data_path}}")

# Your code starts here:
"""
                script_with_data = data_loading_code + "\n" + code
                
            except Exception as dataset_e:
                logger.warning(f"Could not load dataset {dataset_name}: {dataset_e}")
                return f"⚠️  Warning: Could not load dataset '{dataset_name}': {str(dataset_e)}. Continuing without dataset."
        
        # Submit the job with detailed error handling
        logger.info("Submitting job to Azure ML...")
        try:
            result = _submit_command_job(
                subscription_id, resource_group, workspace_name, compute_name,
                script_with_data, environment, 
                display_name or f"Code Execution Job - {dataset_name or 'No Dataset'}",
                "code-execution-experiments",
                inputs if inputs else None,
                None,  # outputs
                client_id, client_secret, tenant_id
            )
            
            if "error" in result:
                logger.error(f"Job submission failed: {result['error']}")
                return f"❌ Error submitting job: {result['error']}"
            
            logger.info(f"Job submitted successfully: {result['job_id']}")
            
            response = f"✅ Code execution job submitted successfully!\n\n"
            response += f"📊 **Job Details:**\n"
            response += f"• Job ID: `{result['job_id']}`\n"
            response += f"• Status: {result['status']}\n"
            response += f"• Compute: {result['compute']}\n"
            response += f"• Environment: {environment}\n"
            
            if dataset_name:
                response += f"• Dataset: {dataset_name}\n"
            
            response += f"\n🔗 **View Job Run:** {result['studio_url']}\n"
            response += f"\n💡 Use this link to monitor your job progress and view results in Azure ML Studio."
            
            return response
            
        except Exception as submit_e:
            logger.error(f"Job submission exception: {submit_e}")
            return f"❌ Failed to submit job: {str(submit_e)}"
        
    except Exception as e:
        logger.error(f"Error in execute_code_job: {e}", exc_info=True)
        return f"❌ Failed to execute code job: {str(e)}"


@mcp.tool()
async def search_dataset(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    dataset_name: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> str:
    """Search for a specific dataset by name in the AzureML workspace."""
    logger.info(f"Searching for dataset: {dataset_name} in workspace: {workspace_name}")
    try:
        ml_client = get_ml_client(subscription_id, resource_group, workspace_name,
                                  client_id, client_secret, tenant_id)
        
        # Try multiple variations of the dataset name
        name_variations = [
            dataset_name,
            dataset_name.upper(),
            dataset_name.lower(),
            dataset_name.replace("_", "-"),
            dataset_name.replace("-", "_"),
            dataset_name.replace(" ", "_"),
            dataset_name.replace(" ", "-")
        ]
        
        found_datasets = []
        
        for name_var in name_variations:
            try:
                # Try different ways to get the dataset
                dataset = None
                # Try with latest label
                try:
                    dataset = ml_client.data.get(name_var, label="latest")
                except:
                    # Try without label
                    try:
                        dataset = ml_client.data.get(name_var)
                    except:
                        # Try with version 1
                        try:
                            dataset = ml_client.data.get(name_var, version="1")
                        except:
                            continue
                
                if dataset:
                    version_str = f"v{dataset.version}" if dataset.version else "v?"
                    dataset_info = {
                        'name': dataset.name,
                        'version': version_str,
                        'description': getattr(dataset, 'description', 'No description'),
                        'type': getattr(dataset, 'type', 'Unknown'),
                        'path': getattr(dataset, 'path', 'No path info')
                    }
                    # Check if we already found this dataset
                    if not any(d['name'] == dataset.name for d in found_datasets):
                        found_datasets.append(dataset_info)
                        logger.info(f"Found dataset: {dataset.name} ({version_str})")
            except Exception as get_e:
                logger.debug(f"Could not find dataset variant {name_var}: {get_e}")
        
        if found_datasets:
            result = f"Found {len(found_datasets)} dataset(s) matching '{dataset_name}':\n"
            for dataset in found_datasets:
                result += f"- {dataset['name']} ({dataset['version']})\n"
                result += f"  Type: {dataset['type']}\n"
                result += f"  Description: {dataset['description']}\n"
                result += f"  Path: {dataset['path']}\n"
            return result
        else:
            return f"No dataset found matching '{dataset_name}'. Try using list_azureml_datasets to see all available datasets."
            
    except Exception as e:
        logger.error(f"Error searching for dataset {dataset_name}: {e}")
        return f"Failed to search for dataset '{dataset_name}': {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AzureML MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Get the MCP server instance
    mcp_server = mcp._mcp_server  # noqa: WPS437

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    logger.info(f"Starting AzureML MCP Server on {args.host}:{args.port}")
    logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    logger.info("🚀 Key tool: execute_code_job - Execute Python code as Azure ML jobs")
    logger.info("📊 Other tools: list_azureml_models, list_azureml_datasets, list_azureml_computes, get_job_status")
    
    uvicorn.run(starlette_app, host=args.host, port=args.port)
