import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from dotenv import load_dotenv

load_dotenv()
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZUREML_RESOURCE_GROUP = os.getenv("AZUREML_RESOURCE_GROUP")
AZUREML_WORKSPACE_NAME = os.getenv("AZUREML_WORKSPACE_NAME")

print(f"Connecting to workspace: {AZUREML_WORKSPACE_NAME}")
print(f"Resource group: {AZUREML_RESOURCE_GROUP}")
print(f"Subscription: {AZURE_SUBSCRIPTION_ID}")

# Create ML client
credential = DefaultAzureCredential()
ml_client = MLClient(credential, AZURE_SUBSCRIPTION_ID, AZUREML_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME)

print("\n" + "="*50)
print("DEBUGGING DATASET DISCOVERY")
print("="*50)

# Method 1: List all data assets
print("\n1. Listing all data assets:")
try:
    datasets = list(ml_client.data.list())
    print(f"Found {len(datasets)} data assets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. Name: {dataset.name}")
        print(f"     Version: {dataset.version}")
        print(f"     Type: {dataset.type}")
        print(f"     Path: {dataset.path}")
        print(f"     Description: {dataset.description}")
        print(f"     Tags: {dataset.tags}")
        print("     ---")
except Exception as e:
    print(f"Error listing data assets: {e}")

# Method 2: Try to get INS_OBJ specifically
print("\n2. Trying to get INS_OBJ dataset specifically:")
try:
    ins_obj = ml_client.data.get("INS_OBJ", label="latest")
    print(f"Found INS_OBJ:")
    print(f"  Name: {ins_obj.name}")
    print(f"  Version: {ins_obj.version}")
    print(f"  Type: {ins_obj.type}")
    print(f"  Path: {ins_obj.path}")
except Exception as e:
    print(f"Error getting INS_OBJ: {e}")

# Method 3: Try different versions
print("\n3. Trying to list all versions of INS_OBJ:")
try:
    versions = list(ml_client.data.list(name="INS_OBJ"))
    print(f"Found {len(versions)} versions of INS_OBJ:")
    for version in versions:
        print(f"  Version {version.version}: {version.name}")
except Exception as e:
    print(f"Error listing INS_OBJ versions: {e}")

# Method 4: Search case-insensitive
print("\n4. Searching for datasets containing 'ins' or 'obj' (case-insensitive):")
try:
    all_datasets = list(ml_client.data.list())
    matching = []
    for dataset in all_datasets:
        if 'ins' in dataset.name.lower() or 'obj' in dataset.name.lower():
            matching.append(dataset)
    
    print(f"Found {len(matching)} datasets with 'ins' or 'obj' in name:")
    for dataset in matching:
        print(f"  - {dataset.name} (v{dataset.version})")
except Exception as e:
    print(f"Error in case-insensitive search: {e}")

print("\n" + "="*50)
print("DEBUGGING COMPLETE")
print("="*50)
