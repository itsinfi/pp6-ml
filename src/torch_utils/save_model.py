import torch
import os

def save_model(model: dict, dataset_name: str, name: str):
    os.makedirs('models', exist_ok=True)
    os.makedirs(f'models/{dataset_name}', exist_ok=True)
    torch.save(model, f'models/{dataset_name}/{name}.pt')