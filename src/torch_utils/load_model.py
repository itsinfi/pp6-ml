import torch

def load_model(dataset_name: str, name: str):
    return torch.load(f'models/{dataset_name}/{name}.pt')