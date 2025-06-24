import numpy as np
import pandas as pd
import torch

def load_embeddings(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv"):
        return pd.read_csv(path).values
    if path.endswith(".pt"):
        return torch.load(path).cpu().numpy()
    raise ValueError(f"Unsupported file type: {path}")