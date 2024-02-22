from pathlib import Path
import torch


def load_state_dict_(object = None, file = str):
    MODEL_PATH = Path("models")
    MODEL_SAVE_PATH = MODEL_PATH / file
    object.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
    return object

def load_state_dict_2(object: torch.nn, file: str):
    SAVE_PATH = file 
    object.load_state_dict(torch.load(f = SAVE_PATH))
    return object