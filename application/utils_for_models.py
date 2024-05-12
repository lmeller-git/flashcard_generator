from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from torchvision.io import read_image


def load_state_dict_(object = None, file = str):
    MODEL_PATH = Path("models")
    MODEL_SAVE_PATH = MODEL_PATH / file
    object.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
    return object

def load_state_dict_2(object: torch.nn, file: str):
    SAVE_PATH = file 
    object.load_state_dict(torch.load(f = SAVE_PATH))
    return object
    
def save_function(object = None, name = str):
    MODEL_PATH = Path("models")
    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    name = f"models/{name}"

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj= object, f = name)

class ppp_files(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform :
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image / 255, label

def accuracy(model, data, batch_size = 1, shuffle = True, device = "cpu"):
    testing_data = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    num_correct_guesses = 0
    for i, (image, label) in enumerate(testing_data):
        logits = model(image.to(device)).squeeze()
        
        output = torch.softmax(logits, dim=0)
        prediction = output.argmax()
        
        if prediction == label.to(device):
            num_correct_guesses += 1
    return num_correct_guesses / len(testing_data) * 100

