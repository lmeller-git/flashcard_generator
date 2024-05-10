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

def training_loop_2(epochs: int, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.nn.Module, train_data: torch.Tensor, test_data: torch.Tensor, device = "cpu", data_loader = DataLoader, batch_size: int = 1, shuffle = True, output = None, performance_tracker = None, tracked_performance: str = None, input_to_performance_tracker = None, y = lambda x: x, x = lambda x: x):
    train_dataloader = data_loader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = data_loader(dataset=test_data, batch_size=batch_size, shuffle=shuffle)
    if output is None:
        output = lambda x: model(x)
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n-----")
        train_loss = 0
        for batch, (x_train, y_train) in enumerate(train_dataloader): # this loops over data if dataloader is not DataLoader for bacthsizes which should input most data instantly. need tp fix this (just input different dataloader??)
            x_train, y_train = x(x_train).to(device), y(y_train).to(device)
            #print(x_train, y_train)
            #print(x_train, "x", y_train, "y")
            model.train()
            y_pred = output(x_train)
            #print(y_pred, y_train)
            loss = criterion(y_pred, y_train)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % (((10 if len(train_dataloader.dataset) < 300 else 100 if len(train_dataloader.dataset) < 2000 else 1000 if len(train_dataloader) < 20000 else 1000000)) if data_loader == DataLoader else 10) == 0:
                print(f"looked at {batch * len(x_train)}/{len(train_dataloader.dataset) if data_loader == DataLoader else len(train_data[0])} samples")
        train_loss /= len(train_dataloader)
        test_loss = 0
        if performance_tracker is not None:
            test_perf = 0
        model.eval()
        with torch.inference_mode():
            for x_test, y_test in test_dataloader:
                x_test, y_test = x(x_test).to(device), y(y_test).to(device)
                test_pred = output(x_test)
                test_loss += criterion(test_pred, y_test)
                if performance_tracker is not None:
                    perf = performance_tracker(y_test, input_to_performance_tracker(test_pred))
                    test_perf += perf
            test_loss /= len(test_dataloader)
            if performance_tracker is not None:
                test_perf /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.3f} | test_loss: {test_loss:.3f}", end = "")
        if performance_tracker is not None:
            print(f" | test_{tracked_performance}: {test_perf}")
            continue
        print()
    
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

