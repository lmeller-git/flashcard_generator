import streamlit as st
from os import listdir
from pathlib import Path
from utils_for_models import *
from PIL import Image
from torchvision.transforms import v2
import random
import json
from torch import nn
import csv
from models_ import PageDecider

st.title("Train the model on your data")
st.error("currently under construction")

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
EPOCHS = 10
train_split = 0.8
model = PageDecider(3, 120, 2, 3, 3).to(device)

TRAINING_DATA_PATH = Path("training_data")
TRAINING_IMAGES_PATH = TRAINING_DATA_PATH / "Images"
TRAINING_LABELS_PATH = TRAINING_DATA_PATH / "labels.json"

PROCESSED_DATA = Path("processed_data")
PROCESSED_DATA_TRAIN_PATH = PROCESSED_DATA / "train_data"
PROCESSED_DATA_TEST_PATH = PROCESSED_DATA / "test_data"
PROCESSED_IMAGES_TRAIN_PATH =  PROCESSED_DATA_TRAIN_PATH / "images"
PROCESSED_IMAGES_TEST_PATH =  PROCESSED_DATA_TEST_PATH / "images"
PROCESSED_LABELS_TRAIN_PATH = PROCESSED_DATA_TRAIN_PATH / "labels.csv"
PROCESSED_LABELS_TEST_PATH = PROCESSED_DATA_TEST_PATH / "labels.csv"

if not PROCESSED_DATA.exists():
    PROCESSED_DATA.mkdir(parents=True)

if not PROCESSED_DATA_TRAIN_PATH.exists():
    PROCESSED_DATA_TRAIN_PATH.mkdir(parents=True)

if not PROCESSED_DATA_TEST_PATH.exists():
    PROCESSED_DATA_TEST_PATH.mkdir(parents=True)

if not PROCESSED_IMAGES_TRAIN_PATH.exists():
    PROCESSED_IMAGES_TRAIN_PATH.mkdir(parents=True)

if not PROCESSED_IMAGES_TEST_PATH.exists():
    PROCESSED_IMAGES_TEST_PATH.mkdir(parents=True)


transform = v2.Compose([
    v2.Resize(size=(64,64), antialias=True)
])


images = [Image.open(f"training_data/Images/{i}") for i in listdir(TRAINING_IMAGES_PATH)]

with open(TRAINING_LABELS_PATH, "r") as f:
    labels = json.load(f)


def mingle_data(images, labels):
    delta_yes_no = sum(labels.values()) - len(labels) / 2
    new_data = []
    new_labels = {}
    transform = v2.RandomApply([
        v2.ColorJitter(0.5, 0.2, 0.5),
        v2.RandomGrayscale(0.2)
    ])
    if delta_yes_no < 0: # more 'No' labels than 'Yes'
        while len(new_data) < abs(delta_yes_no):
            for i, decision in enumerate(labels.values()):
                if decision is False:
                    new_data.append(transform(images[i]))
                    new_labels.update({f"Image_{i + len(images) - 1 + len(new_data)}.png": True})
                elif decision is True:
                    if random.random() < abs(delta_yes_no) / len(labels) * 2:
                        images[i] = transform(images[i])
                if len(new_data) == abs(delta_yes_no) + 1:
                    break
        images.append(new_data)
        labels.update(new_labels)

    elif delta_yes_no > 0:  # more 'Yes' labels than 'No'
        while len(new_data) < abs(delta_yes_no):
            for i, decision in enumerate(labels.values()):
                if decision is True:
                    new_data.append(transform(images[i]))
                    new_labels.update({f"Image_{i + len(images) - 1 + len(new_data)}.png": False})
                elif decision is False:
                    if random.random() < abs(delta_yes_no) / len(labels) * 2:
                        images[i] = transform(images[i])
                if len(new_data) == delta_yes_no + 1:
                    break
        images += new_data
        labels.update(new_labels)

    random_indices = torch.randperm(len(images))
    images = [images[i] for i in random_indices]
    fleeting_list = list(map(lambda x: [x[0], int(x[1])], list(labels.items())))
    fleeting_list = [fleeting_list[i] for i in random_indices]
    labels = dict(fleeting_list)

    return  images, labels

if st.button("save data"):
    images, labels = mingle_data(images, labels)

    train_cutoff = int(train_split * len(images))
    train_images = images[:train_cutoff]
    train_labels = dict(list(labels.items())[:train_cutoff])
    test_images = images[train_cutoff:]
    test_labels = dict(list(labels.items())[:train_cutoff])
    

    with open(PROCESSED_LABELS_TRAIN_PATH, "w") as f:
        write = csv.writer(f)
        write.writerows(train_labels.items())
    
    with open(PROCESSED_LABELS_TEST_PATH, "w") as f:
        write = csv.writer(f)
        write.writerows(test_labels.items())


    for i, image in enumerate(train_images):
        PROCESSED_IMAGE_SAVE_PATH_FOR_TRAINING = f"processed_data/train_data/images/{list(train_labels.keys())[i]}"
        image = image.save(fp = PROCESSED_IMAGE_SAVE_PATH_FOR_TRAINING, format = "PNG")
    
    for i, image in enumerate(train_images):
        PROCESSED_IMAGE_SAVE_PATH_FOR_TESTING = f"processed_data/test_data/images/{list(test_labels.keys())[i]}"
        image = image.save(fp = PROCESSED_IMAGE_SAVE_PATH_FOR_TESTING, format = "PNG")


if st.button("train ai"):

    train_data = ppp_files("processed_data/train_data/labels.csv",
                    "processed_data/train_data/images",
                    transform)

    test_data = ppp_files("processed_data/test_data/labels.csv",
                    "processed_data/test_data/images",
                    transform)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    acc_pre_training = accuracy(model, test_data, device=device)

    training_loop_2(EPOCHS, model, criterion, optimizer, train_data, test_data, device, batch_size=BATCH_SIZE)

    acc_post_training = accuracy(model, test_data, device=device)

    st.write("acc_pre_training: ", acc_pre_training," | ", "acc_post_training: ", acc_post_training)

    save_function(model, "new_page_decider.pth")
    st.write("new model saved succesfully as new_page_decider.pth in models")