import streamlit as st
from utils import *
import json
from pathlib import Path
import shutil
import os
from os import listdir


st.title("Flashcard editor")
st.error("currently under construction")
saved_cards = st.selectbox(label = "which cards do yu want to look at", options=["new flashcards", "edited flashcards"])
save_data = st.toggle(label = "save data to train the model", key = "data_saver")

if saved_cards == "edited flashcards":
    DECK_SAVE_PATH = Path("edited_decks")
    DECKS = "edited_decks"
else:
    DECK_SAVE_PATH = Path("new_decks")
    DECKS = "new_decks"


if not DECK_SAVE_PATH.exists():
    DECK_SAVE_PATH.mkdir(parents=True)


decks = []
for deck in listdir(DECK_SAVE_PATH):
     decks.append(str(deck))

deck_name = st.selectbox("which flashcard stack would you like to access?", options=decks)

try:
    CARD_SAVE_PATH = DECK_SAVE_PATH / deck_name / "flashcards.json"
    IMAGE_SAVE_PATH = DECK_SAVE_PATH / deck_name / "Images"
except:
    st.error("choose another directory")
    st.stop()

with open(CARD_SAVE_PATH, 'r') as f:
    cards = json.load(f)


flashcards = [{"front": card["front"], "back": card["back"], "image": Image.open(f"{DECKS}/{deck_name}/Images/image_{i}.png")} for i, card in enumerate(cards)]

def delete_cards(Path: Path):
    
    for path_ in listdir(Path):
        path = Path / path_
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  
        elif os.path.isdir(path):
            shutil.rmtree(path)  
        else:
            raise ValueError("{} is not a file or dir.".format(path))

def downloader(deck_name = deck_name):
    with open(f"edited_decks/{deck_name}/flashcards.json", 'r') as f:
        flashcards = json.load(f)

    image_bytes = []
    for i in range(len(flashcards)):
        image_path = f"edited_decks/{deck_name}/Images/image_{i}.png"
        with open(image_path, "rb") as img_file:
                    image = Image.open(img_file)
                    image_byte_arr = BytesIO()
                    image.save(image_byte_arr, format = "PNG")
                    image_bytes.append(base64.b64encode(image_byte_arr.getvalue()).decode("utf-8"))
        
    cards = [{"front": f"{flashcards[i]["front"]}", "back": f"{flashcards[i]["back"]}", "image_bytes": image_bytes[i], "image_type": "png"} for i in range(len(flashcards))]
    try:
        download_anki_deck(cards, deck_name = deck_name, deck_filename = deck_name)
    except:
        st.error("No connection to the anki API was possible. Please activate your Anki API.\n\
                To do so:\n\
                - open Anki\n\
                - select add-ons in tools\n\
                - click 'get add-on' and paste 2055492159\n\
                - install add-on and restart anki\n\
                - make sure that webbindport is set to 8765 in the config of anki-connect\n\
                - make sure Enable is checked\n\
                - have anki open in the background and download the deck")

st.header("your flashcards")
form = st.form(key = "card_editor")

with form:
    fronts = []
    backs = []
    excluded = []
    for i, card in enumerate(flashcards):
        st.subheader(f"Card {i + 1}")
        fronts.append(st.text_input(label = "Front", value = card["front"], key = f"front{i}"))
        backs.append(st.text_input(label = "Back:", value = card["back"], key = f"back{i}"))
        st.image(card["image"], "Image")
        excluded.append(st.toggle("Exclude this card", key = f"toggle_card_{i}", value = False))
        st.write(f"currently {'' if excluded[i] else 'not '}excluded")
        st.divider()

    submitter = form.form_submit_button(label = "Finish editing")
    if submitter:
        pass

def save_deck(cards = flashcards, deck_name = deck_name, excluded = excluded, save_data = save_data):
    
    images = [card["image"] for i, card in enumerate(cards) if not excluded[i]]
    images_2 = [card["image"] for _, card in enumerate(cards)]

    texts = [{"front": card["front"], "back": card["back"]} for i, card in enumerate(cards) if not excluded[i]]
    
    def save_training_data(images, excluded):
        
        TRAINING_DATA_DIR = Path("training_data")
        
        if not TRAINING_DATA_DIR.exists():
            TRAINING_DATA_DIR.mkdir(parents=True)
        
        TRAINING_LABEL_SAVE_PATH = TRAINING_DATA_DIR / "labels.json"
        TRAINING_IMAGE_SAVE_DIR = TRAINING_DATA_DIR / "Images"
        
        if not TRAINING_IMAGE_SAVE_DIR.exists():
            TRAINING_IMAGE_SAVE_DIR.mkdir(parents=True)
        
        num_of_img = len(listdir(TRAINING_IMAGE_SAVE_DIR))
        
        # make data
        if not TRAINING_LABEL_SAVE_PATH.exists():
            labels = {}
            for i in range(len(images)):
                labels.update({f"Image_{i + num_of_img}.png": not excluded[i]}) 
        else:
            with open(TRAINING_LABEL_SAVE_PATH, "r+") as f:
                labels = json.load(f)
            for i in range(len(images)):
                labels.update({f"Image_{i + num_of_img}.png": not excluded[i]}) 

        
        with open(f"{TRAINING_LABEL_SAVE_PATH}", "w") as f:
            json.dump(labels, f, ensure_ascii=False)
            
        for i, image in enumerate(images):
            IMAGE_SAVE_PATH_FOR_TRAINING = f"training_data/Images/image_{i + num_of_img}.png"
            image = image.save(fp = IMAGE_SAVE_PATH_FOR_TRAINING, format = "PNG")
    
    if save_data:
        save_training_data(images_2, excluded)
    
    
    DECK_DIR = Path("edited_decks")
    if not DECK_DIR.exists():
        DECK_DIR.mkdir(parents=True)
    SAVE_PATH = DECK_DIR / deck_name
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)
    elif SAVE_PATH.exists():
        for path_ in listdir(SAVE_PATH):
            path = SAVE_PATH / path_
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)  
            elif os.path.isdir(path):
                shutil.rmtree(path)  
            else:
                raise ValueError("{} is not a file or dir.".format(path))
            

    DECK_SAVE_PATH = DECK_DIR / f"{deck_name}"
    if not DECK_SAVE_PATH.exists():
        DECK_SAVE_PATH.mkdir(parents=True)
    
    IMAGE_SAVE_PATH = DECK_SAVE_PATH / "Images"
    if not IMAGE_SAVE_PATH.exists():
        IMAGE_SAVE_PATH.mkdir()
    for i, image in enumerate(images):
        IMAGE_SAVE_PATH_ = f"edited_decks/{deck_name}/Images/image_{i}.png"
        image = image.save(fp = IMAGE_SAVE_PATH_, format = "PNG")

    TEXT_SAVE_PATH = DECK_SAVE_PATH / "flashcards.json"
    with open(TEXT_SAVE_PATH, "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent = 4)

def delete_training_data():
    DECK_DIR = Path("training_data")
    for path_ in listdir(DECK_DIR):
        path = DECK_DIR / path_
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  
        elif os.path.isdir(path):
            shutil.rmtree(path)  
        else:
            raise ValueError("{} is not a file or dir.".format(path))

col1, col2, col3, col4 = st.columns(4)
with col1:
    cards_to_delete = st.selectbox("Select a specific deck to delete", options = ["All", *listdir(Path("edited_decks"))])
    reset_button = st.button(label = "delete saved decks")
with col2:
    downloader_ = st.button(label = "download cards to anki")  
with col3:
    save = st.button(label = "save deck", on_click=save_deck)
with col4:
    delete_train_data = st.button(label = "delete training data", on_click=delete_training_data)

if  reset_button:
    if cards_to_delete == "All":
        delete_cards(Path("edited_decks"))
    else:
        delete_cards(Path("edited_decks") / cards_to_delete)
        shutil.rmtree(Path("edited_decks") / cards_to_delete)

if downloader_:
    downloader(deck_name=deck_name)