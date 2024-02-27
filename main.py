import streamlit as st
from utils import *
from models_ import *
import time
import json
from pathlib import Path
import shutil
import os
from os import listdir

st.title("Flashcard Generator")
st.header("upload pdf files for automated generation of an anki deck based on these")
DECK_DIR = Path("new_decks")
prompt = st.text_input("Here you can input the name of your anki deck")
col1, col2, col3 = st.columns(3) 
with col1:
    num_of_cards = st.toggle("let the program sort irrelevant info out")
with col2:
    continue_upon_exception = st.toggle("Continue with flashcards generated, if API request limit is reached")
with col3:
    save_cards = st.toggle("locally save created cards for editing")
    delete_cards = st.button("delete stored cards")
    cards_to_delete = st.selectbox("Select a specific deck to delete", options = ["All", *listdir(DECK_DIR)])

model = st.selectbox(label = "which model do you want to use", options = ["pretrained", "custom trained"])

file_large = st.file_uploader("Here you can upload one or multiple pdf files with large amount of text", type="pdf", accept_multiple_files=True)
submitter = st.empty()
submitter_small = False
submitter_large = False


submitter_large = submitter.button("submit pdf")
    
error_box = st.empty()
form_box = st.form("main_form", clear_on_submit=True)


if not DECK_DIR.exists():
    DECK_DIR.mkdir(parents=True)

def save_deck(cards, deck_name: str = "MyStreamlitDeck"):

    images = [card["image"] for card in cards]
    print(images[0])
    texts = [{"front": card["front"], "back": card["back"]} for card in cards]

    DECK_SAVE_PATH = DECK_DIR / f"{deck_name}"
    if not DECK_SAVE_PATH.exists():
        DECK_SAVE_PATH.mkdir(parents=True)
    
    IMAGE_SAVE_PATH = DECK_SAVE_PATH / "Images"
    if not IMAGE_SAVE_PATH.exists():
        IMAGE_SAVE_PATH.mkdir()
    for i, image in enumerate(images):
        IMAGE_SAVE_PATH_ = f"new_decks/{deck_name}/Images/image_{i}.png"
        image = image.save(fp = IMAGE_SAVE_PATH_, format = "PNG")

    TEXT_SAVE_PATH = DECK_SAVE_PATH / "flashcards.json"
    with open(TEXT_SAVE_PATH, "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent = 4)

def delete_deck(Path):
    for path_ in listdir(Path):
        path = Path / path_
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))

if delete_cards:
    if cards_to_delete == "All":
        delete_deck(DECK_DIR)
    else:
        delete_deck(DECK_DIR / cards_to_delete)
        shutil.rmtree(DECK_DIR / cards_to_delete)


    
def flashcards_from_large_file(file_, deck_name, mode, model):
    flashcards = []
    image_bytes_ = []
    for file in file_:
        progress_text = f"Generating summaries for {file.name}. Please wait..."
        progress_bar = st.progress(0, text = progress_text) 
        error_box_2 = st.empty()
        
        try:
            text, decision, _image_bytes, images = process_large_pdf(file, model)
        except:
            error_box_2.error("Error processing PDF, please make sure it is a valid PDF file.")
            text = []
            return
        
        
        answers = []
        questions = []
        image_bytes = []
        images_ = []
        
        continue_button_box = st.empty()
        retry = True

        for i, text_ in enumerate(text[:]):
            _answers = ""
            _questions = ""
            c = 0
            _retry = True
            if retry is False:
                break
            if decision[i] is False and num_of_cards is True:
                continue
            while _retry is True:
                try:
                    for _, _text_ in enumerate(text_):
                        try:
                            _answers += hub_chain.invoke({"input": _text_})["text"]  
                        except:
                            raise ValueError
                        try:
                            _questions += hub_chain_3.invoke({"input": _text_})["text"]
                        except:
                            _answers = ""
                            raise ValueError
                    try:
                        image_bytes.append(_image_bytes[i])
                        images_.append(images[i])
                    except:
                        _answers = ""
                        _questions = ""
                        raise ValueError
                    answers.append(_answers)
                    questions.append(_questions)
                    _retry = False
                    progress_bar.progress((i / len(text)), text = progress_text)
                except:
                    error_box_2.error(f"summary could not be generated. This is most likely due to too many calls to the API. The system will continue in 1/2 hour. Flashcards generated so far: {len(questions)}")
                    if mode:
                        retry = False
                        _retry = False
                        break
                    time.sleep(1800)
                    c += 1
                    continue_button_box.empty()

        flashcards_ = [{"summary": answers[i], "question": questions[i]} for i in range(len(questions))]
        flashcards += flashcards_
        image_bytes_ += image_bytes
        progress_bar.empty()

    return flashcards, image_bytes_, images_
    

if submitter_large:
    if prompt:
        deck_name = prompt
    else:
        deck_name = "MyStreamlitDeck"
    with form_box:
        flashcards, image_bytes, images = flashcards_from_large_file(file_large, deck_name, continue_upon_exception, model)
        st.write("flashcards")
        st.write(flashcards)
    
        cards = [{"front": f"{i['question']}", "back": f"{i['summary']}", "image_bytes": image_bytes[num], "image_type": "png"} for num, i in enumerate(flashcards)]
        st.write(f"Anki deck has been created. To download it open up anki and make sure you are conneced to webbindport:8765 in the anki-connect api")
        if save_cards:
            save_deck(cards=[{"front": card["front"], "back": card["back"], "image": images[i]} for i, card in enumerate(cards)], deck_name=prompt)
        
        def download(cards = cards, deck_name = deck_name):
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
        
        form_box.form_submit_button("download cards",  on_click=download)


