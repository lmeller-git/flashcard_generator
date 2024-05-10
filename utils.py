import fitz
from io import BytesIO
import torch
from models_ import PageDecider
from utils_for_models import load_state_dict_2
from torchvision.transforms import v2
from PIL import Image
import requests
from pathlib import Path
import base64
from os import listdir
import os
import shutil


def process_large_pdf(file, model):

    deciding_model = PageDecider(3, 120, 2, 3, 3)
    if model == "original":
        deciding_model = load_state_dict_2(deciding_model, "models/page_decider_2.pth")
    elif model == "custom":
        deciding_model = load_state_dict_2(deciding_model, "models/new_page_decider.pth")
    deciding_model.eval()

    with fitz.open(stream = file.read(), filetype = "pdf") as doc:
        text = []
        images = []
        image_bytes = []
        decisions = []
        for i, page in enumerate(doc):
            try:
                pixmap = page.get_pixmap()
                text.append(split_text(page.get_text()))
                images.append(pix_to_image(page.get_pixmap()))
                image_bytes.append(pixmap_to_bytes(pixmap))
            except:
                continue
            decisions.append(page_decision(px_to_image(pixmap), deciding_model))
    return text, decisions, image_bytes, images

def split_text(text):
    num_container = int(len(text) / 300) if int(len(text) / 300) != 0 else 1
    split = int(len(text) / num_container)
    text_splits = [[] for i in range(num_container)]
    c = 1
    try:
        for i, letter in enumerate(text):
            if (i >= (split * c) and (letter == "." or letter == "!" or letter == "?" or letter == ":")) or i == (len(text) - 1):
                text_splits[c - 1].append(text[:(i + 1)])
                c += 1
                if c == num_container + 1:
                    break
    except:
        return [["."]]
    return text_splits

def page_decision(page, model):
    transform = v2.Compose([
    v2.Resize(size=(64,64), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
    ])
    _page = transform(page)
    with torch.inference_mode():
        logits = model(_page.unsqueeze(dim=0))
        probs = torch.softmax(logits.squeeze(), dim=0)
        pred = torch.argmax(probs, dim=0)
    return bool(pred)

def px_to_image(pixmap):
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    image_tensor = transform(image)
    return image_tensor

def pix_to_image(pixmap):
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return image

def pixmap_to_bytes(pixmap):
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG") 
    base64_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return base64_string

def SVD(image_tensor, CUTOFF):
    U, sigma, Vt = torch.linalg.svd(image_tensor)
    U_new = U[:,:CUTOFF]
    VT_new = Vt[:CUTOFF,:]
    sigma_new = torch.tensor([sigma[i] for i in range(CUTOFF)])
    return U_new, sigma_new, VT_new

def SVD_wrapper(images: Image, CUTOFF: int, colormode: str = "grayscale", SAVE_PATH: Path = Path("compressed_images"), PATH_TO_IMAGE : list = None):
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
    else:
        delete_stuff(SAVE_PATH)
                
    if colormode == "grayscale":
        transform = v2.Compose([v2.Resize((64,64)),
                                v2.Grayscale(),
                                v2.PILToTensor()])
    else:
        transform = v2.Compose([v2.Resize((64,64)),
                                v2.PILToTensor()])
    
    SVD_data = []
    for i, image in enumerate(images):
        image_tensor = transform(image)
        image_tensor = image_tensor.squeeze() / 255
        if colormode == "grayscale":
            SVD_data.append([(SVD(image_tensor, CUTOFF))])
        else:
            color_data = []
            for i in range(3):
                color_data.append((SVD(image_tensor[i], CUTOFF)))
            SVD_data.append(color_data)
    
    for i, image in enumerate(SVD_data):
        if PATH_TO_IMAGE:
            IMAGE_SAVE_PATH = SAVE_PATH / PATH_TO_IMAGE[i][:-4]
        else:
            IMAGE_SAVE_PATH = SAVE_PATH / f"image_{i}"
        IMAGE_SAVE_PATH.mkdir(exist_ok=True, parents=True)
        
        save_SVD(image, IMAGE_SAVE_PATH)

def save_SVD(image, IMAGE_SAVE_PATH):
    for i, color in enumerate(image):
        COLOR_IMAGE_SAVE_PATH = IMAGE_SAVE_PATH / f"color_{i}"
        COLOR_IMAGE_SAVE_PATH.mkdir(exist_ok=True, parents=True)
        for i, tensor in enumerate(["U", "SIGMA", "VT"]):
            torch.save(color[i], f"{COLOR_IMAGE_SAVE_PATH.absolute()}/{tensor}.pth")

def load_SVD(PATH, images: dict):
    transformer = v2.ToPILImage()
    for i, path in enumerate(listdir(PATH)):
        image_tensor = []
        for i, path_ in enumerate(listdir(PATH / path)):
            U = torch.load(f"{(PATH / f"{path}/{path_}/U.pth").absolute()}")
            SIGMA = torch.diag(torch.load(f"{(PATH / f"{path}/{path_}/SIGMA.pth").absolute()}"))
            VT = torch.load(f"{(PATH / f"{path}/{path_}/VT.pth").absolute()}")
            image_tensor.append(torch.linalg.multi_dot([U, SIGMA, VT]))
        images.update({f"{path}":transformer(torch.stack([image_tensor[0], image_tensor[1], image_tensor[2]], dim = 0))})
    return images

def delete_stuff(PATH):
    if not PATH.exists():
        return None
    elif PATH.exists():
        for path_ in listdir(PATH):
            path = PATH / path_
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)  
            elif os.path.isdir(path):
                shutil.rmtree(path)  
            else:
                raise ValueError("{} is not a file or dir.".format(path))

def create_anki_deck(cards, deck_name):
    # Create deck
    requests.post("http://localhost:8765", json={
        "action": "createDeck",
        "version": 6,
        "params": {
            "deck": deck_name
        }
    })
    
    # Add cards to the deck
    # add images to media
    for i, card in enumerate(cards):
        image_bytes = card["image_bytes"]
        if image_bytes:
            media_response = requests.post("http://localhost:8765", json={
                "action": "storeMediaFile",
                "version": 6,
                "params": {
                    "data": image_bytes,
                    "filename": f"{deck_name}_image_{i + 1}.png"
                }
            })
            media_reference = media_response.json()["result"]
        # create cards    
        requests.post("http://localhost:8765", json={
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Front": card["front"],
                        "Back": f"""{card["back"]}, {f'''{f'<div> <img src ={media_reference}/> <div>' if media_reference else''}'''}""",         
                    },
                    "options": {
                        "allowDuplicate": False
                    },
                    "tags": ["streamlit"]
                }
            }
        })

def download_anki_deck(cards, deck_name = "MyStreamlitDeck", deck_filename = "MyStreamlitDeck"):
    deck_name = f"{deck_name}"
    deck_filename = f"{deck_filename}.apkg"
    create_anki_deck(cards, deck_name)




