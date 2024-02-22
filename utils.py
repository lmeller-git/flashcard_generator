import fitz
from io import BytesIO
import torch
from models_ import model_decide
from torchvision.transforms import v2
from PIL import Image
import requests
import base64



def process_large_pdf(file):
    with fitz.open(stream = file.read(), filetype = "pdf") as doc:
        text = []
        images = []
        image_bytes = []
        decisions = []
        for i, page in enumerate(doc):
            pixmap = page.get_pixmap()
            text.append(split_text(page.get_text()))
            images.append(pix_to_image(page.get_pixmap()))
            image_bytes.append(pixmap_to_bytes(pixmap))
            decisions.append(page_decision(px_to_image(pixmap)))
            if i > 10 :
                break
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


def page_decision(page):
    transform = v2.Compose([
    v2.Resize(size=(64,64), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
    ])
    _page = transform(page)
    with torch.inference_mode():
        logits = model_decide(_page.unsqueeze(dim=0))
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

def create_anki_deck(cards, deck_name):
    
    requests.post("http://localhost:8765", json={
        "action": "createDeck",
        "version": 6,
        "params": {
            "deck": deck_name
        }
    })
    
    
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
