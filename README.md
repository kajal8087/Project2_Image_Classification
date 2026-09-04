# Image Classification using MobileNetV2

This is a small Python program which tells what object is there in a photo. I did not
train any model — I used MobileNetV2 which is already trained and knows 1000 types of
objects. My work was to prepare the photo in the correct size and format before giving
it to the model. Then the model gives the answer with a confidence percentage, like
"moped, 87%".

---

## What this project demonstrates

| Point | Where in the code |
| --- | --- |
| Using a pre-trained deep learning model | `MobileNetV2(weights="imagenet")` |
| Preparing an image the way a model expects | resize, array, batch, `preprocess_input` |
| Understanding model output | 1000 scores decoded into a readable label |
| Command line argument handling | `sys.argv` with a usage message |

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## How to Run

```powershell
python image_classification.py images\TwoWhiller.jpg
python image_classification.py "C:\Photos\my dog.jpg"
```

Put the path in quotes if it has spaces in it. JPG and PNG both work.

---

## Output

```
Image Classification Result
----------------------------
Image: images/TwoWhiller.jpg
Prediction: moped
Confidence: 87.45%
```

The first run downloads the model file (about 14 MB), so internet is needed once.
After that it works offline.
