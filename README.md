# FER2013
Facial expression recognition model using CNN and PyTorch

This project is a Facial Expression Recognition model that predicts a person's emotion from a face image.

The model uses a CNN trained for 7 emotion classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Project Files

- `training.ipynb`  
  The notebook used to train the CNN model.

- `inference.py`  
  The Python script used to load the trained model and predict emotions from new images.

- `best_model.pth`  
  The saved trained model weights.

## How to Run

Make sure `inference.py`, `best_model.pth`, and your test image are in the same folder.

Example:

```bash
python inference.py --checkpoint best_model.pth --input myface.jpg
## Demo

A demo is included at the end of `training.ipynb`, where a sample face image is displayed and the trained model predicts the facial expression.


