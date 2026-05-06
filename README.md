# FER2013
# Facial Expression Recognition

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
  The notebook used to train, evaluate, and demo the CNN model.

- `inference.py`  
  The Python script used to load the trained model and predict emotions from new images.

- `best_model.pth`  
  The saved trained model weights.

## Results

- Test Accuracy: 65.76%
- Best Validation Accuracy: 66.86%
- Macro-average AUC: 0.9148

## How to Run

Make sure `inference.py`, `best_model.pth`, and your test image are in the same folder.

Run the model with:

```bash
python inference.py --checkpoint best_model.pth --input myface.jpg
```

Replace `myface.jpg` with the name of the image you want to test.

## Demo

A demo is included at the end of `training.ipynb`. The demo displays a sample face image and uses the trained model to predict the facial expression.

Example output:

```text
Prediction : HAPPY
Confidence : 97.76%
```
## Notes

The model expects face images and resizes them to 48x48 grayscale before prediction.
