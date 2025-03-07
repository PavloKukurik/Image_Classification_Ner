# Task 2: Named Entity Recognition + Image Classification

## Project Overview

#### **`Dataset`**: This project uses the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data) from Kaggle, which contains images of 10 different animal classes (_dir names was translated into english_).

This task implements a **Machine Learning pipeline** that integrates:
- **Named Entity Recognition (NER)**: Extracts animal names from a text description.
- **Image Classification**: Identifies the animal in an image.
- **Final Validation**: Compares the extracted text entities with the classified image to determine if the description matches the image content.

---
## Project Structure
```
task2_ner_image_classification/
│── data/
│   ├── raw-img/                   # Image dataset with 10 animal classes
│── image_classification/
│   ├── train_image.py             # Train the image classification model
│   ├── inference_image.py         # Inference script for image classification
│── ner/
│   ├── train_ner.py               # Train the NER model
│   ├── inference_ner.py           # Inference script for extracting animal names from text
│   ├── ner_data.py                # Sample dataset for NER model
├── pipeline.py                # Final script integrating both models
│── requirements.txt                # List of dependencies
│── README.md                       # Documentation
│── task_2.ipynb                    # Jupyter Notebook for analysis & visualization
```

## Installation & Setup
### 1️⃣ Install Dependencies
Run the following command to install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Named Entity Recognition (NER) Model
Train the NER model to identify animal names in text:
```bash
python ner/train_ner.py
```

### 3️⃣ Train the Image Classification Model
Train the image classifier using ResNet18:
```bash
python image_classification/train_image.py
```

## Running Inference
### Extract Animals from Text (NER Model)
```bash
python ner/inference_ner.py "There is a cat and a dog in the picture."
```
Expected Output:
```
Extracted Animals: ['cat', 'dog']
```

### Classify an Animal in an Image
```bash
python image_classification/inference_image.py data/raw-img/cat/1.jpeg
```
Expected Output:
```
Predicted class: cat
```

### Running the Final Pipeline
The final pipeline integrates both models, comparing the detected animals in the text and the classified image.
```bash
python pipeline.py "There is a dog in the picture." "data/raw-img/dog/OIP-0-TGu-jQ5fRR5fPXr1ijvQHaHa.jpeg"
```
Expected Output:
```
Extracted Animals from Text: ['cat']
Predicted Animal in Image: cat
Does the text match the image? True
```

---

