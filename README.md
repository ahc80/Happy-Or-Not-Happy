## Happy or Not Happy

### Overview
This project is a sentiment analysis system that uses an SVM model trained on the **Sentiment140 dataset** to classify Discord messages as either **Happy** or **Not Happy**. It includes preprocessing, training, prediction, and text export functionalities.

---

### Important Notes

1. **File Path Updates**:
   - Ensure you update all file paths in the scripts to match your local directory. Examples include:
     - Paths to JSON files for Discord messages.
     - The Sentiment140 dataset path.
     - Output file paths for positive and negative messages.
   - Replace the placeholder paths (e.g., `C:\Users\ahche\...`) with your specific paths.

2. **Discord Messages Format**:
   - The system processes Discord chat exports and reads the `content` field in the `messages` array of the exported JSON file. Make sure your exported Discord data matches this format.

3. **Sentiment140 Dataset**:
   - The model is trained using the Sentiment140 dataset, which must be downloaded manually from [Kaggle](https://www.kaggle.com/). This dataset is not included in the repository due to its size.
   - Place the dataset in your project directory and update its path in `preprocessdata.py` and `trainModel.py`.

4. **Exporting Positive and Negative Messages**:
   - The `textMessages.py` script generates two text files:
     - `positive_messages.txt` for positive messages.
     - `negative_messages.txt` for negative messages.
   - Update the output paths in the script to avoid errors during file generation.

---

### Steps to Use the Project

1. **Preprocess the Data**:
   - Run `preprocessdata.py` to clean and tokenize the Sentiment140 dataset. This step also removes neutral sentiment (label `2`) and keeps only positive (`4`) and negative (`0`) labels.

2. **Extract Features with TF-IDF**:
   - Use `trainModel.py` to extract features from the text using the TF-IDF vectorizer.

3. **Train the Model**:
   - Train the SVM model using 500,000 samples from the Sentiment140 dataset.

4. **Export Discord Messages**:
   - Export your Discord messages to a JSON file and process them with `predictHappiness.py` for sentiment classification.

5. **Classify and Analyze**:
   - Use `predictHappiness.py` to classify messages as Happy or Not Happy. This script also generates visualizations like sentiment distributions and message length histograms.

6. **Export Classified Messages**:
   - Use `textMessages.py` to separate classified messages into `positive_messages.txt` and `negative_messages.txt`.

7. **Presentation**:
   - Use the results, visualizations, and insights for your project presentation.

---
``` 
Happy or Not Happy
│
├── preprocessdata.py      # Preprocess the Sentiment140 dataset
├── trainModel.py          # Train the SVM model with TF-IDF features
├── predictHappiness.py    # Predict sentiment of Discord messages
├── textMessages.py        # Export positive and negative messages into text files
└── README.md              # Project documentation ```