import json
import joblib
import matplotlib.pyplot as plt

# Function to clean text (reuse the same logic as in training)
import re
from nltk.tokenize import word_tokenize

# Copied from the other python file
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)     # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text.lower()

# Load and preprocess JSON file
def load_json(file_path):
    with open(file_path, errors="ignore") as file:
        data = json.load(file)

    messages = [m["content"] for m in data["messages"]]
    return messages

# Predict sentiment for messages
def predict_sentiment(messages, vectorizer, model):
    # Clean and preprocess messages
    cleaned_messages = [clean_text(msg) for msg in messages]
    
    # Transform messages using the vectorizer
    transformed_messages = vectorizer.transform(cleaned_messages)
    
    # Predict sentiment
    predictions = model.predict(transformed_messages)
    return predictions, messages

# Save messages to files
def save_messages_to_files(messages, predictions, negative_file, positive_file):
    with open(negative_file, 'w', encoding='utf-8') as neg_file, \
         open(positive_file, 'w', encoding='utf-8') as pos_file:
        
        for message, prediction in zip(messages, predictions):
            if prediction == 0:  # Assuming 0 = Negative
                neg_file.write(message + '\n')
            elif prediction == 1:  # Assuming 1 = Positive
                pos_file.write(message + '\n')

# Main function
def main():
    # File paths
    json_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\cwru2025.json'  # Replace with your JSON file path
    model_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\RealSentiment_model.pkl'
    vectorizer_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\TFIDF_vecotrizer.pkl'
    negative_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\negative_messages2025.txt'
    positive_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\positive_messages2025.txt'
    
    # Load JSON data
    print("Loading JSON data")
    messages = load_json(json_file)
    print(f"Loaded {len(messages)} messages.")
    
    # Load SVM model and TF-IDF vectorizer
    print("Loading SVM model and TF-IDF vectorizer")
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    
    # Predict sentiment
    print("Predicting sentiment")
    predictions, messages = predict_sentiment(messages, vectorizer, model)
    
    # Save messages to separate files
    print("Saving messages to text files")
    save_messages_to_files(messages, predictions, negative_file, positive_file)
    
    print("Analysis complete. Messages saved to:")
    print(f"  - Negative messages: {negative_file}")
    print(f"  - Positive messages: {positive_file}")

if __name__ == '__main__':
    main()
