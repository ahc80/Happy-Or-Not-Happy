import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Uncomment these lines and run once if you havent downloaded the data yet:
nltk.download('stopwords')
nltk.download('punkt_tab')

# These Methods are used to process the code so that it easier to read for the machine as well as for us to read

def clean_text(text):
    """
    Clean the input text by removing URLs, mentions, and non-alphabetic characters.
    
    @param text: The raw text to be cleaned (string).
    @return: A cleaned string with no URLs, mentions, or non-alphabetic chars.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess_data(file_path):
    """
    Preprocess the Sentiment140 dataset by:
    - Loading the CSV file
    - Filtering out neutral sentiments (2)
    - Converting sentiment labels (4 to 1, and 0 to 0)
    - Cleaning and tokenizing text
    - Removing stopwords
    
    @param file_path: Path to the sentiment140 CSV file.
    @return: A preprocessed pandas DataFrame with 'sentiment' and 'text' columns.
    """
    # Load the dataset. No header row, so header=None
    # Columns: 0: sentiment, 1: id, 2: date, 3: query, 4: user, 5: text
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    
    # Keep only sentiment and text columns
    df = df[['sentiment', 'text']]
    
    # Filter out neutral sentiment (2) and keep only 0 (negative) and 4 (positive)
    df = df[df['sentiment'].isin([0, 4])]
    
    # Convert sentiment: 4 -> 1 (positive), 0 -> 0 (negative)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)
    
    # Get stopwords from NLTK
    stop_words = set(stopwords.words('english'))
    
    # Clean, tokenize, and remove stopwords
    df['text'] = df['text'].apply(lambda x: clean_text(x.lower()))  # lowercase & clean
    df['text'] = df['text'].apply(word_tokenize)                     # tokenize
    df['text'] = df['text'].apply(lambda words: [w for w in words if w not in stop_words])
    
    return df

if __name__ == '__main__':
    # Use a raw string for the file path
    df = preprocess_data(r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\Backend\sentiment140.csv')
    print(df.head())
