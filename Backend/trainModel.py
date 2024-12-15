import time  # Import the time module
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from preprocessdata import preprocess_data
import joblib

# Start timing the entire script
start_time = time.time()

# Load and preprocess data
print("preprocessing data")
data_start = time.time()
# Change this to your directory
df = preprocess_data(r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\Backend\sentiment140.csv')

# Limit the dataset to 500,000 samples
df = df.sample(500000, random_state=42)
print(f"Data preprocessing and sampling completed in {time.time() - data_start:.2f} seconds.\n")

# Convert text into TF-IDF features
print("Converting text into TF-IDF")
tfidf_start = time.time()
vectorizer = TfidfVectorizer(max_features=1000)  # Optionally limit the features for faster training
# ^^ Andrew this might be the reason why it took so long actually wait you are limiting it idk
X = vectorizer.fit_transform(df['text'].apply(lambda x: ' '.join(x)))  # Convert list of tokens to a string
# ^^ Splits the words up into the array format for the TF- IDF
y = df['sentiment']
print(f"TF-IDF vectorization completed in {time.time() - tfidf_start:.2f} seconds.\n")

# Split data into training and testing sets
print("Splitting data for training and testing")
split_start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data splitting completed in {time.time() - split_start:.2f} seconds.\n")

# Train the model
print("Training")
train_start = time.time()
svm = SVC(kernel='linear', C=1, verbose = True, shrinking = False)  # Use a linear kernel for faster training // Verbose for debugging, shrinking for less computational error
svm.fit(X_train, y_train) # Important line
print(f"Model training completed in {time.time() - train_start:.2f} seconds.\n")

# Evaluate the model
print("Evaluating")
eval_start = time.time()
y_pred = svm.predict(X_test) # Important Line
print(classification_report(y_test, y_pred))
print(f"Model evaluation completed in {time.time() - eval_start:.2f} seconds.\n")

# Save the model for later use
print("Saving")
save_start = time.time()
joblib.dump(svm, 'RealSentiment_model.pkl')
joblib.dump(vectorizer, 'TFIDF_vecotrizer.pkl')
print(f"Model saved in {time.time() - save_start:.2f} seconds.\n")

# Total script runtime
print(f"Total script runtime: {time.time() - start_time:.2f} seconds.")
