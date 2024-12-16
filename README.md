## Notes

### Steps
1. Preprocess to tokenize the words. Use an NLP like NTKL to preprocess it
2. Feature extractor with tfidf
3. Commence the Training
    4. 500,000 Samples used of the 1.6Million Tweet Dataset (Sentiment140)
5. Training Complete: Use DiscordExporterChat to Export the Materials
6. Run the Model to test out the Discord Chats
7. Make Presentation

The Structure:

Happy or Not Happy
│
├── preprocessdata.py  # For text preprocessing
├── trainModel.py         # For training the model
├── predict.py             # For making predictions
└── README.md              # Project description