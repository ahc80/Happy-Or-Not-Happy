## Notes

### Steps
1. Preprocess to tokenize the words. Use an NLP like NTKL to preprocess it
2. Feature extractor with tfidf
3. Commence the Training
    4. (is 1.6million tweets bad?)
5. Training Complete: Use DiscordExporterChat to Export the Materials
6. Run the Model to test out the Discord Chats
7. Make Presentation

The Ideal Structure:

Happy or Not Happy
│
├── preprocessdata.py  # For text preprocessing
├── trainModel.py         # For training the model
├── predict.py             # For making predictions
└── README.md              # Project description


TODO Discord stuff