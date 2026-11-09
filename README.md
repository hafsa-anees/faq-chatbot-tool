\# FAQ Chatbot



\*\*Made by:\*\* \[Hafsa]



\## Description

A chatbot that answers frequently asked questions about an online shop.



\## How it works

1\. Text preprocessing with NLTK (lowercase, tokenize, remove stopwords, lemmatize)

2\. TF-IDF converts questions into vectors

3\. Cosine similarity finds the most similar FAQ

4\. The best matching answer is shown in a chat interface

5\. If the match score is low, the bot says it did not understand



\## Technologies Used

\- Python

\- NLTK

\- scikit-learn

\- Streamlit (chat UI)



\## How to Run

1\. pip install -r requirements.txt

2\. python -m nltk.downloader punkt punkt\_tab stopwords wordnet

3\. streamlit run app.py

