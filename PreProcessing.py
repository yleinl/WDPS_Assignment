import re

import requests
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")


def token_processing(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmatized = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    filtered_words = [word.lemma_ for word in doc if not word.is_stop]

    return {
        'original_text': text,
        'tokens': tokens,
        'lemmatized': lemmatized,
        'pos_tags': pos_tags,
        'filtered_words': filtered_words
    }


def text_preprocessing(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def preprocess_text(text):
    function_words = ["a", "an", "the", "is", "and", "of", "in", "to", "for"]

    doc = nlp(text)
    tokens = [token.text for token in doc if token.text.lower() not in function_words]
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def remove_punctuation(text):
    doc = nlp(text)
    processed_text = []
    for token in doc:
        if not token.is_punct:
            processed_text.append(token.text)

    return " ".join(processed_text)


def replace_wh_word(question, answer):
    doc = nlp(question)
    replaced_text = []

    for token in doc:
        if token.text.lower() in ["who", "whom", "whose", "what", "which", "where", "when", "why", "how"]:
            if answer is not None:
                replaced_text.append(answer)
            else:
                replaced_text.append(token.text)
        else:
            replaced_text.append(token.text)

    return " ".join(replaced_text)


def contains_wh_word(sentence):
    doc = nlp(sentence)
    wh_words = ["who", "whom", "whose", "what", "which", "where", "when", "why", "how"]
    for token in doc:
        if token.text.lower() in wh_words:
            return True
    return False


def clean_question(text):
    text = text.lower()
    function_words = ["a", "an", "the", "is", "and", "of", "in", "to", "for"]
    wh_words = ["who", "whom", "whose", "what", "which", "where", "when", "why", "how"]
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in function_words and word not in wh_words])
    return text


def extract_text_and_tokenize(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        tokens = re.findall(r'\b\w+\b', text)
        return set(tokens)
    except Exception as e:
        print(f"cannot access the page：{e}")
        return set()