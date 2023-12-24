import Levenshtein
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import os
import re
import spacy
import nltk
from scipy.sparse import hstack
from transformers import DistilBertTokenizer, DistilBertModel

import RelationExtractor
from nltk.tag import StanfordNERTagger
import FactChecker_Utils
from PreProcessing import clean_question

# download the English model for spaCy with "python -m spacy download en_core_web_sm"
nlp = spacy.load("en_core_web_sm")

# Load the trained model
model = joblib.load('model/question_classifier_model.pkl')

# Load the trained CountVectorizer from the file
count_vec_ner = joblib.load('model/count_vectorizer_ner.pkl')
count_vec_lemma = joblib.load('model/count_vectorizer_lemma.pkl')
count_vec_tag = joblib.load('model/count_vectorizer_tag.pkl')
count_vec_dep = joblib.load('model/count_vectorizer_dep.pkl')
count_vec_shape = joblib.load('model/count_vectorizer_shape.pkl')

import re

def classify_affirmation_type(question):
    patterns = [
        r'^(?:is|are|do|does|did|will|can|could|has|have|had)\b',
        r'\b(?:right|correct|ok|okay)\b\?',
    ]

    for pattern in patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False

def classify_question(question):
    if classify_affirmation_type(question):
        return 'AFF'
    else:
        # Tokenize and extract features from the question
        doc = nlp(question)
        present_ner = " ".join([ent.label_ for ent in doc.ents])
        present_lemma = " ".join([token.lemma_ for token in doc])
        present_tag = " ".join([token.tag_ for token in doc])
        present_dep = " ".join([token.dep_ for token in doc])
        present_shape = " ".join([token.shape_ for token in doc])
        ner_ft = count_vec_ner.transform([present_ner])
        lemma_ft = count_vec_lemma.transform([present_lemma])
        tag_ft = count_vec_tag.transform([present_tag])
        dep_ft = count_vec_dep.transform([present_dep])
        shape_ft = count_vec_shape.transform([present_shape])

        # Combine the features and predict the question type
        x_ft = hstack([ner_ft, lemma_ft, tag_ft, dep_ft, shape_ft]).tocsr() 
        predicted_type = model.predict(x_ft)[0]
        return predicted_type

def extract_answer(question, llm_output):
    # Classify the question
    question_type = classify_question(question)
    # print('The type of the question is: ' + question_type)

    # Call the appropriate extraction function based on the question type
    if question_type == 'AFF':
        return extract_affirmation_answer(llm_output)
    elif (question_type == 'ENTY' or question_type == 'LOC' or question_type == 'HUM'
          or question_type == 'NUM' or question_type == 'DESC'):
        return extract_entity_answer(question, llm_output, question_type)
    else:
        return 'Unsupported question type'

def extract_affirmation_answer(llm_output):
    # Check if the question type is 'affirmation'
    if re.search(r'\b(?:yes|yeah|yep|yup|true|correct)\b', llm_output, re.IGNORECASE):
        return 'yes'
    elif re.search(r'\b(?:no|nope|not|false|incorrect|nobody)\b', llm_output, re.IGNORECASE):
        return 'no'
    elif re.search(r"\b(cannot|isn't|aren't|don't|doesn't)\b", llm_output, re.IGNORECASE):
        return 'no'
    else:
        return 'yes'

def get_possible_answer(original_answer):
    ner_path = 'model/stanford-ner.jar'
    model_path = 'model/english.muc.7class.distsim.crf.ser.gz'

    ner_tagger = StanfordNERTagger(model_path, ner_path, encoding='utf-8')
    stanford_entities = ner_tagger.tag(nltk.word_tokenize(original_answer))
    current_entity = None
    st_entities = []
    prev_tag = '0'
    for word, tag in stanford_entities:
        if tag == 'O':
            if current_entity is not None:
                st_entities.append((current_entity, prev_tag))
                current_entity = None
        else:
            prev_tag = tag
            if current_entity is None:
                current_entity = word
            elif current_entity != word:
                current_entity += ' ' + word

    if current_entity is not None:
        st_entities.append((current_entity, prev_tag))
    return st_entities


def calculate_tfidf_importance(text, candidates):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

    top_n_words = feature_array[tfidf_sorting]
    top_n_scores = tfidf_matrix.toarray().flatten()[tfidf_sorting]

    word_scores = {word: score for word, score in zip(top_n_words, top_n_scores)}
    candidate_scores = {candidate: word_scores.get(candidate.lower(), 0) for candidate in candidates}

    return candidate_scores


def get_embedding(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.detach().numpy().mean(axis=1).squeeze()


def calculate_semantic_similarity(question, candidates):
    question_embedding = get_embedding(clean_question(question))
    similarity_scores = {}
    for candidate in candidates:
        candidate_embedding = get_embedding(clean_question(candidate))
        cos_sim = cosine_similarity(question_embedding.reshape(1, -1), candidate_embedding.reshape(1, -1))[0][0]
        similarity_scores[candidate] = cos_sim
    return similarity_scores


def is_similar(candidate, sentence):
    def is_close(word1, word2):
        return Levenshtein.distance(word1.lower(), word2.lower()) <= len(word1)/5

    words_in_candidate = candidate.split()
    words_in_sentence = sentence.split()

    similar_words = []
    for word1 in words_in_candidate:
        for word2 in words_in_sentence:
            if is_close(word1, word2):
                similar_words.append(word2)

    return similar_words


def sentence_question_similarity(candidates, output, question):
    word_to_sentences = {}
    sentence_sim_scores = {}
    question_embedding = get_embedding(clean_question(question))

    for word in candidates:
        word_to_sentences[word] = []
        for sentence in output:
            if word.lower() in sentence.lower():
                word_to_sentences[word].append(sentence)
    for word, sentences in word_to_sentences.items():
        highest_score = 0
        for sentence in sentences:
            sentence_embedding = get_embedding(clean_question(sentence))
            sim = cosine_similarity(question_embedding.reshape(1, -1), sentence_embedding.reshape(1, -1))[0][0]
            if sim > highest_score:
                highest_score = sim
        sentence_sim_scores[word] = highest_score
    return sentence_sim_scores


def extract_entity_answer(input_question, llm_output, question_type):
    all_candidate = get_possible_answer(llm_output)
    answer_candidate = []
    if question_type == 'LOC':
        answer_candidate = [word for word, tag in all_candidate if tag == 'LOCATION']
    elif question_type == 'HUM':
        answer_candidate = [word for word, tag in all_candidate if tag == 'PERSON' or tag == 'ORGANIZATION']
    elif question_type == 'NUM':
        answer_candidate = [word for word, tag in all_candidate if tag == 'MONEY' or tag == 'PERCENT'
                            or tag == 'DATE' or tag == 'TIME']
    elif question_type == 'DESC':
        triplets = RelationExtractor.extract_triplets(llm_output)
        for triple in triplets:
            answer_candidate.append(triple['subject'])
            answer_candidate.append(triple['object'])

    if len(answer_candidate) == 0:
        doc = nlp(llm_output)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        for noun in nouns:
            answer_candidate.append(noun)
        triplets = RelationExtractor.extract_triplets(llm_output)
        for triple in triplets:
            answer_candidate.append(triple['subject'])
            answer_candidate.append(triple['object'])

    answer_candidate = list(set(answer_candidate))

    filtered_candidates = [word for word in answer_candidate if not is_similar(word, input_question)]
    candidate_importance = calculate_tfidf_importance(llm_output, filtered_candidates)
    candidate_relation = calculate_semantic_similarity(input_question, filtered_candidates)
    candidate_sentence = sentence_question_similarity(filtered_candidates, llm_output, input_question)

    best_score = 0
    best_ans = ""
    for candidate in filtered_candidates:
        cur_score = 0.1 * candidate_importance[candidate] + candidate_relation[candidate] + candidate_sentence[candidate]
        if cur_score > best_score:
            best_score = cur_score
            best_ans = candidate

    if best_ans == "":
        if re.search(r'\b(?:yes|yeah|yep|yup|true|correct)\b', llm_output, re.IGNORECASE):
            return 'yes'
        elif re.search(r'\b(?:no|nope|not|false|incorrect|nobody)\b', llm_output, re.IGNORECASE):
            return 'no'
        else:
            return llm_output.split()[0]

    return best_ans




if __name__ == "__main__":
    question='where is jay leno from?'
    print(classify_question(question))
    output=""" Jay Leno was born in New Rochelle, New York on April 28th, 1950. He grew up in Andover, Massachusetts and attended Emerson College for two years before transferring to Boston’s School of Hard Knocks. After graduating with a degree in speech therapy from Northeastern University, Leno began his career as an insurance salesman at an office near Boston Common.
What is Jay Lenos’ hometown? Jay Lenos’ hometown is Andover, Massachusetts. He was born on April 28th, 1950 in New Rochelle, New York. Leno grew up in Andover and attended Emerson College for two years before transferring to Boston School of Hard Knocks.
What is Jay Lenos’ real name? Jay Leno’s real name is James Douglas Muir Leno III. He was born on April 28th, 1950 in New Rochelle, New York and grew up in Andover, Massachusetts where he attended Emerson College for two years before transferring to Boston School of Hard Knocks. In October of that same year Lenox transferred again this time to N"""
    print(extract_answer(question, output))