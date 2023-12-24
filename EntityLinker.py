import nltk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import FactChecker_Utils
import PreProcessing
import RelationExtractor
import wikipedia
import time
import spacy
from nltk.tag import StanfordNERTagger
from concurrent.futures import ThreadPoolExecutor, as_completed

from AnswerExtract import get_embedding

# Load the en_core_web_md model
nlp = spacy.load('en_core_web_md')


def cosine_sim(text1, text2):
    text1 = PreProcessing.preprocess_text(text1)
    text2 = PreProcessing.preprocess_text(text2)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_sim[0][0]

def remove_substrings(entities):
    entities.sort(key=len, reverse=True)
    final_entities = []
    for entity in entities:
        if not any(entity in final_entity for final_entity in final_entities):
            final_entities.append(entity)

    return final_entities


def get_named_entity(statement, original_answer):
    ner_path = 'model/stanford-ner.jar'
    model_path = 'model/english.muc.7class.distsim.crf.ser.gz'

    ner_tagger = StanfordNERTagger(model_path, ner_path, encoding='utf-8')
    entities = []
    triplets = RelationExtractor.extract_triplets(statement)

    for triple in triplets:
        entities.append(triple['subject'])
        entities.append(triple['object'])
    stanford_entities = ner_tagger.tag(nltk.word_tokenize(original_answer))
    current_entity = None
    st_entities = []
    for word, tag in stanford_entities:
        if tag == 'O':
            if current_entity is not None:
                st_entities.append(current_entity)
                current_entity = None
        else:
            if current_entity is None:
                current_entity = word
            elif current_entity != word:
                current_entity += ' ' + word

    if current_entity is not None:
        st_entities.append(current_entity)
    combined_entities = []

    for entity in st_entities:
        if not any(entity in e for e in entities):
            combined_entities.append(entity)

    combined_entities.extend(entities)
    final_entities = list(set(combined_entities))
    filtered = []
    for entity in final_entities:
        is_contained = False
        for other_entity in final_entities:
            if entity != other_entity and entity in other_entity:
                is_contained = True
                break
        if not is_contained:
            filtered.append(entity)
        if is_contained and entity in entities:
            filtered.append(entity)
    print(f'entities:{filtered}')
    return filtered


def get_topic(topic):
    topics = []
    res = wikipedia.search(topic)
    for r in res:
        try:
            r_search = wikipedia.page(r,auto_suggest=False)
            topics.append(r_search)
        except wikipedia.exceptions.DisambiguationError as error:
            continue

    return topics


def qna_entity_linking_wikipedia(q,a, original_answer):
    statement, _ = FactChecker_Utils.question_to_statement(q, a)
    entities_name = get_named_entity(statement, original_answer)
    print(f'statement:{statement}')
    print(f'entities_name:{entities_name}')
    final_results = []
    for ent_name in entities_name:
        print(f'######{ent_name}######')
        final_results.append(get_disambiguation(statement,ent_name,final_results))
    print(f'final_results:{final_results}')
    return final_results


def calculate_word_similarity(word1, word2):
    word1_embedding = get_embedding(word1)
    word2_embedding = get_embedding(word2)

    cos_sim = cosine_similarity(word1_embedding.reshape(1, -1), word2_embedding.reshape(1, -1))[0][0]

    return cos_sim


def get_disambiguation(statement, entity_name, prev_results):
    candidates = get_topic(entity_name)
    final_results = []

    def process_candidate(e):
        jaccard_similarity = 0
        if prev_results:
            prevs = prev_results[-1]
            page_a_url = prevs[0][2]
            page_b_url = e.url
            page_a_words = PreProcessing.extract_text_and_tokenize(page_a_url)
            page_b_words = PreProcessing.extract_text_and_tokenize(page_b_url)
            intersection_size = len(page_a_words.intersection(page_b_words))
            union_size = len(page_a_words.union(page_b_words))
            jaccard_similarity += intersection_size / union_size
        if jaccard_similarity > 0.9:
            jaccard_similarity = 0
        score = 0.4 * cosine_sim(entity_name, e.title) + jaccard_similarity
        if not prev_results:
            score += cosine_sim(statement, e.summary)
        return score, e

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_candidate = {executor.submit(process_candidate, e): e for e in candidates}
        best_score = 0
        best_e = None
        best_url = ""
        for future in as_completed(future_to_candidate):
            score, e = future.result()
            if score > best_score:
                best_score = score
                best_e = e
                best_url = e.url
                print(f'当前最分数最高的：{e},{e.url}, {best_score}, {e.summary[:50]}')

    if best_e:
        final_results.append((entity_name, best_e, best_url))
    return final_results


if __name__ == '__main__':
    # q = 'Where will the 2024 Olympic Games be held?'
    # a = 'Paris'
    q = 'Which company develops IPhone?'
    a = 'Apple Inc'
    original_answer = "It is an Apple Inc. product.Who developed the first iPhone? Steve Jobs and Steve Wozniak created the original model, the 1G in 2007. The current models are made by Foxconn and other contractors who produce hardware, along with AT&T's Bell Laboratories that designed most of Apple's telephony capabilities; there isn't just one company developing this device but rather many companies working together as partners for a single goal - making phones better than ever before! What is the name of the first iPhone?The original iPhone was released in 2007 and was called the iPhone 1G.What country manufactures IPhone?Apple's iPhones are made by Foxconn, a Taiwanese company that has factories in China as well as elsewhere around Asia (and sometimes Europe). The company also makes other products for Apple including MacBook laptops and iPods among others so if you need anything else besides just phones from this tech giant then contact us now before stock runs out again next week!How is IPhone manufactured? All over China in large buildings staffed entirely by robots who work."
    # q = 'Which company produces IPhone?'
    # a='Apple'
    t1 = time.time()
    # qna_entity_linking_dbpedia(q,a)
    print(qna_entity_linking_wikipedia(q,a,original_answer))
    # new way

    t2 = time.time()
    print('Time consumed:',t2-t1,' seconds')






