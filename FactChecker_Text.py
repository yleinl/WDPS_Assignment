import requests
import wikipediaapi
from scipy.spatial.distance import cosine

from FactChecker_KB import bert_encode
from FactChecker_Utils import question_to_statement
from RelationExtractor import extract_triplets
from PreProcessing import text_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_minimal_segment(sentence, relation, object):
    # subj_idx = sentence.find(subject)
    rel_idx = sentence.find(relation)
    obj_idx = sentence.find(object)

    # if subj_idx == -1 or rel_idx == -1 or obj_idx == -1:
    if rel_idx == -1 or obj_idx == -1:
        return None

    start_idx = min(rel_idx, obj_idx)
    end_idx = max(rel_idx + len(relation), obj_idx + len(object))
    return sentence[start_idx:end_idx]


def fact_checker_text(question, answer, entity_linking_res):
    return max(fact_checker_text_sbj(question, answer, entity_linking_res),
               fact_checker_text_obj(question, answer, entity_linking_res))


def fact_checker_text_sbj(question, answer, entity_linking_res):
    statement, yesno = question_to_statement(question, answer)
    triplets = extract_triplets(statement)
    confidence = []
    explanation = []
    queried_sim = []
    for triple in triplets:
        url = ""
        for item in entity_linking_res:
            for text, _, url_linked in item:
                if text == triple['subject']:
                    url = url_linked
        response = requests.get(url)
        if response.status_code != 200:
            continue
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="wdps_assignment/1.0 (https://github.com/yleinl/WDPS)"
        )
        page_name = url.rsplit('/', 1)[-1]
        page = wiki_wiki.page(page_name)
        if not page.exists():
            # print(f"Wikipedia page for {triple['subject']} does not exist.")
            continue
        sentences = page.text.split('.')
        subject = triple['subject'].replace("_", " ")
        relation = triple['relation'].replace("_", " ")
        object = triple['object'].replace("_", " ")
        if relation == "wikiPageWikiLink":
            relation = "is"
        relevant_sentences = [s for s in sentences if relation in s and object in s]
        if relevant_sentences:
            return 1
        relevant_sentences = [s for s in sentences if object in s]
        if not relevant_sentences:
            print(
                f"No sentences found in the Wikipedia page for {triple['subject']} containing {triple['object']}.")
            continue
        for relevant_result in relevant_sentences:
            res_triples = extract_triplets(relevant_result)
            for res_triple in res_triples:
                if res_triple['object'] == triple['object']:
                    res_relation_embedding = bert_encode(res_triple['relation'])
                    relation_embedding = bert_encode(triple['relation'])
                    cosine_sim = 1 - cosine(res_relation_embedding.squeeze().detach().numpy(),
                                            relation_embedding.squeeze().detach().numpy())
                    queried_sim.append(cosine_sim)

    if not queried_sim:
        return 0
    else:
        return max(queried_sim)


def fact_checker_text_obj(question, answer, entity_linking_res):
    statement, yesno = question_to_statement(question, answer)
    triplets = extract_triplets(statement)
    queried_sim = []
    for triple in triplets:
        url = ""
        for item in entity_linking_res:
            for text, _, url_linked in item:
                if text == triple['subject']:
                    url = url_linked
        response = requests.get(url)
        if response.status_code != 200:
            continue
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="wdps_assignment/1.0 (https://github.com/yleinl/WDPS)"
        )
        page_name = url.rsplit('/', 1)[-1]
        page = wiki_wiki.page(page_name)
        if not page.exists():
            # print(f"Wikipedia page for {triple['subject']} does not exist.")
            continue
        sentences = page.text.split('.')
        subject = triple['subject'].replace("_", " ")
        relation = triple['relation'].replace("_", " ")
        object = triple['object'].replace("_", " ")
        if relation == "wikiPageWikiLink":
            relation = "is"
        relevant_sentences = [s for s in sentences if relation in s and object in s]
        if relevant_sentences:
            return 1
        relevant_sentences = [s for s in sentences if object in s]
        if not relevant_sentences:
            print(
                f"No sentences found in the Wikipedia page for {triple['object']} containing {triple['subject']}")
            continue
        for relevant_result in relevant_sentences:
            res_triples = extract_triplets(relevant_result)
            for res_triple in res_triples:
                # if res_triple['subject'] == triple['object'] or res_triple in ['he', 'she', 'it']:
                if res_triple['object'] == triple['subject']:
                    res_relation_embedding = bert_encode(res_triple['relation'])
                    relation_embedding = bert_encode(triple['relation'])
                    cosine_sim = 1 - cosine(res_relation_embedding.squeeze().detach().numpy(),
                                            relation_embedding.squeeze().detach().numpy())
                    queried_sim.append(cosine_sim)

    if not queried_sim:
        return 0
    else:
        return max(queried_sim)


if __name__ == '__main__':
    q = 'Who develops IPhone?'
    a = 'Apple Inc'
    original_answer = "It is an Apple Inc. product.Who developed the first iPhone? Steve Jobs and Steve Wozniak created the original model, the 1G in 2007. The current models are made by Foxconn and other contractors who produce hardware, along with AT&T's Bell Laboratories that designed most of Apple's telephony capabilities; there isn't just one company developing this device but rather many companies working together as partners for a single goal - making phones better than ever before! What is the name of the first iPhone?The original iPhone was released in 2007 and was called the iPhone 1G.What country manufactures IPhone?Apple's iPhones are made by Foxconn, a Taiwanese company that has factories in China as well as elsewhere around Asia (and sometimes Europe). The company also makes other products for Apple including MacBook laptops and iPods among others so if you need anything else besides just phones from this tech giant then contact us now before stock runs out again next week!How is IPhone manufactured? All over China in large buildings staffed entirely by robots who work."
    entity_linking = [[('Apple Inc', 'kong', 'https://en.wikipedia.org/wiki/Apple_Inc.')],
                      [('IPhone', 'kong', 'https://en.wikipedia.org/wiki/IPhone')],
                      [('Apple Inc.', 'kong', 'https://en.wikipedia.org/wiki/Apple_Inc.')],
                      [('Steve Wozniak', 'kong', 'https://en.wikipedia.org/wiki/Steve_Wozniak')],
                      [('Foxconn', 'kong', 'https://en.wikipedia.org/wiki/Foxconn')],
                      [("AT&T's Bell Laboratories", 'kong', 'https://en.wikipedia.org/wiki/AT%26T_Laboratories')],
                      [('China', 'kong', 'https://en.wikipedia.org/wiki/China')],
                      [('Asia', 'kong', 'https://en.wikipedia.org/wiki/Asia')],
                      [('Apple', 'kong', 'https://en.wikipedia.org/wiki/Apple')],
                      [('MacBook', 'kong', 'https://en.wikipedia.org/wiki/MacBook')],
                      [('China', 'kong', 'https://en.wikipedia.org/wiki/China')]]

    cosine_similarity = fact_checker_text(q, a, entity_linking)
    print(cosine_similarity)
