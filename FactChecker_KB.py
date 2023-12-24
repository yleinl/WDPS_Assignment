import spacy
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import RelationExtractor
import requests
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.spatial.distance import cosine

from FactChecker_Utils import question_to_statement, triplets_preprocessing

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
nlp = spacy.load("en_core_web_sm")


def get_wikidata_id_from_wikipedia(page_title):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "format": "json",
        "titles": page_title
    }
    if not page_title:
        return None
    response = requests.get(url, params=params)
    data = response.json()
    page_id = next(iter(data["query"]["pages"]))
    page_data = data["query"]["pages"][page_id]
    if 'pageprops' in page_data and 'wikibase_item' in page_data['pageprops']:
        return page_data['pageprops']['wikibase_item']
    else:
        return None


def get_wikipedia_title_from_wikidata_id(wikidata_id, language='en'):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "props": "sitelinks",
        "sitefilter": f"{language}wiki"
    }
    response = requests.get(url, params=params)
    data = response.json()
    entities = data.get("entities", {})
    if wikidata_id in entities:
        sitelinks = entities[wikidata_id].get("sitelinks", {})
        wikipedia_link = f"{language}wiki" in sitelinks and sitelinks[f"{language}wiki"].get("title")
        return wikipedia_link
    else:
        return None


def generate_dbpedia_sparql_query_relation(dbpedia_triplets):
    sparql_query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT DISTINCT ?relation
    WHERE {
    """

    conditions = []
    for triple in dbpedia_triplets:
        subject = f"dbr:{triple['subject']}"
        obj = f"dbr:{triple['object']}"
        condition = f"{subject} ?relation {obj}"
        conditions.append(condition)

    sparql_query += " UNION ".join(["{" + c + "}" for c in conditions])
    sparql_query += "}"
    return sparql_query


def generate_wikidata_sparql_query_relation(wikidata_triplets):

    sparql_query = """
    SELECT DISTINCT ?relation
    WHERE {
    """

    conditions = []
    for triple in wikidata_triplets:
        subject_id = get_wikidata_id_from_wikipedia(triple['subject'])
        object_id = get_wikidata_id_from_wikipedia(triple['object'])
        if not(subject_id and object_id):
            continue
        subject = f"wd:{subject_id}"
        obj = f"wd:{object_id}"

        condition = f"{subject} ?relation {obj}"
        conditions.append(condition)

    sparql_query += " UNION ".join(["{" + c + "}" for c in conditions])
    sparql_query += "}"
    return sparql_query


def generate_dbpedia_sparql_query_subject(dbpedia_triplets):
    sparql_query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>

    SELECT DISTINCT ?subject
    WHERE {
    """

    conditions = []
    for triple in dbpedia_triplets:
        relation = f"dbo:{triple['relation']}"
        obj = f"dbr:{triple['object']}"

        if relation != "is":
            condition = f"?subject {relation} {obj}"
            conditions.append(condition)
    sparql_query += " UNION ".join(["{" + c + "}" for c in conditions])
    sparql_query += "} LIMIT 10"
    # print("subject", sparql_query)
    return sparql_query


def generate_dbpedia_sparql_query_object(dbpedia_triplets):
    sparql_query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>

    SELECT DISTINCT ?object
    WHERE {
    """

    conditions = []
    for triple in dbpedia_triplets:
        subject = f"dbr:{triple['subject']}"
        relation = f"dbo:{triple['relation']}"

        if relation != "is":
            condition = f"{subject} {relation} ?object"
            conditions.append(condition)

    sparql_query += " UNION ".join(["{" + c + "}" for c in conditions])
    sparql_query += "} LIMIT 10"
    return sparql_query


def generate_sparql_query(triplets_query):
    return (generate_dbpedia_sparql_query_relation(triplets_query),
            generate_wikidata_sparql_query_relation(triplets_query),
            generate_dbpedia_sparql_query_subject(triplets_query),
            generate_dbpedia_sparql_query_object(triplets_query))


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def query_dbpedia_relation(sparql_query):
    endpoint_url = "http://dbpedia.org/sparql"
    session = requests_retry_session()
    try:
        response = session.get(endpoint_url, params={'query': sparql_query, 'format': 'json'})
        if response.ok:
            results = response.json()
            relations = []
            for result in results["results"]["bindings"]:
                if 'relation' in result:
                    full_uri = result['relation']['value']
                    last_part = full_uri.split('/')[-1]
                    relations.append(last_part)
            return relations
        else:
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def query_wikidata_relation(sparql_query):
    endpoint_url = "https://query.wikidata.org/sparql"
    session = requests_retry_session()
    try:
        response = session.get(endpoint_url, params={'query': sparql_query, 'format': 'json'})
        if response.ok:
            results = response.json()
            relations = []
            for result in results["results"]["bindings"]:
                if 'relation' in result:
                    full_uri = result['relation']['value']
                    last_part = full_uri.split('/')[-1]
                    relations.append(last_part)
            return relations
        else:
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def query_dbpedia_subject(sparql_query):
    endpoint_url = "http://dbpedia.org/sparql"
    session = requests_retry_session()
    try:
        response = session.get(endpoint_url, params={'query': sparql_query, 'format': 'json'})
        if response.ok:
            results = response.json()
            subjects = []
            for result in results["results"]["bindings"]:
                if 'subject' in result:
                    full_uri = result['subject']['value']
                    last_part = full_uri.split('/')[-1]
                    subjects.append(last_part)
            return subjects
        else:
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def query_dbpedia_object(sparql_query):
    endpoint_url = "http://dbpedia.org/sparql"
    session = requests_retry_session()
    try:
        response = session.get(endpoint_url, params={'query': sparql_query, 'format': 'json'})
        if response.ok:
            results = response.json()
            objects = []
            for result in results["results"]["bindings"]:
                if 'object' in result:
                    full_uri = result['object']['value']
                    last_part = full_uri.split('/')[-1]
                    objects.append(last_part)
            return objects
        else:
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def query_knowledge_bases(dbpedia_query_relation, dbpedia_query_subject, dbpedia_query_object, wikidata_query):
    dbpedia_relations = query_dbpedia_relation(dbpedia_query_relation)
    dbpedia_subject = query_dbpedia_subject(dbpedia_query_subject)
    dbpedia_object = query_dbpedia_object(dbpedia_query_object)
    wikidata_relations = query_wikidata_relation(wikidata_query)
    for i in range(len(wikidata_relations)):
        wikidata_relation = wikidata_relations[i]
        modified_relation = get_wikipedia_title_from_wikidata_id(wikidata_relation)
        if modified_relation == "Property_talk":
            modified_relation = "is"
        wikidata_relations[i] = modified_relation
    combined_relations = list(set(dbpedia_relations + wikidata_relations))
    combined_relations = [relation for relation in combined_relations if relation]
    return combined_relations, dbpedia_subject, dbpedia_object


def bert_encode(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def fact_checker_dbs(check_triplets, dbpedia_query_relation, dbpedia_query_subject,
                     dbpedia_query_object, wikidata_query):
    kbs_relations, kbs_subject, kbs_object = query_knowledge_bases(dbpedia_query_relation, dbpedia_query_subject,
                                                                   dbpedia_query_object, wikidata_query)
    scores_relation = []
    scores_subject = []
    scores_object = []
    for triple in check_triplets:
        triple_relation = triple['relation']
        triple_subject = triple['subject']
        triple_object = triple['object']
        triple_relation_embedding = bert_encode(triple_relation)

        for db_relation in kbs_relations:
            db_embedding = bert_encode(db_relation)
            cosine_sim = 1 - cosine(triple_relation_embedding.squeeze().detach().numpy(),
                                    db_embedding.squeeze().detach().numpy())
            scores_relation.append((triple_relation, db_relation, cosine_sim))

        for db_subject in kbs_subject:
            db_embedding = bert_encode(db_subject)
            cosine_sim = 1 - cosine(triple_relation_embedding.squeeze().detach().numpy(),
                                    db_embedding.squeeze().detach().numpy())
            scores_subject.append((triple_subject, db_subject, cosine_sim))

        for db_object in kbs_object:
            db_embedding = bert_encode(db_object)
            cosine_sim = 1 - cosine(triple_relation_embedding.squeeze().detach().numpy(),
                                    db_embedding.squeeze().detach().numpy())
            scores_object.append((triple_object, db_object, cosine_sim))

    return scores_relation, scores_subject, scores_object


def fact_checker_kbs(question, answer, entity_linking_result):
    text, yesno = question_to_statement(question, answer)
    extracted_triplets = triplets_preprocessing(RelationExtractor.extract_triplets(text), text, entity_linking_result)
    if not extracted_triplets:
        return [], [], 0
    dbpedia_query_relation, wikidata_query_relation, dbpedia_query_subject, dbpedia_query_object = (
        generate_sparql_query(extracted_triplets))
    evaluated_scores = fact_checker_dbs(extracted_triplets, dbpedia_query_relation, dbpedia_query_subject,
                                        dbpedia_query_object, wikidata_query_relation)
    evaluated_scores = [item for item in evaluated_scores if item]

    is_empty = all(not sublist for sublist in evaluated_scores)
    highest_kb_score = 0
    if not is_empty:
        for each_scores in evaluated_scores:
            for each_score in each_scores:
                if each_score[2]:
                    if each_score[2] > highest_kb_score:
                        highest_kb_score = each_score[2]
    else:
        highest_kb_score = 0
    return extracted_triplets, evaluated_scores, highest_kb_score


if __name__ == '__main__':
    q = 'Who is the husband of Jill Biden ?'
    a = 'Joe Biden'
    entity_linking = [[('Joe Biden', 'kong', 'https://en.wikipedia.org/wiki/Joe_Biden')], [('Jill Biden', 'kong', 'https://en.wikipedia.org/wiki/Jill_Biden')], [('husband', 'kong', 'https://en.wikipedia.org/wiki/Husband')]]
    triplets, scores, highest_score = fact_checker_kbs(q, a, entity_linking)
    print(triplets, scores, highest_score)