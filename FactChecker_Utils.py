import re

from PreProcessing import remove_punctuation, contains_wh_word, replace_wh_word

def question_to_statement(question: str, answer: str) -> object:
    if answer == "yes":
        return remove_punctuation(question), True
    elif answer == "no":
        return remove_punctuation(question), False
    else:
        if contains_wh_word(question):
            question = replace_wh_word(question, answer)
            return remove_punctuation(question), True
        else:
            return remove_punctuation(question) + ' ' + answer, True


def find_relation_and_check_of(text, relation):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() == relation:
            start_index = max(i, 0)
            end_index = min(i + 3, len(words))
            if "of" in words[start_index:end_index]:
                return True
    if relation.endswith("ed"):
        return True
    return False


def triplets_preprocessing(triplets, text, entity_linking_res):
    for triple in triplets:
        if find_relation_and_check_of(text, triple['relation']):
            tmp_sbj = triple['subject']
            tmp_obj = triple['object']
            triple['subject'] = tmp_obj
            triple['object'] = tmp_sbj
        subject_entity = []
        object_entity = []
        for item in entity_linking_res:
            for text, _, url in item:
                if text == triple['subject']:
                    subject_entity = url
                if text == triple['object']:
                    object_entity = url
        if subject_entity:
            triple['subject'] = extract_last_segment(subject_entity)
        if object_entity:
            triple['object'] = extract_last_segment(object_entity)
    return triplets


def extract_last_segment(url):
    match = re.search(r'/([^/]+)/*$', url)

    if match:
        last_segment = match.group(1)
        last_segment = re.sub(r'[^a-zA-Z0-9_]', '', last_segment)
        return last_segment
    else:
        return ""