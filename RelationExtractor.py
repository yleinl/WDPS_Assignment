import spacy
from openie import StanfordOpenIE

nlp = spacy.load("en_core_web_sm")


def extract_triplets(sentence):
    # 定义停用词列表
    stopwords = {'is', 'of', 'in', 'at', 'by', 'for', 'with', 'the', 'a', 'an', 'and', 'to','was'}

    triples = []
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        text = sentence
        for triple in client.annotate(str(text)):
            cleaned_relation = ' '.join([word for word in triple['relation'].split() if word.lower() not in stopwords])

            if cleaned_relation:
                triple['relation'] = cleaned_relation
                triples.append(triple)
            else:
                triple['relation'] = "is"
                triples.append(triple)

    return triples


if __name__ == '__main__':
    print(extract_triplets("Donald is the father of Ivanka"))