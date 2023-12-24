import time

from FactChecker_KB import fact_checker_kbs
from FactChecker_Text import fact_checker_text
from FactChecker_Utils import question_to_statement


def FactChecker(question, extracted_answer, entity_link_res):
    text_info_confidence = fact_checker_text(question, extracted_answer, entity_link_res)
    triplets, scores, graph_info_confidence = fact_checker_kbs(question, extracted_answer, entity_link_res)
    statement, yesno = question_to_statement(question, extracted_answer)
    if text_info_confidence > 0.9 or graph_info_confidence > 0.9:
        if yesno:
            return "correct"
        else:
            return "incorrect"
    else:
        return "incorrect"

