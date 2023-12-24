import time

from ctransformers import AutoModelForCausalLM
import AnswerExtract
import EntityLinker
from FactChecker import FactChecker
import numpy as np
import re
import argparse

repository = "TheBloke/Llama-2-7B-GGUF"
model_file = "llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")


def process_question(line, f_output):
    groups = re.match(r"(.*)[\t](.*)", line).groups()
    line_starter = groups[0]
    prompt = groups[1]
    print("Computing the answer (can take some time)...")
    completion = llm(prompt)
    print("COMPLETION: %s" % completion)
    f_output.writelines([line_starter, '\t', 'R"', completion, '"\n'])
    # extract answer
    extract = AnswerExtract.extract_answer(prompt, completion)
    f_output.writelines([line_starter, '\t', 'A"', extract, '"\n'])
    # entity linking
    EL = EntityLinker.qna_entity_linking_wikipedia(prompt, extract, completion)
    # fact checking
    correctness = FactChecker(prompt, extract, EL)
    f_output.writelines([line_starter, '\t', 'C"', correctness, '"\n'])
    EL = np.squeeze(EL)
    for ent in EL:
        f_output.writelines([line_starter, '\t', 'E"', ent[0], '"\t"', ent[2], '"\n'])
    return


def main(input_file):
    f_output = open("output.txt", "a+", encoding='utf-8')
    f_output.truncate(0)
    with open(input_file, "r", encoding='utf-8') as f_input:
        line = f_input.readline()
        while line:
            process_question(line, f_output)
            line = f_input.readline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input file.')
    parser.add_argument('--i', type=str, default='demo_standard_input.txt',
                        help='Input file name (default: demo_standard_input.txt)')
    args = parser.parse_args()
    main(args.i)
