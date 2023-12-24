def transform(inputfile,outputfile):
    f_input = open(inputfile, "r", encoding='utf-8')
    line = f_input.readline()
    output = []
    starter = 'question-'
    n = 1
    while line:
        line_starter = starter + str(n).zfill(3)
        output.append([line_starter,'\t',line])
        line = f_input.readline()
        n += 1
    f_input.close()
    with open(outputfile, "a+", encoding='utf-8') as f_output:
        f_output.truncate(0)  # always clear all before writing to txt
        for i in output:
            f_output.writelines(i)
    f_output.close()


if __name__ == '__main__':
    transform(inputfile='example_questions.txt',outputfile='standard_input.txt')
    transform('demo_example_questions.txt','demo_standard_input.txt')