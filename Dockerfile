FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk python3.10 python3-pip && \
    apt-get clean;


ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

RUN pip3 install --no-cache-dir \
    nltk==3.8.1 \
    spacy==3.7.2 \
    scikit-learn==1.3.0 \
    wikipedia-api==0.6.0 \
    transformers==4.35.2 \
    scipy==1.9.3 \
    wikipedia==1.4.0 \
    sparqlwrapper==2.0.0 \
    stanford-openie==1.3.1 \
    ctransformers==0.2.27  \
    python-Levenshtein==0.23.0


RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_md
RUN python3 -c "from ctransformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-GGUF')"


RUN python3 -m nltk.downloader punkt

WORKDIR /app

COPY . /app
RUN unset DEBIAN_FRONTEND

CMD ["python3", "main.py", "--i", "demo_standard_input.txt"]
