# Introduction
This is the assignment for the course Web Data Processing System.
Contributors: Yitao Lei, Ivy Rui Wang, Zhongyu Shi
Tester: Delong Yuan
## Modules
- [x] Answer Extraction
  - [x] Question Classifier 
  - [x] Answer Extraction
- [x] Fact Checking
  - [x] KBs Search
  - [x] Text Search
  - [x] Final Fact Checker
- [x] Entity Linking
  - [x] Named Entity Recognition
  - [x] Entity Disambiguation
# Environment
## Language
python 3.10  
java 8 (To use Stanford NLP)
## Dependency
nltk==3.8.1 \
spacy==3.7.2 \
scikit-learn==1.3.0 \
wikipedia-api==0.6.0 \
transformers==4.35.2 \
scipy==1.9.3 \
wikipedia==1.4.0 \
sparqlwrapper==2.0.0 \
stanford-openie==1.3.1 \
ctransformers==0.2.27 \
python-Levenshtein==0.23.0  
# Run
To run the WDPS assignment, build by the docker file(https://github.com/yleinl/WDPS/blob/main/Dockerfile) attached in the project.
Please change the ```demo_standard_input.txt``` content to your test data.

```bash
docker build -t wdps_group9
docker run -t wdps_group9
```

or you can

```bash
docker build -t wdps_group9
docker run -it wdps_group9 /bin/bash
cd app
wget ****/your_data.txt
python3 main.py --i your_data.txt
```





