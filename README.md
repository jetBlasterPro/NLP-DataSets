# NLP-DataSets

### Aim: 1a Convert the given text to speech.

### Requirements:

```python
pip install nltk
pip install gtts
pip install --upgrade wheel
pip install playsound
```

### Program:
```python
from gtts import gTTS
from playsound import playsound

mytext = "Hello World"
lang = "en"

myobj = gTTS(text = mytext, lang = lang, slow = False)

myobj.save("./myFile.mp3")
playsound("./myFile.mp3")
```

### Output:

myFile.mp3 audio file is getting created and it plays the file with playsound() method, while running the program

### Aim: 1b Convert audio file to Text.
### Requirements: 
``` python
pip install SpeechRecognition
```
### Program:
``` python
import speech_recognition as sr

filename = "./myFile.wav"
r = sr.Recognizer()

with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    text = r.recognize_google(audio_data)
    print(text)
```
### Output:
![[Pasted image 20240510114236.png]]


### Aim: 2a Study of Brown Corpus with various methods like filelds, raw, words, sents, categories.
### Requirements:
```python
pip install nltk
nltk.download("brown")
```
### Program:
```python
import nltk
nltk.download("brown")
from nltk.corpus import brown

print("File ids of brown corpus\n", brown. fileids())

ca01 = brown.words("ca01")
print("\nca01 has following words:\n", ca01)
print("\nca01 has ", len(ca01), " words")

print("\n\nCategories or file in brown corpus: \n")
print(brown.categories())

print("\n\nStatistics for each text:\n")
print("Avg-Word-Len\tAvg-Sentence-Len\tNo.-of-Times-Each-Word-Appears-On-Avg\t\tFile-Name")
for fileid in brown.fileids():
	num_chars = len(brown.raw(fileid))
	num_words = len(brown.words(fileid))
	num_sents = len(brown.sents(fileid))
	num_vocab = len(set([w.lower() for w in brown.words(fileid)]))
	print(int(num_chars/num_words), "\t\t\t", num_words/num_sents, "\t\t\t", int(num_words/num_vocab), "\t\t\t", fileid)
```
### Output:
![[Pasted image 20240510120743.png]]![[Pasted image 20240510120828.png]]

### Aim: 2b Create and use your own corpora (plaintext, categorical)
### Requirements:
```python
pip install nltk
nltk.download("punkt")
```
#### Creating user defined corpus
1. In same program directory create new folder with name `user-defined-corpus-txt`
2. Open `user-defined-corpus-txt` and create new `txt` file with any name here I have given `KP2B401.txt`
3. Open the `txt` file and enter any sentence here I have given `The Quick Brown Fox Jumpped Over The Lazy Dog`.
![[Pasted image 20240510123931.png]]
### Program:
```python
import nltk
nltk.download("punkt")
from nltk.corpus import PlaintextCorpusReader
corpus_root = "./user-defined-corpus-txt"
fileList = PlaintextCorpusReader(corpus_root, ".*")

print("\n File List \n")
print(fileList.fileids())
print(fileList.root)
print("\n\nStatistics for each text:\n")
print("Avg_Word_Len\tAvg_Sentence_Len\tno._of_Times_Each_Word_Appears_On_Avg\tFile_Name")

for fileid in fileList.fileids():
	num_chars = len(fileList.raw(fileid))
	num_words = len(fileList.words(fileid))
	num_sents = len(fileList.sents(fileid))
	num_vocab = len(set([w.lower() for w in fileList.words(fileid)]))
	print(int(num_chars/num_words), "\t\t\t", int(num_words/num_sents), "\t\t\t", int(num_words/num_vocab), "\t\t\t", fileid)
```
### Output:
![[Pasted image 20240510122934.png]]

### Aim: 2c Study of tagged corpora with methods like tagged_sents, tagged_words.
### Requirements:
```python
pip install nltk
import nltk
nltk.download('punkt')
nltk.download('words')
```
### Program:
```python
import nltk
nltk.download('punkt')
nltk.download('words')
from nltk import tokenize

para = "Hello World! From kirti college. Today we will be learning NLTK."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n=========\n", sents)

print("\nword tokenization\n========\n")
for index in range(len(sents)):
    words = tokenize.word_tokenize(sents[index])
    print(words)
```
### Output:
![[Pasted image 20240517123002.png]]

### Aim: 2d Map Words to Properties Using Python Dictionaries
### Program:
```python
thisdict={
    "brand":"Porsche",
    "model":"911 gt3 rs",
    "year":2003
    }

print(thisdict)
print(thisdict["brand"])
print(len(thisdict))
print(type(thisdict))
```
### Output:
![[Pasted image 20240517124341.png]]


### Aim: 3a Study Default Tagger
### Requirements:
```python
pip install nltk
import nltk
nltk.download('treebank')
```
### Program:
```python
import nltk
nltk.download('treebank')
from nltk.tag import DefaultTagger
from nltk.corpus import treebank
exptagger = DefaultTagger("NN")
testsentences=treebank.tagged_sents()[1000:]
print(exptagger.accuracy(testsentences))
print(exptagger.tag_sents([['Hey',','],['How', 'are', 'you','?']]))
```
### Output:
![[Pasted image 20240517125434.png]]

### Aim: 3b Study Unigram Tagger

### Requirements:
```python
pip install nltk
import nltk
nltk.download('treebank')
```
### Program:
```python
from nltk.tag import UnigramTagger
from nltk.corpus import treebank

train_sents = treebank.tagged_sents()[:10]
tagger = UnigramTagger(train_sents)
print(treebank.sents()[0])
print('\n',tagger.tag(treebank.sents()[0]))

tagger.tag(treebank.sents()[0])
tagger = UnigramTagger(model={'Pierre':'NN'})
print('\n',tagger.tag(treebank.sents()[0]))
```
### Output:
![[Pasted image 20240517130043.png]]


### Aim: 4a Study of Wordnet Dictionary with methods as synsets, definitions, examples, antonyms
### Requirements:
```python
pip install nltk
nltk.download("wordnet")
```
### Program: 
```python
import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet
print(wordnet.synsets("computer"))
print(wordnet.synset("computer.n.01").definition())
print("Examples: ",wordnet.synset("computer.n.01").examples())
print(wordnet.lemma("buy.v.01.buy").antonyms())
```
### Output:
![[op-1.png]]

### Aim: 4b Write a program using python to find synonym and antonym of word "active" using Wordnet.
### Requirements:
```python
pip install nltk
nltk.download("wordnet")
```
### Program:
```python
from nltk.corpus import wordnet
print(wordnet.synsets("active"))
print(wordnet.lemma('active.a.01.active').antonyms())
```
### Output:
![[op-4b.png]]

### Aim: 4c Compare two nouns
### Requirements:
```python
pip install nltk
nltk.download("wordnet")
```
### Program:
```python
import nltk
from nltk.corpus import wordnet
syn1 = wordnet.synsets("cricket")
syn2 = wordnet.synsets("hokey")
for s1 in syn1:
    for s2 in syn2:
        print("Path similarity of: ")
        print(s1, "(",s1.pos(),")","[",s1.definition(),"]")
        print(s1, "(",s2.pos(),")","[",s2.definition(),"]")
        print("is", s1.path_similarity(s2))
```
### Output:
![[op-4c.png]]


### Aim: 5a Tokenization using Python’s split() function
### Program:
```python
text = """ This tool is an a beta stage. Alexa developers can use Get Metrics API to seamlessly analyse
metric. It also supports custom skill model, prebuilt Flash Briefing model, and the Smart Home Skill API.
You can use this tool for creation of monitors, alarms, and dashboards that spotlight changes. The
release of these three tools will enable developers to create visual rich skills for Alexa devices with
screens. Amazon describes these tools as the collection of tech and tools for creating visually rich and
interactive voice experiences. """
data = text.split('.')
for i in data:
    print (i)
```
### Output:
![[op-5a.png]]

### Aim: 5b Tokenization using Regular Expressions (RegEx)
### Requirements:
```python
pip install nltk
```
### Program:
```python
import nltk
from nltk.tokenize import RegexpTokenizer
tk = RegexpTokenizer('\s+',gaps = True)
str = "Let's use RegexpTokenizer to split"
tokens = tk.tokenize(str)
print(tokens)
```
### Output:
![[op-5b.png]]

### Aim: 5c Tokenization using NLTK
### Requirements:
```python
pip install nltk
```
### Program:
```python
import nltk
from nltk.tokenize import word_tokenize
str = "Let's use word_tokenize to split"
print(word_tokenize(str))
```
### Output:
![[eop1.png]]

### Aim: 5d Tokenization using Spacy
### Requirements:
```python
pip install spacy
```
### Program:
```python
import spacy
nlp = spacy.blank("en")
str = "Let's use spacy to split"
doc = nlp(str)
words = [word.text for word in doc]
print(words)
```
### Output:
![[op-5d.png]]

### Aim: 5e Tokenization using Keras
### Requirements:
```python
pip install tensorflow
pip install keras
pip install Keras-Preprocessing
```
### Program:
```python
import keras
from keras.preprocessing.text import text_to_word_sequence
str = "Let's use keras to split"
tokens = text_to_word_sequence(str)
print(tokens)
```
### Output:
![[Pract_5c.png]]

### Aim: 5f Tokenization using Gensim
### Requirements:
```python
pip install gensim

gensim requires C++ 
```
### Program:
```python
from gensim.utils import tokenize
str = "I love to study Natural Language Processing in Python"
list(tokenize(str))
```
### Output:
![[Pract_5-gensim.png]]


### Aim: 6a Named Entity recognition using user defined text.
### Requirements:
```python
pip install spacy
python -m spacy download en_core_web_sm
```
### Program:
```python
import spacy
nlp = spacy.load("en_core_web_sm")
text = ("When Sebastian Thrun started working on self-driving cars at " 
"Google in 2007, few people outside of the company took him " 
"seriously. “I can tell you very senior CEOs of major American " 
"car companies would shake my hand and turn away because I wasn’t " 
"worth talking to,” said Thrun, in an interview with Recode earlier " 
"this week.")

doc = nlp(text)
print("Noun phrases: ",[chunk.text for chunk in doc.noun_chunks])
print("Verbs: ",[token.lemma_ for token in doc if token.pos_=="VERB"])
```
### Output:
![[Pasted image 20240517122148.png]]

### Aim: 6b Named Entity recognition with diagram using NLTK corpus – treebank.
### Requirements:
```python
pip install nltk
nltk.download('treebank')
```
### Program:
```python
import nltk
nltk.download('treebank')
from nltk.corpus import treebank_chunk
treebank_chunk.tagged_sents()[0]
treebank_chunk.chunked_sents()[0]
treebank_chunk.chunked_sents()[0].draw()
```
### Output:
![[Pasted image 20240523015720.png]]


### Aim: 7a Define grammar using nltk. Analyze a sentence using the same
### Requirements:
```python
pip install nltk
import nltk
nltk.download("punkt")
nltk.download("treebank")
```
### Program:
```python
import nltk
from nltk import tokenize
nltk.download("punkt")
nltk.download("treebank")
grammar1 = nltk.CFG.fromstring("""
S -> VP
VP -> VP NP
NP -> Det NP
Det -> 'that'
NP -> singular Noun
NP -> 'flight'
VP -> 'Book'
""")
sentence = "Book that flight"
for index in range(len(sentence)):
    all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)
parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
    print(tree)
    tree.draw()
```
### Output:
![[7aop1.PNG]]

### Aim: 7b Implementation of Deductive Chart Parsing using context free grammar and a given sentence.
### Requirements:
```python
pip install nltk
```
### Program:
```python
import nltk
from nltk import tokenize
grammar1 = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'a' | 'my'
N -> 'bird' | 'balcony'
V -> 'saw'
P -> 'in'
""")
sentence = "I saw a bird in my balcony"
for index in range (len(sentence)):
    all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)
parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
    print(tree)
    tree.draw()
```
### Output:
![[7bop1.PNG]]
![[7bop2.PNG]]

### Aim: 8 Study PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer Study WordNetLemmatizer
### Requirements:
```python
pip install nltk
import nltk
nltk.download("wordnet")
```
### Program:
```python
import nltk
nltk.download("wordnet")
word = "running"
print("PoterStemmer")
from nltk.stem import PorterStemmer
word_stemmer = PorterStemmer()
print(word_stemmer.stem(word))

print("---------------------------------")
print("Lancaster Stemmer")
from nltk.stem import LancasterStemmer
Lanc_stemmer = LancasterStemmer()
print(Lanc_stemmer.stem(word))

print("---------------------------------")
print('RegexpStemmer')
import nltk
from nltk.stem import RegexpStemmer
Reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
print(Reg_stemmer.stem(word))


print("---------------------------------")
print('SnowballStemmer')
from nltk.stem import SnowballStemmer
english_stemmer = SnowballStemmer('english')
print(english_stemmer.stem (word))

print("---------------------------------")
print('WordNetLemmatizer')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("word :\tlemma")
print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))

print("better :", lemmatizer.lemmatize("better", pos ="a"))
```
### Output:
![[8op1.png]]


### Aim: 9 Finite Automata
### Program:
```python
def FA(s):
    if len(s) < 3:
        return "Rejected"

    if s[0] == '1':
        if s[1] == '0':
            if s[2] == '1':
                for i in range(3, len(s)):
                    if s[i] != "1":
                        return "Rejected"
                return "Accepted"
            return "Rejected"
        return "Rejected"
    return "Rejected"
        
inputs=['1','10101','101','10111','01010','100','','10111101','1011111']
for i in inputs:
    print(i, FA(i))
```
### Output:
![[op9.png]]
