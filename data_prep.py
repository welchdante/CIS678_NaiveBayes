from os import listdir
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from pprint import pprint
import string
import re

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def clean_doc(doc):
	tokens = doc.split()
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	tokens = [word for word in tokens if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

def add_doc_to_vocab(filename, vocab):
	doc = load_doc(filename)
	tokens = clean_doc(doc)
	stemmed_doc = stem_doc(tokens)
	vocab.update(stemmed_doc)
 
def process_docs(directory, vocab):
	for filename in listdir(directory):
		path = directory + '/' + filename
		add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def stem_doc(tokens):
	ps = PorterStemmer()
	stemmed_doc = []
	for word in tokens:
		stemmed_word = ps.stem(word)
		stemmed_doc.append(stemmed_word)
	return stemmed_doc

vocab = Counter()
process_docs('imdb/train/pos', vocab)
process_docs('imdb/train/neg', vocab)
#print(vocab.most_common(50))
#print(len(vocab))
min_occurance = 3
tokens = [k for k,c in vocab.items() if c >= min_occurance]
#print(len(tokens))
save_list(tokens, 'vocab.txt')
