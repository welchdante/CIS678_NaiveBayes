from os import listdir
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from pprint import pprint
import string
import re
import csv

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def dict_to_csv(dictionary):
	with open("frequency_table.csv", "w") as f:
		csv.writer(f).writerows((k,) + v for k, v in dictionary.items())

def csv_to_dict(filename):
	out_dict = {}
	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			out_dict[row[0]] = int(row[1]), int(row[2])
	return out_dict

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
 
def add_all_docs_to_vocab(directory, vocab):
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

class Document():
	def __init__(self, list_of_words):
		self.bag_of_words = {}
		self.size = 0

		for word in list_of_words:
			if word in self.bag_of_words:
				self.bag_of_words[word] += 1
			else:
				self.bag_of_words[word] = 1

			self.size += 1

	def get_word_count(self, word):
		res = 0
		if(word in self.bag_of_words):
			res = self.bag_of_words[word]
		return res

class NaiveBayes():
	def __init__(self, filename):
		self.positive_documents = []
		self.negative_documents = []
		self.all_documents = []
		self.num_positive = 0
		self.num_negative = 0
		self.current_doc = []
		self.vocabulary = load_doc(filename).split()
		self.frequency_table = {}
		self.frequency_table = self.frequency_table.fromkeys(self.vocabulary)

	def iterate_docs(self, directory):
		for filename in listdir(directory):
			path = directory + '/' + filename
			self.count_total(directory)
			document = self.get_document(path)
			if "pos" in directory:
				self.positive_documents.append(document)
			elif "neg" in directory:
				self.negative_documents.append(document)
			else:
				"There was an error"
			self.all_documents.append(document)

	def get_document(self, path):
		doc = load_doc(path)
		tokens = clean_doc(doc)
		stemmed_doc = stem_doc(tokens)
		return stemmed_doc

	def count_total(self, directory):
		if "pos" in directory:
			self.num_positive += 1
		elif "neg" in directory:
			self.num_negative += 1
		else:
			print("There was an error")

	def calc_frequencies(self):
		for word in self.vocabulary:
			pos_word_count = 0
			neg_word_count = 0

			for i in range(len(self.positive_documents)):
				current_doc = Document(self.positive_documents[i])
				pos_word_count += current_doc.get_word_count(word)
				self.positive_documents[i] = self.remove_word(word, self.positive_documents[i])

			for i in range(len(self.negative_documents)):
				current_doc = Document(self.negative_documents[i])
				neg_word_count += current_doc.get_word_count(word)
				self.negative_documents[i] = self.remove_word(word, self.negative_documents[i])

			self.frequency_table[word] = (pos_word_count, neg_word_count)

		dict_to_csv(self.frequency_table)

			#prob_pos = pos_word_count / self.num_positive
			#prob_neg = neg_word_count / self.num_negative

			#print(word + "\t\t\t positive: %3f \t negative: %3f" % (prob_pos, prob_neg))

	def remove_word(self, word, doc):
		if word in doc:
			doc.remove(word)
		return doc


# bayes = NaiveBayes('vocab.txt')
  bayes.iterate_docs('practice/train/pos')
  bayes.iterate_docs('practice/train/neg')
# bayes.calc_frequencies()

