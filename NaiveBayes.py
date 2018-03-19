from os import listdir
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from pprint import pprint
from decimal import *
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
		self.pos_accuracy = 0
		self.neg_accuracy = 0

	def append_docs(self, directory):
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

	def remove_word(self, word, doc):
		if word in doc:
			doc.remove(word)
		return doc

	def get_path(self, filename, pos_or_neg, base_path):
		path = ""
		if pos_or_neg == 'neg':
			path = base_path + 'neg/' + filename
		elif pos_or_neg == 'pos':
			path = base_path + 'pos/' + filename
		else:
			print("There was an error")
		return path

	def predict(self, directory, pos_or_neg):
		num_correct = 0
		total_guesses = 0
		pos_predictions = 0
		neg_predictions = 0 
		equal_pos_neg = 0
		for filename in listdir(directory):
			probabilities_pos = []
			probabilities_neg = []
			base_path = 'imdb/test/'
			path = self.get_path(filename, pos_or_neg, base_path)
			current_doc = load_doc(path)
			tokens = clean_doc(current_doc)
			stemmed_doc = stem_doc(tokens)
			doc = Document(stemmed_doc)
			frequency_table = csv_to_dict('full_frequency_table.csv')
			getcontext().prec = 500
			for word in doc.bag_of_words:
				if word in frequency_table:
					word_freq_pos = frequency_table[word][0]
					word_freq_neg = frequency_table[word][1]
					weighted_prob_pos = Decimal(word_freq_pos / self.num_positive)
					weighted_prob_neg = Decimal(word_freq_neg / self.num_negative)
					probabilities_pos.append(weighted_prob_pos)
					probabilities_neg.append(weighted_prob_neg)
						
			total_prob_pos = Decimal(len(self.positive_documents) / len(self.all_documents))
			total_prob_neg = Decimal(len(self.negative_documents) / len(self.all_documents))
			probabilities_pos.append(total_prob_pos)
			probabilities_neg.append(total_prob_neg)
			
			prob_pos = 1
			prob_neg = 1

			for i in range(len(probabilities_pos)):
				prob_pos *= Decimal(probabilities_pos[i])
			for i in range(len(probabilities_neg)):
				prob_neg *= Decimal(probabilities_neg[i])
			
			for word in doc.bag_of_words:
				p_x = 1
				if word in frequency_table:
					p_x *= Decimal((frequency_table[word][0] + frequency_table[word][1]) / len(self.all_documents))

			prob_pos = Decimal(prob_pos / p_x)
			prob_neg = Decimal(prob_neg / p_x)

			if prob_pos > prob_neg:
				pos_predictions += 1
			elif prob_neg > prob_pos:
				neg_predictions += 1
			else: 
				equal_pos_neg += 1

		if pos_or_neg == 'pos':
			self.pos_accuracy = pos_predictions / (pos_predictions + neg_predictions)
		else:
			self.neg_accuracy = neg_predictions / (pos_predictions + neg_predictions)

vocab = Counter()
process_docs('imdb/train/pos', vocab)
process_docs('imdb/train/neg', vocab)
process_docs('imdb/test/neg', vocab)
process_docs('imdb/test/pos', vocab)
min_occurance = 3
tokens = [k for k,c in vocab.items() if c >= min_occurance]
save_list(tokens, 'vocab.txt')
bayes = NaiveBayes('vocab.txt')
bayes.append_docs('imdb/train/pos')
bayes.append_docs('imdb/train/neg')
bayes.calc_frequencies()
bayes.frequency_table = csv_to_dict('frequency_table.csv')
bayes.predict('imdb/test/pos', 'pos')
bayes.predict('imdb/test/neg', 'neg')
print(bayes.pos_accuracy)
print(bayes.neg_accuracy)