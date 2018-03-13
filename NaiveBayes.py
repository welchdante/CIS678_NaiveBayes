class Document():
	def __init__(self, text):
		self.bag_of_words = {}
		self.size = 0
		text = text.split()

		for word in text:
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
	def __init__(self, thing):
		self.thing = thing


doc1 = Document("hate shit sue")
doc2 = Document("love happy bob")