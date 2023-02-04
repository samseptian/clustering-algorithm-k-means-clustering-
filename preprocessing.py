import json
import numpy as np

from string import punctuation
from nltk.tokenize import word_tokenize

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

sastrawi_stopword_factory = StopWordRemoverFactory()
sastrawi_stemmer_factory = StemmerFactory()

sastrawi_stemmer = sastrawi_stemmer_factory.create_stemmer()
sastrawi_stopwords = sastrawi_stopword_factory.get_stop_words()


def TfidfWeighting(dataset):
	vocabulary = dataset['vocabulary']
	docs = dataset['preprocessing']['stemmed']

	n_features = len(vocabulary)
	n_samples = len(docs)

	tf = np.zeros((n_samples, n_features))
	for i, doc in enumerate(docs):
		for word in set(doc):
			index = vocabulary.index(word)
			tf[i][index] = doc.count(word) / len(doc)

	df = np.zeros((n_samples, n_features))
	idf = np.empty(n_features, dtype=float)
	for v, word in enumerate(vocabulary):
		df_v = 1
		for w, doc in enumerate(docs):
			if word in doc:
				df_v += 1
			df[w][v] = df_v
		idf[v] = np.log10(n_samples / df_v)

	tfidf = tf * idf

	dataset.update({
		'weight': {
			'tf': tf,
			'df': df,
			'tfidf': tfidf
		}
	})


def Preprocessing(dataset):
	stemmer = sastrawi_stemmer
	stopwords = sastrawi_stopwords + list(punctuation)

	with open('cache/manual_stopwords.txt') as f:
		stopwords += f.read().splitlines()

	with open('cache/stemmed_cache.json') as f:
		stem = json.loads(f.read())

	with open('cache/corrected_stem.json') as f:
		correct_stem = json.loads(f.read())

	lowers = []
	tokenized = []
	non_stopwords = []
	stemmed = []
	vocabulary = []
	for text in dataset['raws']:

		# case folding
		lower = text.lower()
		lowers.append(lower)

		# tokenizing
		tokens = word_tokenize(lower)
		tokenized.append(tokens)

		# stopword removal
		non_sw = [word for word in tokens if word not in stopwords and not word.isnumeric()]
		non_stopwords.append(non_sw)

		# stemming
		roots = []
		for word in non_sw:
			try:
				root = stem[word]
			except KeyError:
				root = stemmer.stem(word)
				stem[word] = root
			if not root:
				continue
			try:
				root = correct_stem[root]
			except KeyError:
				pass
			roots.append(root)
		stemmed.append(roots)

		for root in roots:
			if root not in vocabulary:
				vocabulary.append(root)

	with open('cache/stemmed_cache.json', 'w') as f:
		f.write(json.dumps(stem, indent=4))

	vocabulary = sorted(vocabulary)

	dataset.update({
		'vocabulary': vocabulary,
		'preprocessing': {
			'lowers': lowers,
			'tokenized': tokenized,
			'non_stopwords': non_stopwords,
			'stemmed': stemmed,
		}
	})
	TfidfWeighting(dataset)
