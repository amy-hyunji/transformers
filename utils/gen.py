from pattern3.en import*
import nltk
import argparse
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
import random

# convert to wordnet pos
def tag_map(pos):
	_pos = pos[0]
	if (_pos == "J"): return wn.ADJ
	elif (_pos == "V"): return wn.VERB
	elif (_pos == "R"): return wn.ADV
	else: return wn.NOUN

"""
return tokenList, tagDict, lemmaDict 
tokenList = list of tokens
tagDict = dict with {key: token, val: tag}
lemmaDict = dict with {key: token, val: lemma}
"""
def preprocess(tokens, debug=False):
	tokenList = []
	tagDict = dict()
	lemmaDict = dict()

	_lemma = WordNetLemmatizer()
	for token, tag in pos_tag(tokens):
		if (debug):
			print("token: {}, tag: {}".format(token, tag))
		tokenList.append(token)
		tagDict[token] = tag

		lem = _lemma.lemmatize(token)
		lemmaDict[token] = lem 
		if (debug):
			print(token, "=>", lem)
	return tokenList, tagDict, lemmaDict 

# input synset / return name and pos
def get_name_pos_from_syn(syn):
	name = syn.name().split('.')[0]
	pos = syn.name().split('.')[1]
	return name, pos

"""
synDict = dict with {key: token, value: synset}
"""
def make_synDict(tokenList, tagDict, lemmaDict, debug=False, thres=-1):
	synDict = dict()

	# make synset using lemma
	for token in tokenList:
		lem = lemmaDict[token]
		if(len(wn.synsets(lem)) == 0):
			continue
		else:
			synList = wn.synsets(lem)		# list of synset

			"""
			only choose synset with SAME word with lem
			"""
			tempList = []
			for elem in synList:
				name, pos = get_name_pos_from_syn(elem)
				if (name == lem):
					tempList.append(elem)

			"""
			only choose synset with SAME pos_tag 
			"""
			appendList = []
			ori_tag = tag_map(tagDict[token])
			for elem in tempList:
				_name, _pos = get_name_pos_from_syn(elem)
				if (ori_tag != _pos):
					# remove the ones that do not match
					continue
				else:
					appendList.append(elem)

			if (len(appendList) == 0):
				# no matching element with same pos
				continue
			elif (thres > 0 and len(appendList) > thres):
				continue
			else:
				synDict[token] = appendList

	return synDict

def get_tense(syn, ori_pos):
	
	"""
	VB:	Verb, base form
	VBD:	Verb, past tense
	VBG:	Verb, gerund or present participle
	VBN:	Verb, past participle
	VBP:	Verb, non-3rd person singular present
	VBZ:	Verb, 3rd person singular present
	"""

	name, pos = get_name_pos_from_syn(syn)
	
	# default values
	_tense = "present"			# infintive, present, past, future
	_person = 1						# 1, 2, 3, or None
	_number = "singular"			# SG, PL
	_mood = "indicative"			# indicative, imperative, conditional, subjunctive
	_aspect = "imperfective"	# imperfective, perfective, progressive

	if (ori_pos == "VBD"):
		_tense = "past"
	elif (ori_pos == "VBG"):
		_aspect = "progressive"
	elif (ori_pos == "VBN"):
		_tense = "past"
		_aspect = "progressive"
	elif (ori_pos == "VBZ"):
		_person = 3

	return conjugate (name,
							tense = _tense, 
							person = _person,
							number = _number,
							mood = _mood, 
							aspect = _aspect,
							negated = False)

"""
hypernymDict = dict of {key: token, value: list of hypernyms}
"""
def make_hypernymDict(synDict, tokenList, tagDict, lemmaDict, debug=False, thres=-1):
	hypernymDict = dict()

	for token in tokenList:
		if not (token in synDict.keys()):
			continue
		appendList = []
		ori_tag = tag_map(tagDict[token])
		for syn in synDict[token]:
			hyperList = syn.hypernyms()
			
			"""
			only choose synset with SAME pos_tag 
			"""
			hyper = []
			for elem in hyperList:
				_name, _pos = get_name_pos_from_syn(elem)
				if (ori_tag != _pos):
					# remove the ones that do not match
					continue
				else:
					hyper.append(elem)

			if ((thres > 0) and (len(hyper) > thres)):
				continue
			elif (len(hyper) == 0):
				continue
			else:
				for _elem in hyper:
					appendList.append(_elem)
		hypernymDict[token] = appendList

	return hypernymDict

def replace_all(sen, ori_word, ori_hyper):
	print("[Original Sentence] {}".format(sen))
	for elem in ori_hyper:
		_name, _pos = get_name_pos_from_syn(elem)
		print("[Replace Sentence] {}".format(sen.replace(ori_word, _name)))

def main(sen):
	tokens = word_tokenize(sen)
	tokenList, tagDict, lemmaDict = preprocess(tokens, debug=False)
	synDict = make_synDict(tokenList, tagDict, lemmaDict, debug=False)
	hypernymDict = make_hypernymDict(synDict, tokenList, tagDict, lemmaDict, debug=False)

	_key = list(hypernymDict.keys())
	randNum = random.randint(1, len(_key)) # choose the word to change randomly
	ori_word = _key[randNum-1]
	ori_hyper = hypernymDict[ori_word]
	replace_all(sen, ori_word, ori_hyper)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--sen", required=True, default="I am planning to do this today")
	args = parser.parse_args()
	main(args.sen)
