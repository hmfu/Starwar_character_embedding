import os
import numpy as np
import tensorflow as tf
import pickle as pkl
from utils import *
from Starwar_character_embedding import GloveModel
import re
import math
from util_evaluate import Evaluator

class Data_formater(object):

	def __init__(self):
		pass

	def load_data(self, file_name_list, use_cent_norm):

		self.char_script_tup_list_list = []

		for file_name in file_name_list:

			char_script_tup_list = []
			with open(file_name) as f:
				next(f)

				for line in f:
					splitted = line.rstrip().split('"')
					token = splitted[3]
					script = re.sub('[^a-zA-Z]+', ' ', splitted[5])[:-1].lower()

					pos, neg, neu = calculate_senti_score(script)
					raw_score = 0.0 if math.isnan(pos - neg) else pos - neg
					score = raw_score * len(script.split(' ')) ** 0.5 if use_cent_norm else raw_score

					char_script_tup_list += [(token, score)]

			self.char_script_tup_list_list += [char_script_tup_list]

	def build_char2idx(self):

		self.char_set = set([tup[0] for tup_list in self.char_script_tup_list_list for tup in tup_list])
		self.char_num = len(self.char_set)

		self.char2idx = {}
		next_idx = 0

		for char in self.char_set:
			self.char2idx[char] = next_idx
			next_idx += 1

		self.idx2char = {v:k for k, v in self.char2idx.items()}

	def build_score_mat(self, use_cent_norm, use_dist_discount):

		self.score_mat = np.zeros(shape = (self.char_num, self.char_num))
		self.count_mat = np.zeros(shape = (self.char_num, self.char_num))

		for char_script_tup_list in self.char_script_tup_list_list:
			for idx in range(len(char_script_tup_list) - 1):
				
				char = char_script_tup_list[idx][0]
				score = char_script_tup_list[idx][1]

				next_char = None
				next_score = None

				next_iter = idx + 1
				while True:
					if next_iter == len(char_script_tup_list):
						break
					if char_script_tup_list[next_iter][0] != char:
						next_char = char_script_tup_list[next_iter][0]
						next_score = char_script_tup_list[next_iter][1]
						break
					next_iter += 1

				if next_char:
					dist = (next_iter - idx)
					real_score = (score + next_score) / dist if use_dist_discount else score + next_score
					self.score_mat[self.char2idx[char], self.char2idx[next_char]] += real_score
					self.score_mat[self.char2idx[next_char], self.char2idx[char]] += real_score
					self.count_mat[self.char2idx[char], self.char2idx[next_char]] += 2 / dist if use_cent_norm else 2
					self.count_mat[self.char2idx[next_char], self.char2idx[char]] += 2 / dist if use_cent_norm else 2

		self.epsilon = 0.0001
		self.count_mat += self.epsilon
		self.score_mat = self.score_mat / self.count_mat ** 0.5 if use_cent_norm else self.score_mat / self.count_mat

		self.max_score = np.amax(self.score_mat)
		for idx in range(self.char_num):
			self.score_mat[idx, idx] = self.max_score

		# This part is for checking only.
		rebel = ('BASE VOICE', 'CONTROL OFFICER', 'MAN', 'PORKINS', 'REBEL OFFICER', 'RED ELEVEN',
             'RED TEN', 'RED SEVEN', 'RED NINE', 'RED LEADER', 'BIGGS', 'GOLD LEADER',
             'WEDGE', 'GOLD FIVE', 'REBEL', 'DODONNA', 'CHIEF', 'TECHNICIAN', 'WILLARD',
             'GOLD TWO', 'MASSASSI INTERCOM VOICE', 'THREEPIO', 'HAN and LUKE', 'HAN/PILOT', 'HAN', 'LUKE', 'LEIA', 'THREEPIO', 'SECOND THREEPIO', 'YODA', 'REBEL FIGHTER', 'REBEL PILOT', 'REBEL CAPTAIN', 'BEN', 'Y-WING PILOT', 'ANAKIN', 'LANDO')

		imperial = ('CAPTAIN', 'CHIEF PILOT', 'TROOPER', 'OFFICER', 'DEATH STAR INTERCOM VOICE', 
                'FIRST TROOPER', 'SECOND TROOPER', 'FIRST OFFICER', 'OFFICER CASS', 'TARKIN',
                'INTERCOM VOICE', 'MOTTI', 'TAGGE', 'TROOPER VOICE', 'ASTRO-OFFICER',
                'VOICE OVER DEATH STAR INTERCOM', 'SECOND OFFICER', 'GANTRY OFFICER', 
                'WINGMAN', 'IMPERIAL OFFICER', 'COMMANDER', 'VOICE', 'VADER', 'DEATH STAR CONTROLLER', 'IMPERIAL SOLDIER', 'EMPEROR')
		
		neutral = ('WOMAN', 'BERU', 'CREATURE', 'DEAK', 'OWEN', 'BARTENDER', 'CAMIE', 'JABBA',
               'AUNT BERU', 'GREEDO', 'NEUTRAL', 'HUMAN', 'FIXER')

		ranked_idx = np.argsort(self.score_mat, axis = None)
		
		for idx in ranked_idx:
			
			char1_idx, char2_idx = idx // self.char_num, idx % self.char_num
			char1, char2 = self.idx2char[char1_idx], self.idx2char[char2_idx]
			
			if (char1 in rebel and char2 in rebel) or (char1 in imperial and char2 in imperial):
				relation = 1
			elif (char1 in rebel and char2 in imperial) or (char1 in imperial and char2 in rebel):
				relation = -1
			else:
				relation = 0

			print ('%d\t%s\t%s\n\t\t\t%f\t%f' \
					% (relation, char1, char2, self.score_mat[char1_idx, char2_idx], self.count_mat[char1_idx, char2_idx]))

	def svd_embed(self, embed_dim):
		
		u, s, _ = np.linalg.svd(self.score_mat, full_matrices = False)
		embed_table = np.matmul(u, np.diag(s))[:, :embed_dim]

		Evaluator(char_embed_dict = {self.idx2char[idx]: embed_table[idx] for idx in range(self.char_num)}, \
				pca_file_name = 'pca_by_senti.pdf', \
				tsne_file_name = 'tsne_by_senti.pdf', \
				print_pca_var_ratio = False, \
				show_figures = False, \
				list_wrong_chars = False, \
				include_unseen_as_neutral = False, \
				tsne_epochs = 500, \
				print_unseen_chars = True, \
				svm_c = None, \
				svm_val_slice_num = 5)

	def build_data(self):

		char_list1 = []
		char_list2 = []
		weight_list = []
		label_list = []

		for char_idx1 in range(self.char_num):
			for char_idx2 in range(char_idx1 + 1):
				if self.score_mat[char_idx1, char_idx2] != self.epsilon:
					score = self.score_mat[char_idx1, char_idx2] + 1.0
					char_list1 += [[char_idx1]]
					char_list2 += [[char_idx2]]
					weight_list += [[math.exp(score - (self.max_score + 1.0))]]
					label_list += [[score]]


		char_arr1 = np.array(char_list1)
		char_arr2 = np.array(char_list2)
		weight_arr = np.array(weight_list)
		label_arr = np.array(label_list)

		self.data_tup = (char_arr1, char_arr2, weight_arr, label_arr)

if __name__ == '__main__':

	file_name_list = ['SW_EpisodeIV.txt', 'SW_EpisodeV.txt', 'SW_EpisodeVI.txt']
	file_name_list = [os.path.join('star-wars-movie-scripts/', file_name) for file_name in file_name_list]

	data_formater = Data_formater()
	data_formater.load_data(file_name_list = file_name_list, use_cent_norm = 1)
	data_formater.build_char2idx()
	data_formater.build_score_mat(use_cent_norm = 1, use_dist_discount = 1)
	data_formater.svd_embed(embed_dim = 20)
	data_formater.build_data()

	glove_model = GloveModel()
	glove_model.buildModel(embedDim = 10, tokenNum = data_formater.char_num)
	glove_model.trainModel(learningRate = 0.01, epochNum = 500, evalEpochNum = 50, dataTup = data_formater.data_tup)
	glove_model.saveEmbedding(fileName = 'senti_table_tup.pkl', tokDict = data_formater.idx2char)