from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import svm
import matplotlib.pyplot as plt
import random
import numpy as np

class Evaluator(object):

	def __init__(self, char_embed_dict, pca_file_name, tsne_file_name, print_pca_var_ratio, show_figures, list_wrong_chars, include_unseen_as_neutral, tsne_epochs, print_unseen_chars, svm_c, svm_val_slice_num):
		
		self.char_embed_dict = char_embed_dict
		self.pca_file_name = pca_file_name
		self.tsne_file_name = tsne_file_name
		self.print_pca_var_ratio = print_pca_var_ratio
		self.show_figures = show_figures
		self.include_unseen_as_neutral = include_unseen_as_neutral
		self.tsne_epochs = tsne_epochs
		self.print_unseen_chars = print_unseen_chars
		self.svm_c = svm_c
		
		self.build_char_groups()
		self.char_arr, self.embed_arr = self.build_data_arr()
		self.ground_truth_arr = self.build_ground_truth_arr()
		self.label_arr, self.centroids = self.k_means()

		# Here we compute the accuracies of different label matches and select the better one.
		acc_tuple1 = self.get_accuracy(label_for_imperial = 0)
		acc_tuple2 = self.get_accuracy(label_for_imperial = 1)
		final_acc_tuple = acc_tuple1 if acc_tuple1[0] > acc_tuple2[0] else acc_tuple2

		print ('\nimperial color:\t', final_acc_tuple[4])
		print ('rebel color:\t', self.the_other_color(final_acc_tuple[4]))
		print ('\nAccuracy with neutral characters as correct: ', final_acc_tuple[0])
		print ('Accuracy with neutral correcters excluded:   ', final_acc_tuple[1])

		self.plot_pca()
		self.plot_tsne()
		
		if list_wrong_chars:
			print ('\nimperial characters mispredicted to be rebel:')
			print (final_acc_tuple[2])
			print ('\nrebel characters mispredicted to be imperial:')
			print (final_acc_tuple[3])

		self.svm(val_slice_num = svm_val_slice_num)

	def svm(self, val_slice_num):

		svm_label_list = []
		svm_embed_list = []
		
		for char_idx in range(len(self.char_arr)):
			if self.ground_truth_arr[char_idx] != 2:
				svm_label_list += [self.ground_truth_arr[char_idx].tolist()]
				svm_embed_list += [self.embed_arr[char_idx].tolist()]

		svm_char_num = len(svm_label_list)
		svm_slice_size = svm_char_num // val_slice_num

		if self.svm_c == None:
			c_list = [1.0, 3.0, 9.0, 27.0, 81.0, 243.0]
			correct_pred_list = [0] * len(c_list)
		else:
			correct_pred = 0

		for slice_idx in range(val_slice_num):
			is_last = slice_idx == val_slice_num - 1
			start_idx = slice_idx * svm_slice_size
			end_idx = (slice_idx + 1) * svm_slice_size if not is_last else svm_char_num

			test_x = svm_embed_list[start_idx: end_idx]
			test_y = svm_label_list[start_idx: end_idx]
			train_x = svm_embed_list[0:start_idx] + svm_embed_list[end_idx:] if not is_last else svm_embed_list[0:start_idx]
			train_y = svm_label_list[0:start_idx] + svm_label_list[end_idx:] if not is_last else svm_label_list[0:start_idx]

			if self.svm_c == None:

				for c_idx in range(len(c_list)):
					svm_model = svm.SVC(C=c_list[c_idx], cache_size=200, class_weight=None, coef0=0.0, \
						decision_function_shape='ovr', \
						degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, \
						shrinking=True, tol=0.001, verbose=False)
					svm_model.fit(train_x, train_y)
					correct_pred_list[c_idx] += sum(np.equal(np.array(svm_model.predict(test_x)), test_y))
			
			else:
				svm_model = svm.SVC(C=self.svm_c, cache_size=200, class_weight=None, coef0=0.0, \
						decision_function_shape='ovr', \
						degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, \
						shrinking=True, tol=0.001, verbose=False)
				svm_model.fit(train_x, train_y)
				correct_pred += sum(np.equal(np.array(svm_model.predict(test_x)), test_y))

		if self.svm_c != None:
			print ('\nAccuracy of SVM with c = %f:\t' % self.svm_c, correct_pred / svm_char_num)
		else:
			print ('\n')
			for pair in zip(c_list, correct_pred_list):
				print ('Accuracy of SVM with c = %f:\t' % pair[0], pair[1] / svm_char_num)

	def the_other_color(self, color):

		the_other_color = 'blue' if color == 'red' else 'red'

		return the_other_color

	def get_accuracy(self, label_for_imperial):

		correct = 0
		neutral = 0

		char_num = len(self.char_arr)
		
		missed_imperial_list = []
		missed_rebel_list = []
		
		for char_idx in range(char_num):
			
			label, truth = self.label_arr[char_idx], self.ground_truth_arr[char_idx]

			if truth == 2:
				neutral += 1
				continue
						
			if (label == label_for_imperial and truth == 0) or (label != label_for_imperial and truth == 1):
				correct += 1
			else:
				if truth == 0:
					missed_imperial_list += [self.char_arr[char_idx]]
				else:
					missed_rebel_list += [self.char_arr[char_idx]]

		imperial_color = 'blue' if label_for_imperial == 1 else 'red'

		return (correct + neutral) / char_num, correct / (char_num - neutral), missed_imperial_list, missed_rebel_list, \
				imperial_color

	def build_char_groups(self):

		self.rebel = ('BASE VOICE', 'CONTROL OFFICER', 'MAN', 'PORKINS', 'REBEL OFFICER', 'RED ELEVEN',
             'RED TEN', 'RED SEVEN', 'RED NINE', 'RED LEADER', 'BIGGS', 'GOLD LEADER',
             'WEDGE', 'GOLD FIVE', 'REBEL', 'DODONNA', 'CHIEF', 'TECHNICIAN', 'WILLARD',
             'GOLD TWO', 'MASSASSI INTERCOM VOICE', 'THREEPIO', 'HAN and LUKE', 'HAN/PILOT', 'HAN', 'LUKE', 'LEIA', 'THREEPIO', 'SECOND THREEPIO', 'YODA', 'REBEL FIGHTER', 'REBEL PILOT', 'REBEL CAPTAIN', 'BEN', 'Y-WING PILOT', 'ANAKIN', 'LANDO')

		self.imperial = ('CAPTAIN', 'CHIEF PILOT', 'TROOPER', 'OFFICER', 'DEATH STAR INTERCOM VOICE', 
                'FIRST TROOPER', 'SECOND TROOPER', 'FIRST OFFICER', 'OFFICER CASS', 'TARKIN',
                'INTERCOM VOICE', 'MOTTI', 'TAGGE', 'TROOPER VOICE', 'ASTRO-OFFICER',
                'VOICE OVER DEATH STAR INTERCOM', 'SECOND OFFICER', 'GANTRY OFFICER', 
                'WINGMAN', 'IMPERIAL OFFICER', 'COMMANDER', 'VOICE', 'VADER', 'DEATH STAR CONTROLLER', 'IMPERIAL SOLDIER', 'EMPEROR')
		
		self.neutral = ('WOMAN', 'BERU', 'CREATURE', 'DEAK', 'OWEN', 'BARTENDER', 'CAMIE', 'JABBA',
               'AUNT BERU', 'GREEDO', 'NEUTRAL', 'HUMAN', 'FIXER')

		self.legal_chars = self.rebel + self.imperial + self.neutral
		
	def build_ground_truth_arr(self):

		ground_truth_list = []
    	
		for char in self.char_arr:
			if char in self.imperial:
				ground_truth_list += [0]
			elif char in self.rebel:
				ground_truth_list += [1]
			elif char in self.neutral:
				ground_truth_list += [2]
			elif self.include_unseen_as_neutral:
				ground_truth_list += [2]
				self.unseen_chars += [char]
			else:
				raise Exception('The character %s does not exist!' % char)

		if self.print_unseen_chars:
			print ('These are unseen characters in the character embedding dictionay:', self.unseen_chars)

		return np.array(ground_truth_list)

	def build_data_arr(self):

		if self.include_unseen_as_neutral:
			char_arr = np.array(list(self.char_embed_dict.keys()))
			self.unseen_chars = []
		else:
			char_arr = np.array([char for char in list(self.char_embed_dict.keys()) if char in self.legal_chars])
			self.unseen_chars = [char for char in list(self.char_embed_dict.keys()) if char not in self.legal_chars]

		embed_arr = np.array([self.char_embed_dict[key] for key in char_arr])

		return char_arr, embed_arr

	def k_means(self):

		kmeans = KMeans(n_clusters = 2)
		labels = kmeans.fit_predict(self.embed_arr)
		centroids = kmeans.cluster_centers_

		return np.array(labels), np.array(centroids)

	def plot_pca(self):

		pca = PCA(n_components = 2)
		pca_outcome = pca.fit_transform(self.embed_arr)
		variance_ratio = pca.explained_variance_ratio_

		if self.print_pca_var_ratio:
			print ('\nvariance ratio of the first two dimension of PCA: ', variance_ratio)

		marker_list = ['o', 'x', 'v'] # The last one is for neutral characters.
		color1 = 'b'
		color2 = 'r'

		fig, ax = plt.subplots()

		for char_idx in range(len(pca_outcome)):
			x, y = tuple(pca_outcome[char_idx])
			color = color1 if self.label_arr[char_idx] == 1 else color2 
			marker = marker_list[self.ground_truth_arr[char_idx]]

			ax.scatter([x], [y], c = color, marker = marker, label = ['yes'])
			ax.annotate(self.char_arr[char_idx], (x, y), size = 7)
			ax.set_xlabel('first dimension')
			ax.set_ylabel('second dimension')
			ax.set_title('PCA outcome')

		fig.savefig(self.pca_file_name)

		if self.show_figures:
			plt.show()

	def plot_tsne(self):
		
		tsne = TSNE(n_components = 2, verbose = 0, perplexity = 40, n_iter = self.tsne_epochs)
		tsne_outcome = tsne.fit_transform(self.embed_arr)

		marker_list = ['o', 'x', 'v'] # The last one is for neutral characters.
		color1 = 'b'
		color2 = 'r'

		fig, ax = plt.subplots()

		for char_idx in range(len(tsne_outcome)):
			x, y = tuple(tsne_outcome[char_idx])
			color = color1 if self.label_arr[char_idx] == 1 else color2 
			marker = marker_list[self.ground_truth_arr[char_idx]]

			ax.scatter([x], [y], c = color, marker = marker, label = ['yes'])
			ax.annotate(self.char_arr[char_idx], (x, y), size = 7)
			ax.set_xlabel('first dimension')
			ax.set_ylabel('second dimension')
			ax.set_title('t-SNE outcome')

		fig.savefig(self.tsne_file_name)

		if self.show_figures:
			plt.show()


if __name__ == '__main__':

	pass

	# Evaluator(char_embed_dict = test_char_embed_dict, pca_file_name = 'pca_file.pdf', tsne_file_name = 'tsne_file.pdf', print_pca_var_ratio = True, show_figures = True, list_wrong_chars = True, include_unseen_as_neutral = True)