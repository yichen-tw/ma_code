from collections import Counter
import datetime
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

#-----------------------------------------------------------------------------------------------------------------------
#	Our code derives from
#	BERT Fine-Tuning Tutorial with PyTorch: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
#	written by Chris McCormick and Nick Ryan
#-----------------------------------------------------------------------------------------------------------------------
# Data Preprocessing
#-----------------------------------------------------------------------------------------------------------------------

""" Get the Real World Worry Data from the CSV file """
""" Encode RWWD emotion labels  """
def load_rwwd_data(data_path, emotion_deletes, emotion_substitutes, do_reorder_emotions):	
	n_classes = []

	rwwd_df = pd.read_csv(data_path, usecols=["chosen_emotion", "text_long", "text_short"])
	n_classes = rwwd_df['chosen_emotion'].unique()
	print("")
	print("COVID-19 Real World Worry Dataset")
	print("")
	print("There are {} Emotion Classes: ".format(len(n_classes)), n_classes)
	print("")
	print(rwwd_df['chosen_emotion'].value_counts(sort=True))
	print("")
	
	if emotion_deletes or emotion_substitutes:
		rwwd_df, n_classes = modify_classes(rwwd_df, emotion_deletes, emotion_substitutes, n_classes)
	
	chosen_emotion = rwwd_df.chosen_emotion.to_list()
	text_long = rwwd_df.text_long.to_list()
	text_short = rwwd_df.text_short.to_list()

	print("Number of Long Texts: ", len(text_long))
	print("Number of Short Texts: ", len(text_short))
	print("")

	""" 
	Encode RWWD emotion labels
	Replace the categorical values with numeric values
	"""
	if do_reorder_emotions:
		# encode so that related emotions group into a spectrum
		print("Reorder Emotion Classes")
		emotions = ['Disgust',
					'Anger',
					'Fear',
					'Anxiety',
					'Sadness',
					'Desire',
					'Relaxation',
					'Happiness']
		for emo in emotion_deletes:
			if emo in emotions:
				emotions.remove(emo)
		for emo in emotion_substitutes.keys():
			if emo in emotion_substitutes:
				emotions.remove(emo)
		n_classes = emotions
		encoded_chosen_emotion = [emotions.index(emo) for emo in chosen_emotion]
		print("New Order: ", n_classes)
		print("")
		print("Encoded Examples:")
		print(chosen_emotion[:10])
		print(encoded_chosen_emotion[:10])
		print("")
	else:
		label_encoder = LabelEncoder()
		encoded_chosen_emotion = label_encoder.fit_transform(list(chosen_emotion))
		encoded_chosen_emotion = encoded_chosen_emotion.tolist()
		n_classes = list(label_encoder.classes_)
		print("Emotions Order: ", n_classes)
		print("")
		print("Encoded Examples:")
		print(chosen_emotion[:10])
		print(encoded_chosen_emotion[:10])
		print("")

	return encoded_chosen_emotion, text_long, text_short, n_classes


""" Modify Emotion Classes """
def modify_classes(rwwd_df, emotion_deletes, emotion_substitutes, n_classes):

	if len(emotion_deletes) > 0:
		print("Dropping {} Emotion Class(es): ".format(len(emotion_deletes)), emotion_deletes)
		for emo in emotion_deletes:
			if emo in rwwd_df['chosen_emotion'].values:
				rwwd_df = rwwd_df.drop(rwwd_df.index[rwwd_df['chosen_emotion'] == emo])
				n_classes = rwwd_df['chosen_emotion'].unique()
		print("There are {} Emotion Classes: ".format(len(n_classes)), n_classes )
		print("")
		print(rwwd_df['chosen_emotion'].value_counts(sort=True))
		print("")
        
	if len(emotion_substitutes) > 0:
		print("Remapping {} Emotion Class(es): ".format(len(emotion_substitutes)), emotion_substitutes)
		rwwd_df['chosen_emotion'] = rwwd_df['chosen_emotion'].apply(lambda emo: emotion_substitutes[emo] if emo in emotion_substitutes.keys() else emo)
		n_classes = rwwd_df['chosen_emotion'].unique()
		print("There are {} Emotion Classes: ".format(len(n_classes)), n_classes )
		print("")
		print(rwwd_df['chosen_emotion'].value_counts(sort=True))
		print("")

	return rwwd_df, n_classes


""" Balance Class Distribution of Datasets """
def resampling(texts, rwwd_labels, do_resampling):

	y = rwwd_labels
	X = np.array([texts])
	X = X.transpose()

	print("Original Class Distribution")
	print(Counter(y))
	print("")

	if "RO" in do_resampling:
		oversample = RandomOverSampler(sampling_strategy={0:428, 2:108, 3:68, 5:160}, random_state=42)
		X_over, y_over = oversample.fit_resample(X, y)
		
		print("After Oversampling")
		print(Counter(y_over))
		print("")

		y = y_over
		X = X_over

	if "RU" in do_resampling:
		undersample = RandomUnderSampler(sampling_strategy={1:714}, random_state=42)
		X_under, y_under = undersample.fit_resample(X, y)

		print("After Undersampling")
		print(Counter(y_under))
		print("")

		y = y_under
		X = X_under

	rwwd_labels = y
	texts = X
	texts = texts.transpose()
	texts = texts.ravel()
	texts = texts.tolist()
	
	return texts, rwwd_labels


""" Get sum total of samples for each emotion class """
def display_count_labels(all_labels_list, n_classes):
	all_labels_dict = {}
	all_labels_list.sort()
	
	for label in all_labels_list:
		for emo in n_classes:
			if label == n_classes.index(emo):
				all_labels_dict[emo] = all_labels_dict.get(emo, 0) + 1

	df_total_labels = pd.DataFrame.from_dict(all_labels_dict, orient='index', columns=['Number of Samples'])
	print(df_total_labels)
	print("Total Number of Texts   ", len(all_labels_list))
	print("")


#-----------------------------------------------------------------------------------------------------------------------
# Data splitting & BERT Input Data Formatting
#-----------------------------------------------------------------------------------------------------------------------

""" Split data into train and test sets """
def get_train_test_split(X_texts, y_labels, split_mod, n_splits):

	if split_mod == "TTS_v1":
		print("Random Train/Test Split (Version 1)")
		X_train, X_val = [], []
		y_train, y_val = [], []

		for text, label in zip(X_texts, y_labels):
			train_text, val_text, train_label, val_label = train_test_split(text, label, test_size=0.1, random_state=22)

			X_train = X_train + train_text
			X_val = X_val + val_text

			y_train = y_train + train_label
			y_val = y_val + val_label

		return X_train, X_val, y_train, y_val

	elif split_mod == "TTS_v2":
		print("Random Train/Test Split (Version 2)")
		X_train, X_val = [], []
		y_train, y_val = [], []

		X_train, X_val, y_train, y_val = train_test_split(X_texts, y_labels, test_size=0.1, random_state=22)

		return X_train, X_val, y_train, y_val

	else:
		if split_mod == "KF":
			print("K-Fold")
			kf = KFold(n_splits=n_splits, shuffle=True, random_state=6)
			tt_split = kf.split(X_texts, y_labels)

		if split_mod == "SKF":
			print("Stratified K-Fold")
			skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
			tt_split = skf.split(X_texts, y_labels)

		if split_mod == "SS":
			print("Shuffle Split")
			ss = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=8)
			tt_split = ss.split(X_texts, y_labels)

		if split_mod == "SSS":
			print("Stratified Shuffle Split")
			sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=9)
			tt_split = sss.split(X_texts, y_labels)
		
		return tt_split


""" Transform dataset into BERT/Pytorch format """
def transform_dataset(tokenizer, texts, labels):
	# Tokenize into BERT encodings and convert to tensors
	max_len = 0

	encodings = tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
	input_ids = encodings['input_ids']
	print("Number of Tokenized Texts: ", len(input_ids))
	
	for ids in input_ids:
		max_len = max(max_len, len(ids))
	print("Max text length: ", max_len)
	print("")
	attention_masks = encodings['attention_mask']
	labels = torch.tensor(labels)
	tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

	return tensor_dataset


""" BERT Tokenization """
def encode_tokens(tokenizer, texts, labels):
	max_len = 0

	for text in texts:
		input_ids = tokenizer.encode(text, add_special_tokens=True)
		max_len = max(max_len, len(input_ids))
	print("")
	print("Max token sequence length: ", max_len, " / Limit imposed: 512")

	if max_len > 512:
		max_len = 512
		print("")
		print("Set max length to ", max_len, " for BERT tokenizer")

	input_ids = []
	attention_masks = []

	for text in texts:
		encoded_dict \
		= tokenizer.encode_plus(
			text,
			add_special_tokens = True,
			max_length = max_len,
			pad_to_max_length = True,
			truncation = True,
			return_tensors = 'pt'
		)
		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)
	print("")
	print('Original: ', texts[0])
	print('Token IDs: ', input_ids[0])
	print('Attention Mask: ', attention_masks[0])
	tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

	return tensor_dataset


""" DataLoaders for training and validation sets """
def get_dataloaders(batch_size, train_dataset, evaluation_dataset):
		if train_dataset is not None:
			train_loader \
			= DataLoader(
				train_dataset,
				sampler = RandomSampler(train_dataset), # Select batches randomly
				batch_size = batch_size # Defined above: Train with this batch size
				)
		if evaluation_dataset is not None:
			evaluation_loader \
			= DataLoader(
				evaluation_dataset, 
				sampler = SequentialSampler(evaluation_dataset), # Pull out batches sequentially
				batch_size = batch_size # Defined above: Evaluate with this batch size
				)

		return train_loader, evaluation_loader

#-----------------------------------------------------------------------------------------------------------------------
# BERT Model
#-----------------------------------------------------------------------------------------------------------------------

""" Format elapsed times """
def format_time(elapsed):
	# Take a time in seconds and returns a string hh:mm:ss
	elapsed_rounded = int(round(elapsed))

	return str(datetime.timedelta(seconds=elapsed_rounded))


""" Cuda preliminaries """
def device_selector ():
	# Check GPU availability
	if torch.cuda.is_available():
		
		device = torch.cuda.set_device(0) # Select GPU for use
		device = torch.device('cuda')
		print("GPU(s) available: ", torch.cuda.device_count())
		print("We will use GPU index: ", torch.cuda.current_device())
		print("We will use GPU Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
	else:
		print("No GPU(s) available, using CPU")
		device = torch.device('cpu')

	return device


""" Fine-tune and evaluate the model """
def train_evaluate_model(model, device, optim, scheduler, epochs , train_loader, evaluation_loader):
	
	# Set the seed value all over the place to make this reproducible.
	seed_val = 28
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# Measure the total training time for the whole run
	total_t0 = time.time()

	model.to(device)

	# Save training and validation loss, validation accuracy, and timings
	train_status = []
	eval_status = []

	# x-axis in *def plot_training_validation_status*
	list_epochs = []

	# Use these for precision_score, recall_score, f1_score, and confusion_matrix
	predictions, true_labels = [], []
	
	for epoch_i in range(0, epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

		if train_loader is not None:
			
			print("Training")
			# Measure how long each training epoch takes
			t0 = time.time()

			# Reset the total loss and accuracy for this epoch
			total_training_loss = 0
			total_training_accuracy = 0

			model.train()

			for batch in train_loader:
				input_ids = batch[0].to(device)
				attention_mask = batch[1].to(device)
				labels = batch[2].to(device)
				
				optim.zero_grad()
				model.zero_grad()

				outputs = model(input_ids=input_ids, token_type_ids=None ,attention_mask=attention_mask, labels=labels)
				loss = outputs.loss
				logits = outputs.logits
				total_training_loss += loss.item()
				logits = logits.detach().cpu().numpy()
				label_ids = labels.to('cpu').numpy()
				total_training_accuracy += flat_accuracy(logits, label_ids)

				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optim.step()
				scheduler.step()
		
			avg_training_loss = total_training_loss / len(train_loader)
			avg_training_accuracy = total_training_accuracy / len(train_loader)
			print("Average training loss: {0:.2f}".format(avg_training_loss))
			print("Average training accuracy: {0:.2f}".format(avg_training_accuracy))
		
			training_time = format_time(time.time()-t0)
			print("Training epcoh took: {:}".format(training_time))

			list_epochs.append(epoch_i + 1)
			train_status.append(
				{
					'Epoch': epoch_i + 1,
					'Training Loss': round(avg_training_loss, 5),
					'Training Accuracy': round(avg_training_accuracy, 5),
					'Training Time': training_time,
				}
			)

		"""
		Evaluation
		After the completion of each training epoch, 
		measure our performance on our evaluation set
		"""
		if evaluation_loader is not None:

			print("")
			print("Running Evaluation")
			t0 = time.time()

			model.eval()

			# Tracking variables
			total_evaluation_accuracy = 0
			total_evaluation_loss = 0

			for batch in evaluation_loader:
				input_ids = batch[0].to(device)
				attention_mask = batch[1].to(device)
				labels = batch[2].to(device)
				
				with torch.no_grad():
					outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
					loss = outputs.loss
					logits = outputs.logits # The "logits" are the output values prior to applying an activation function like the softmax.
					
					logits = logits.detach().cpu().numpy()
					label_ids = labels.to('cpu').numpy()

					predictions.append(logits)
					true_labels.append(label_ids)

					total_evaluation_loss += loss.item()
					total_evaluation_accuracy += flat_accuracy(logits, label_ids)

			avg_evaluation_loss = total_evaluation_loss / len(evaluation_loader)
			avg_evaluation_accuracy = total_evaluation_accuracy / len(evaluation_loader)
			print("Average evaluation loss: {0:.2f}".format(avg_evaluation_loss))
			print("Average evaluation accuracy: {0:.2f}".format(avg_evaluation_accuracy))

			evaluation_time = format_time(time.time() - t0)
			print("evaluation took: {:}".format(evaluation_time))

			# Record all statistics from this epoch
			if not list_epochs:
					list_epochs.append(epoch_i + 1)
			eval_status.append(
				{
					'Epoch': epoch_i + 1,
					'Evaluation Loss': round(avg_evaluation_loss, 5),
					'Evaluation Accuracy': round(avg_evaluation_accuracy, 5),
					'Evaluation Time': evaluation_time
				}
			)

	print("")
	print("Training complete!")
	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
	print("")
		
	return model, list_epochs, train_status, eval_status, predictions, true_labels

#-----------------------------------------------------------------------------------------------------------------------
# Compute Results and Graphics
#-----------------------------------------------------------------------------------------------------------------------

""" Simple calculation of accuracy from predicted vs true labels """
def flat_accuracy(predictions, true_labels):
	# Combine logit arrays into one after argmax'ing to get dominant label
	predictions_flat = np.argmax(predictions, axis=1).flatten()
	true_labels_flat = true_labels.flatten()

	return np.sum(predictions_flat == true_labels_flat) / len(true_labels_flat)


""" Create DataFrame and Graphics of loss and accuracy values """
def plot_training_evaluation_status(list_epochs, train_status, eval_status):
	train_eval_status = []

	if train_status and eval_status:
		print("Combine train_status & eval_status")
		for train_dic, eval_dic in zip(train_status, eval_status):
			train_dic.update(eval_dic)
			train_eval_status.append(train_dic)

	# Create a DataFrame (contains info.: loss, accuracy, and timings)
	if not eval_status:
		df_stats = pd.DataFrame(data=train_status)
	elif not train_status:
		df_stats = pd.DataFrame(data=eval_status)
	else:
		df_stats = pd.DataFrame(data=train_eval_status)
		new_order = ['Epoch', 'Training Loss', 'Evaluation Loss', 'Training Accuracy', 'Evaluation Accuracy', 'Training Time', 'Evaluation Time']
		df_stats = df_stats.reindex(columns=new_order)
	
	pd.set_option('precision', 2)
	df_stats = df_stats.set_index('Epoch')
	print(df_stats)
	print("")
	
	# Save the DataFrame
	time = str(datetime.datetime.now())[:18]
	if not eval_status:
		filename = "train_status" + " " + time
	elif not train_status:
		filename = "eval_status" + " " + time
	else:
		filename = "train_eval_status" + " " + time
	filename = re.sub("[^\w]+", "_", filename.lower())
	df_stats.to_csv(filename + ".csv")
	
	# Use plot styling from seaborn
	sns.set(style='darkgrid')

	# Increase the plot size and font size
	sns.set(font_scale=1.5)
	plt.rcParams["figure.figsize"] = (12,6)
	
	# Plot the learning curve
	if not eval_status:
		plt.plot(df_stats['Training Loss'], marker='o', linestyle='-', color='#EE7733', label="Training Loss")
		plt.plot(df_stats['Training Accuracy'], marker='o', linestyle='-', color='#007799', label="Training Accuracy")
	
	if not train_status:
		plt.plot(df_stats['Evaluation Loss'], marker='o', linestyle='-', color='#117733', label="Evaluation Loss")
		plt.plot(df_stats['Evaluation Accuracy'], marker='o', linestyle='-', color='#AA4499', label="Evaluation Accuracy")
	
	if train_eval_status:
		plt.plot(df_stats['Training Loss'], marker='o', linestyle='-', color='#EE7733', label="Training Loss")
		plt.plot(df_stats['Evaluation Loss'], marker='o', linestyle='-', color='#117733', label="Evaluation Loss")
		plt.plot(df_stats['Training Accuracy'], marker='o', linestyle='-', color='#007799', label="Training Accuracy")
		plt.plot(df_stats['Evaluation Accuracy'], marker='o', linestyle='-', color='#AA4499', label="Evaluation Accuracy")

	# Label the plot
	plt.xlabel("Epoch")
	plt.ylabel("Loss & Accuracy")
	plt.legend()
	plt.xticks(list_epochs)
	
	# Save the plot
	plt.savefig(filename + ".png")


""" Calculate Precision, Recall , and F1 Scores """
def compute_evaluation_scores(predictions, true_labels):
	# Combine the results across all batches and pick dominant label
	predictions = np.concatenate(predictions, axis=0)
	predictions = np.argmax(predictions, axis=1).flatten()

	# Also combine true labels for each batch into single list
	true_labels = np.concatenate(true_labels, axis=0)

	precision = precision_score(true_labels, predictions, average = 'weighted', labels=np.unique(predictions))
	recall = recall_score(true_labels, predictions, average = 'weighted')
	f1 = f1_score(true_labels, predictions, average = 'macro')
	print("Precision: ", precision)
	print("Recall: ", recall)
	print("F1 Score: ", f1)
	print("")

	return precision, recall, f1

def plot_metrics (predictions, true_labels, n_classes):
	# Combine the results across all batches and pick dominant label
	predictions = np.concatenate(predictions, axis=0)
	predictions = np.argmax(predictions, axis=1).flatten()

	# Also combine true labels for each batch into single list
	true_labels = np.concatenate(true_labels, axis=0)

	
	classification_repo_no_dict = classification_report(true_labels, predictions, target_names=n_classes)
	print(classification_repo_no_dict)
	print("")

	classification_repo = classification_report(true_labels, predictions, target_names=n_classes, output_dict=True)
	df_classification_repo = pd.DataFrame(classification_repo).transpose()

	confus_matri = confusion_matrix(true_labels, predictions, labels=np.unique(true_labels), normalize=None)
	print("Confusion Matrix")
	print(confus_matri)
	print("")
	display_confus_matri = ConfusionMatrixDisplay(confusion_matrix=confus_matri, display_labels=np.unique(true_labels))

	# Save the classification report and the confusion matrix
	time = str(datetime.datetime.now())[:18]
	filename_stamp = re.sub("[^\w]+", "_", time.lower())
	df_classification_repo.to_csv("classification_repo_" + filename_stamp + ".csv")
	((display_confus_matri.plot()).figure_).savefig("confusion_matrix_" + filename_stamp + ".png")

	return classification_repo
