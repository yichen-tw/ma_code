from data_preprocessing_bert import *


#---------------------------------------------------------------------------------------------------------------------------
# Main routine
#--------------------------------------------------------------------------------------------------------------------------- 

def main():

	#-----------------------------------------------------------------------------------------------------------------------
	# BERT Model Setup
	#-----------------------------------------------------------------------------------------------------------------------
	flavour = 'bert-base-uncased'
	batch_size = 16
	learning_rate = 2e-5
	epochs = 2

	### Load Pretrained BERT Tokenizer
	tokenizer = BertTokenizer.from_pretrained(flavour, do_lower_case=True)

	#-----------------------------------------------------------------------------------------------------------------------
	### GPU Availibility
	device = device_selector()

	#-----------------------------------------------------------------------------------------------------------------------
	# Control Variables
	#-----------------------------------------------------------------------------------------------------------------------
	### Modify Emotion Class(es) & Class Distribution 
	### Original Emotion Classes ['Anger', 'Anxiety', 'Desire', 'Disgust', 'Fear', 'Happiness', 'Relaxation', 'Sadness']
	emotion_deletes = ['Anxiety'] # e.g.: "['Desire', 'Disgust',... ]"
	emotion_substitutes = {'Anger':'Happiness', 'Desire':'Happiness', 'Disgust':'Happiness'} # {'old class':'new class'} e.g.: "{'Happiness':'Relaxation','Disgust':'Anger',...}"
	
	### Reorder emotion classes so that related emotions group into a spectrum
	do_reorder_emotions = False

	### Balance class distribution of datasets
	### "RO" = Random Oversampling
	### "RU" = Random Undersampling
	resample_rwwd = []
	resample_training_set = []

	### Choose Split Mode
	### "TTS_v1" = a random Train/Test Split (rwwd_long and rwwd_short are splitted seperately)
	### "TTS_v2" = a random Train/Test Split (combine rwwd_long and rwwd_short first, then split)
	### "KF"  = K-Folds cross-validator
	### "SKF" = Stratified K-Folds cross-validator
	### "SS"  = ShuffleSplit (Random permutation cross-validator)
	### "SSS" = Stratified ShuffleSplit cross-validator
	split_mod = "TTS_v1"
	n_splits = 2
	
	#-----------------------------------------------------------------------------------------------------------------------
	### Load RWWD (COVID-19 Real World Worry Datatset) & Encode Emotion Labels
	rwwd_data_path = "covid19worry/files/meta/rwwd_full.csv"
	encoded_rwwd_emotion, rwwd_long, rwwd_short, n_classes = load_rwwd_data(rwwd_data_path, emotion_deletes, emotion_substitutes, do_reorder_emotions)
	print("RWWD Current State")
	display_count_labels(encoded_rwwd_emotion + encoded_rwwd_emotion, n_classes)

	#-----------------------------------------------------------------------------------------------------------------------
	### Balance Calss Distribution of Datasets 
	if resample_rwwd != []:
		rwwd_long, encoded_rwwd_emotion_long = resampling(rwwd_long, encoded_rwwd_emotion, resample_rwwd)
		rwwd_short, encoded_rwwd_emotion_short = resampling(rwwd_short, encoded_rwwd_emotion, resample_rwwd)
		print("RWWD After Resampling")
		display_count_labels(encoded_rwwd_emotion_long + encoded_rwwd_emotion_short, n_classes)

		if split_mod == "TTS_v1":
			texts_lists = [rwwd_long, rwwd_short]
			labels_lists = [encoded_rwwd_emotion_long, encoded_rwwd_emotion_short]
		else:
			X_texts = rwwd_long + rwwd_short
			y_labels = encoded_rwwd_emotion_long + encoded_rwwd_emotion_short

	else:
		if split_mod == "TTS_v1":
			texts_lists = [rwwd_long, rwwd_short]
			labels_lists = [encoded_rwwd_emotion, encoded_rwwd_emotion]
		else:
			X_texts = rwwd_long + rwwd_short
			y_labels = encoded_rwwd_emotion + encoded_rwwd_emotion
		
	#-----------------------------------------------------------------------------------------------------------------------
	# Use Random Train/Test Split & Continue Main Routine
	#-----------------------------------------------------------------------------------------------------------------------
	if split_mod == "TTS_v1" or split_mod == "TTS_v2":
		if split_mod == "TTS_v1":
			X_train, X_val, y_train, y_val = get_train_test_split(texts_lists, labels_lists, split_mod, n_splits)

			print("Training Dataset")
			display_count_labels(y_train, n_classes)
			print("Evaluation / Validation Dataset")
			display_count_labels(y_val, n_classes)
		else:
			X_train, X_val, y_train, y_val = get_train_test_split(X_texts, y_labels, split_mod, n_splits)

			print("Training Dataset")
			display_count_labels(y_train, n_classes)
			print("Evaluation / Validation Dataset")
			display_count_labels(y_val, n_classes)

#-----------------------------------------------------------------------------------------------------------------------
### Only Resample Training Dataset
		if resample_training_set != []:
			X_train, y_train = resampling(X_train, y_train, resample_training_set)
			print("Training Dataset After Resampling")
			display_count_labels(y_train, n_classes)

	#-----------------------------------------------------------------------------------------------------------------------
	### Tokenization & BERT Input Formatting
		train_dataset = transform_dataset(tokenizer, X_train, y_train)
		val_dataset = transform_dataset(tokenizer, X_val, y_val)
		
		train_loader, validation_loader = get_dataloaders(batch_size, train_dataset, val_dataset)

	#-----------------------------------------------------------------------------------------------------------------------
	### Load Pretrained BERT
		model  = BertForSequenceClassification.from_pretrained(flavour, num_labels=len(n_classes), output_attentions = False, output_hidden_states = False)
		model.cuda()

	#-----------------------------------------------------------------------------------------------------------------------
	### Optimizer & Learning Rate Scheduler
		optim = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)  
		total_steps = len(train_loader) * epochs
		scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, num_training_steps = total_steps)
	
	#-----------------------------------------------------------------------------------------------------------------------
	### Train & Evaluate Model
		model, list_epochs, train_status, val_status, predictions, true_labels = train_evaluate_model(model, device, optim, scheduler, epochs, train_loader, validation_loader)
		# model, list_epochs, train_status, val_status, predictions, true_labels = train_evaluate_model(model, device, optim, scheduler, epochs, train_loader, None)
		# model, list_epochs, train_status, val_status, predictions, true_labels = train_evaluate_model(model, device, optim, scheduler, epochs, None, validation_loader)
	
	#-----------------------------------------------------------------------------------------------------------------------
	### Visualize Results from Model Training and Evaluation
		plot_training_evaluation_status(list_epochs, train_status, val_status)
		compute_evaluation_scores(predictions, true_labels)
		plot_metrics(predictions, true_labels, n_classes)

	#-----------------------------------------------------------------------------------------------------------------------
	# Cross Validation
	#-----------------------------------------------------------------------------------------------------------------------
	else:
		tt_split = get_train_test_split(X_texts, y_labels, split_mod, n_splits)
		for e, (train_index, val_index) in enumerate(tt_split):
			if n_splits > 1:
				print("Fold ", e + 1)
			X_train, X_val = np.array(X_texts)[train_index], np.array(X_texts)[val_index]
			y_train, y_val = np.array(y_labels)[train_index], np.array(y_labels)[val_index]

			X_train = X_train.tolist()
			X_val = X_val.tolist()
			y_train = y_train.tolist()
			y_val = y_val.tolist()

			print("Training Dataset")
			display_count_labels(y_train, n_classes)
			print("Evaluation / Validation Dataset")
			display_count_labels(y_val, n_classes)

			# train_dataset = transform_dataset(tokenizer, X_train, y_train)
			# val_dataset = transform_dataset(tokenizer, X_val, y_val)

			train_dataset = encode_tokens(tokenizer, X_train, y_train)
			val_dataset = encode_tokens(tokenizer, X_val, y_val)

			train_loader, validation_loader = get_dataloaders(batch_size, train_dataset, val_dataset)

	#-----------------------------------------------------------------------------------------------------------------------
	### Load Pretrained BERT
			model  = BertForSequenceClassification.from_pretrained(flavour, num_labels=len(n_classes), output_attentions = False, output_hidden_states = False)
			model.cuda()

	#-----------------------------------------------------------------------------------------------------------------------
	### Optimizer & Learning Rate Scheduler
			optim = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
			total_steps = len(train_loader) * epochs
			scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, num_training_steps = total_steps)

	#-----------------------------------------------------------------------------------------------------------------------
	### Train & Evaluate Model
			model, list_epochs, train_status, val_status, predictions, true_labels = train_evaluate_model(model, device, optim, scheduler, epochs, train_loader, validation_loader)

	#-----------------------------------------------------------------------------------------------------------------------
	### Visualize Results from Model Training and Evaluation
			compute_evaluation_scores(predictions, true_labels)
			plot_metrics(predictions, true_labels, n_classes)


if __name__ == "__main__":
	main()
