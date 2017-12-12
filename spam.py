import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix 


#in root folder ['spam.py','train-mails','test-mails']

train_dir='./train-mails'
test_dir='./test-mails'

#Collecting features out of mails , based on that we are going to train our model.
#We are calling 3000 features in each mail based on that our system will define a mail to be a spam or non-spam
def make_Dictionary(mail_dir):

	# collecting the name of all emails in './train' directory
	dir_ls=os.listdir(mail_dir)

	# maping the whole list of names with there relative path to spam.py. i.e. 'spam.text' to './train/spam.text'
	email_ls = [os.path.join(mail_dir,fi) for fi in dir_ls]

	# we are going to store all the words in first body of mail
	all_words=[]

	for i in email_ls:
			# we are going to open all mails.
			fo=open(i,'r')
			# iterate through them to find the words 
			for m,line in enumerate(fo):
				# as the body start after 0,1 lines.
				if m==2:
					words=line.split()
					# store the words in body in 'all_words'.
					all_words+=words
			# closing the current opened file.
			fo.close()
	# counting the number of repeation of a particular word in the array 'all_words'.
	dictionary=Counter(all_words)

	# To remove some non-words
	# use list() instead keys(), python 3.0
	remove_nonwords=list(dictionary)
	for i in remove_nonwords:
		# if the word contain any character other then alphabets. return false
		if i.isalpha()==False:
			del dictionary[i]
		#if length of word is less than 3 remove it. i.e. remove '.','is' etc.
		elif len(i)<3:
			del dictionary[i]
	# As we decide our datasets will have only 3000 features so we need to extract top 3000  repeated words.
	dictionary=dictionary.most_common(3000)

	return dictionary

# Now we actually need to create a datasets of samples and their corresponding fetaures.
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix
		

# Calling the make_Dictionary to decide the features.
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels
train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training Naive bayes classifier

model1 = MultinomialNB()
model1.fit(train_matrix,train_labels)

# Test the unseen mails for Spam
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = model1.predict(test_matrix)
print (confusion_matrix(test_labels,result1))

