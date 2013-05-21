from __future__ import division
import os, sys
import re
import nltk
from nltk import bigrams, trigrams
import math
import xlrd
import linecache
import itertools

punctuation = re.compile(r'[-.?!,<''``"*:>;()|0-9]')

stopwords = nltk.corpus.stopwords.words('english')

redundant_words = ['degree','required','a','an','the','will',
                   'must','should','not','needs','as','cv',
                   'resume','your','recruiter','expected',
                   'evening','job','you','located','client','you',
                   'also','need','opportunities','opportunity','apply','to','we',
                   'are','currently','looking','superb','our','for','specialised',
                   'leading','firm','currently','working','within','small','team',
                   'family','seeking','role']


CORPUS_DIR=""

categories = ['Sal1', 'Sal2', 'Sal3', 'Sal4', 'Sal5', 'Sal6','Sal7','Sal8','Sal9','Sal10','Sal11','Sal12','Sal13','Sal14','Sal15']


def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))


def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))


class Instance(object):

    def __init__(self, title,Description,Location_normalized,Contract_type,Contract_time,Company,Category_sal,label):
        
        self.title = title
        self.Description = Description
        self.Location_normalized = Location_normalized
        self.Contract_type = Contract_type
        self.Contract_time = Contract_time
        self.Company = Company
        self.Category_sal = Category_sal
        self.label = label
        

    def __repr__(self):
        return "%s | %s | %s" % (self.title, self.label)


def extract_important_words_with_tags(desc_text):
    tokens = nltk.word_tokenize(desc_text)
    tokens = [token.lower() for token in tokens if len(token) > 2]
    tokens = [word for word in tokens if word not in stopwords]
 
    tokens = [punctuation.sub("", word) for word in tokens]
    tokens = [w for w in tokens if w != ' ']
    tokens = [w for w in tokens if w != '']
    
    tagged_sentences = nltk.pos_tag(tokens)
    filtered_tags = [(w,t) for (w,t) in tagged_sentences if t <> 'JJ']
    filtered_tags = [(w,t) for (w,t) in filtered_tags if not t.startswith('VB')]
    
    filtered_tags = [(w,t) for (w,t) in filtered_tags if t <> 'MD']
    filtered_words = [w for (w,t) in filtered_tags]
    filtered_words = [w for w in filtered_words if w not in redundant_words]
    return filtered_words
##    return tokens




def read_corpus(label,fname):
    
    labeled_data_loc = os.path.join(CORPUS_DIR,fname)
    print "Successful location"


    try:
        data = []
        print "Processing data...."
        workbook = xlrd.open_workbook(labeled_data_loc)
        worksheet = workbook.sheet_by_index(0)
        num_rows = worksheet.nrows - 1
        num_cells = worksheet.ncols - 1
        curr_row = -1
        while curr_row < num_rows:
            label_1 = "Sal"
            curr_row += 1
            row = worksheet.row(curr_row)       
            curr_cell = -1
            while curr_cell < num_cells:
                
                curr_cell += 1
                    
                cell_value = worksheet.cell_value(curr_row, curr_cell)
     
                if(curr_cell == 0):
                    tokens = nltk.word_tokenize(cell_value)
                    tokens = [token.lower() for token in tokens if len(token) > 2]
                    tokens = [word for word in tokens if word not in stopwords]
                    tokens = [punctuation.sub("", word) for word in tokens]
                    tokens = [w for w in tokens if w != ' ']
                    tokens = [w for w in tokens if w != '']
                        
                            
                    temp_text = ''
                    for f in tokens:
                        temp_text = temp_text + ' ' + f

                        
                    title = temp_text
                        

                elif(curr_cell == 1):
                    Description = cell_value
                    # Building the vocabulary
                        
##                    tokens_desc = nltk.word_tokenize(cell_value)
##                    tokens_desc = [punctuation.sub("", word) for word in tokens_desc]
##                    tokens_desc = [word for word in tokens_desc if word not in stopwords]
##                    tokens_desc = [w for w in tokens_desc if w != ' ']
##                    tokens_desc = [w for w in tokens_desc if w != '']
##                    tokens_desc = [token.lower() for token in tokens_desc]
##                        
##                    final_tokens = []
##                    bi_tokens = bigrams(tokens_desc)
##                    final_tokens.extend(bi_tokens)
##                    docs[curr_row] = {'freq': {}, 'tf': {}, 'idf': {},
##                        'tf-idf': {}, 'tokens': []}
##
##                    for token in final_tokens:
##                        docs[curr_row]['freq'][token] = freq(token, final_tokens)
##                        docs[curr_row]['tf'][token] = tf(token, final_tokens)
##                        docs[curr_row]['tokens'] = final_tokens
##
##
##                    vocabulary.append(final_tokens)

                elif(curr_cell == 2):
                    Location_normalized = cell_value

                elif(curr_cell == 3):
                    Contract_type = cell_value

                elif(curr_cell == 4):
                    Contract_time = cell_value

                elif(curr_cell == 5):
                    Company = cell_value

                elif(curr_cell == 6):
                    Category_sal = cell_value

                elif(curr_cell == 7):
                   
                    label_1 = cell_value
                    
                        
                    
                    
            instance = Instance(title, Description,Location_normalized,Contract_type,Contract_time,Company,Category_sal,label_1)  
            data.append(instance)

    except IOError, e:
        sys.stderr.write("read_corpus(): %s\n" % e)

    return data
        
    

def extract_features(instance,row_num):
    if(row_num % 100 == 0):
        print "Processing"+str(row_num)
    feature_set={}
    feature_set[str(0)] = instance.title
    feature_set[str(1)] = instance.Location_normalized
    feature_set[str(2)] = instance.Contract_type
    feature_set[str(3)] = instance.Contract_time
    feature_set[str(4)] = instance.Company
    feature_set[str(5)] = instance.Category_sal

##    Description_text = instance.Description
##    Tagged_Description_text = extract_important_words_with_tags(Description_text)
##
##    for k in range(0,len(Tagged_Description_text)):
##        feature_set[str(k+6)] = Tagged_Description_text[k]

##    init_len = len(Tagged_Description_text)
##    words = {}
##    for token in docs[row_num]['tf']:
##        #The Inverse-Document-Frequency
##        docs[row_num]['idf'][token] = idf(token, vocabulary)
##        #The tf-idf
##        docs[row_num]['tf-idf'][token] = tf_idf(token, docs[row_num]['tokens'], vocabulary)
##
##    for token in docs[row_num]['tf-idf']:
##        words[token] = docs[row_num]['tf-idf'][token]
##    count1 = 0
##    expected_count = 0.1 * len(words)
##    for item in sorted(words.items(), key=lambda x: x[1], reverse=True):
##        if(count1 < expected_count):
##            feature_set[str(6+count1+init_len)] = item[0]
##            count1 = count1 + 1
##                        

    return feature_set


def make_training_data(data):
    training_data_1=[]
    row_num = 0
    for instance in data:
        
        feature_set = extract_features(instance,row_num)
        row_num = row_num + 1
        label = instance.label
        training_data_1.append((feature_set,label))
        
    return training_data_1


def make_classifier(training_data):
    return nltk.classify.naivebayes.NaiveBayesClassifier.train(training_data)
   


def split_data(data, train_frac=.1):


    #random.shuffle(data)


    train_size = int(len(data) * train_frac)
    test_set = data[:train_size]
    training_set = data[train_size:]

    return training_set, test_set

if __name__ == '__main__':

    


    
    import argparse
    import random
    parser = argparse.ArgumentParser(description='Classify salary categories.')
    parser.add_argument('category', action='store', type=str,
                        help="One of the salary Categories: %s" % categories)


    category = 'Sal2'
    if not category in categories:
        sys.stderr.write("Illegal category: %s\n" % category)
        parser.print_help()
        sys.exit()

    vocabulary = []
    docs = {}
    train_fname = "Sample_data_set.xls"
    Complete_data = read_corpus(category,train_fname)

    Complete_make_data = make_training_data(Complete_data)

    [training_data_1,test_data_1] = split_data(Complete_make_data, train_frac=.1)    
    [training_data_2,test_data_2] = split_data(training_data_1, train_frac=.1)    
    [training_data_3,test_data_3] = split_data(training_data_2, train_frac=.1)
    [training_data_4,test_data_4] = split_data(training_data_3, train_frac=.1)
    [training_data_5,test_data_5] = split_data(training_data_4, train_frac=.1)
    [training_data_6,test_data_6] = split_data(training_data_5, train_frac=.1)
    [training_data_7,test_data_7] = split_data(training_data_6, train_frac=.1)
    [training_data_8,test_data_8] = split_data(training_data_7, train_frac=.1)
    [training_data_9,test_data_9] = split_data(training_data_8, train_frac=.1)

    training_data_2 = training_data_2 + test_data_1
    
    training_data_3 = training_data_3 + test_data_1
    training_data_3 = training_data_3 + test_data_2

    training_data_4 = training_data_4 + test_data_1
    training_data_4 = training_data_4 + test_data_2
    training_data_4 = training_data_4 + test_data_3

    training_data_5 = training_data_5 + test_data_1
    training_data_5 = training_data_5 + test_data_2
    training_data_5 = training_data_5 + test_data_3
    training_data_5 = training_data_5 + test_data_4


    training_data_6 = training_data_6 + test_data_1
    training_data_6 = training_data_6 + test_data_2
    training_data_6 = training_data_6 + test_data_3
    training_data_6 = training_data_6 + test_data_4
    training_data_6 = training_data_6 + test_data_5


    training_data_7 = training_data_7 + test_data_1
    training_data_7 = training_data_7 + test_data_2
    training_data_7 = training_data_7 + test_data_3
    training_data_7 = training_data_7 + test_data_4
    training_data_7 = training_data_7 + test_data_5
    training_data_7 = training_data_7 + test_data_6


    training_data_8 = training_data_8 + test_data_1
    training_data_8 = training_data_8 + test_data_2
    training_data_8 = training_data_8 + test_data_3
    training_data_8 = training_data_8 + test_data_4
    training_data_8 = training_data_8 + test_data_5
    training_data_8 = training_data_8 + test_data_6
    training_data_8 = training_data_8 + test_data_7

    training_data_9 = training_data_9 + test_data_1
    training_data_9 = training_data_9 + test_data_2
    training_data_9 = training_data_9 + test_data_3
    training_data_9 = training_data_9 + test_data_4
    training_data_9 = training_data_9 + test_data_5
    training_data_9 = training_data_9 + test_data_6
    training_data_9 = training_data_9 + test_data_7
    training_data_9 = training_data_9 + test_data_8

    


    for p in range(1,9):
        print "Iteration: " + str(p)

        training_set_name = "training_data_"+str(p)
        test_set_name = "test_data_"+str(p)

        training_data = eval(training_set_name)
        test_data = eval(test_set_name)
        
        print "*************************************"
        print "Successfully split the training and test sets"
        print "The training set contains: " + str(len(training_data))
        print "The test set contains: " + str(len(test_data))

        
        print "Successfully converted the training data to required format"
        classifier = make_classifier(training_data)
       
        print "Successfully converted the test data to required format"
        
        print " "
        print "Training_Accuracy: "+ str(nltk.classify.accuracy(classifier, training_data))

        print "Validation Accuracy: "+ str(nltk.classify.accuracy(classifier, test_data))

        print "*************************************"

   

##        for e in test_d:
##           dist_1 = classifier.prob_classify(e[0])
##           if dist_1.prob(category) < 1.0:
##               print "Classification: " + category + ",Prob: "+ str(dist_1.prob(category))+ "Original label: "+ e[1]
       








