#Import all required libraries
#Based off of Text Classification with Spacy by Rithesh Sreenivasan
#https://github.com/rsreetech/TextClassificationWithSpacy/blob/master/TweetTextClassificationWithSpacy.ipynb
import spacy
import random
import time
import numpy as np
import pandas as pd
import re
import string

from spacy.util import minibatch, compounding
import sys
from spacy import displacy
from itertools import chain

from sklearn.metrics import classification_report

def load_data_spacy(file_path):
  
    train_data = pd.read_csv(file_path)
    train_data.dropna(axis = 0, how ='any',inplace=True) 
    train_data['Num_words_text'] = train_data['data'].apply(lambda x:len(str(x).split())) 
    mask = train_data['Num_words_text'] >2
    train_data = train_data[mask]
    print(train_data['label'].value_counts())
    
   
    train_texts = train_data['data'].tolist()
    train_cats = train_data['label'].tolist()
    final_train_cats=[]
    for cat in train_cats:
        cat_list = {}
        if cat == 'First Party Collection/Use':
            cat_list['First Party Collection/Use'] =  1
            cat_list['Data Security'] =  0
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  0
        elif cat == 'Data Security':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] =  1
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  0
        elif cat == 'Third Party Sharing/Collection':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] =  0
            cat_list['Third Party Sharing/Collection'] =  1
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  0
        elif cat == 'User Choice/Control':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] =  0
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  1
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  0
        elif cat == 'User Access, Edit and Deletion':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] =  0
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  1
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  0
        elif cat == 'International and Specific Audiences':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] =  0
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  1
            cat_list['Data Retention'] =  0
        elif cat == 'Data Retention':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] = 0
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  1
        elif cat == 'Do Not Track':
            cat_list['First Party Collection/Use'] =  0
            cat_list['Data Security'] = 0
            cat_list['Third Party Sharing/Collection'] =  0
            cat_list['User Choice/Control'] =  0
            cat_list['User Access, Edit and Deletion'] =  0
            cat_list['International and Specific Audiences'] =  0
            cat_list['Data Retention'] =  0
            cat_list['Do Not Track'] =  1
        final_train_cats.append(cat_list)
    
    training_data = list(zip(train_texts, [{"cats": cats} for cats in final_train_cats]))
    return training_data,train_texts,train_cats


training_data,train_texts,train_cats = load_data_spacy("C:\\Users\\lisa\\.spyder-py3\\trainIOT.csv")
print(training_data[:10])
print(len(training_data))
test_data,test_texts,test_cats   = load_data_spacy("C:\\Users\\lisa\\.spyder-py3\\testIOT.csv")
print(len(test_data))



def Sort(sub_li): 
  
    # reverse = True (Soresulting_list = list(first_list)rts in Descending  order) 
    # key is set to sort using second element of  
    # sublist lambda has been used 
    return(sorted(sub_li, key = lambda x: x[1],reverse=True))  

def evaluate(tokenizer, textcat, test_texts, test_cats ):
    docs = (tokenizer(text) for text in test_texts)
    preds = []
    for i, doc in enumerate(textcat.pipe(docs)):
        #print(doc.cats.items())
        scores = Sort(doc.cats.items())
        #print(scores)
        catList=[]
        for score in scores:
            catList.append(score[0])
        preds.append(catList[0])
        
    labels = ['First Party Collection/Use', 
              'Third Party Sharing/Collection',
              'User Choice/Control','Data Security',
              'International and Specific Audiences','User Access, Edit and Deletion','Policy Change','Data Retention','Do Not Track']
    
    print(classification_report(test_cats,preds,labels=labels))
    docs = (tokenizer(text) for text in test_texts)
    preds = []
    for i, doc in enumerate(textcat.pipe(docs)):
        #print(doc.cats.items())
        scores = Sort(doc.cats.items())
        #print(scores)
        catList=[]
        for score in scores:
            catList.append(score[0])
        preds.append(catList[0])
        
    labels = ['positive', 'negative','neutral']
    
    print(classification_report(test_cats,preds,labels=labels))
    
    

def train_spacy(train_data, iterations,test_texts,test_cats, model_arch, dropout = 0.3, model=None,init_tok2vec=None):
    ''' Train a spacy NER model, which can be queried against with test data
   
    train_data : training data in the format of (sentence, {cats: ['positive'|'negative'|'neutral']})
    labels : a list of unique annotations
    iterations : number of training iterations
    dropout : dropout proportion for training
    display_freq : number of epochs between logging losses to console
    '''
    
    nlp = spacy.load("en_core_web_md")
    

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": model_arch}
        )
        nlp.add_pipe(textcat, last=True)
        
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("First Party Collection/Use")
    textcat.add_label("Third Party Sharing/Collection")
    textcat.add_label("User Choice/Control")
    textcat.add_label("Data Security")
    textcat.add_label("International and Specific Audiences")
    textcat.add_label("User Access, Edit and Deletion")
    textcat.add_label("Policy Change")
    textcat.add_label("Data Retention")
    textcat.add_label("Do Not Track")


    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(16.0, 64.0, 1.5)
        for i in range(iterations):
            print('Iteration: '+str(i))
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the test data 
                evaluate(nlp.tokenizer, textcat, test_texts,test_cats)
        with nlp.use_params(optimizer.averages):
            modelName = model_arch+"PrivacyPolicies"
            filepath = "C:\\Users\\lisa\\.spyder-py3\\"+modelName+"\\"
            nlp.to_disk(filepath)
    return nlp

nlp = train_spacy(training_data, 10,test_texts,test_cats,"bow")
#nlp = train_spacy(training_data, 5,test_texts,test_cats,"simple_cnn")
#nlp = train_spacy(training_data, 10,test_texts,test_cats,"ensemble")


#nlp2 = spacy.load("C:\\Users\\lisa\\.spyder-py3\\bowPrivacyPolicies")
#doc2 = nlp2(test_texts[100])
#print("Text: "+ test_texts[100])
#print("Orig :"+ test_cats[100])
#print(" Predicted :") 
#print(doc2.cats)
#print("=======================================")
#doc2 = nlp2(test_texts[1000])
#print("Text: "+ test_texts[1000])
#print(" Orig :"+test_cats[1000])
#print(" Predicted Cats:") 
#print(doc2.cats)