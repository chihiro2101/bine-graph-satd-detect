#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
from sklearn import preprocessing
from data_utils import DataUtils
from graph_utils import GraphUtils
import random
import math
import os
import pandas as pd
from sklearn import metrics
import json
import re
import string
from nltk.corpus import stopwords
from nltk import re, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
import os
import subprocess
from sklearn.model_selection import KFold



def get_dataset(df):
    # df = pd.read_csv("data/satd-comments-manual-subclass-simple-preprocessed.csv")
    before = len(df)
    print(len(df))
    df = df.dropna()
    print(len(df))
    after = len(df)
    assert before == after

    # import pdb;pdb.set_trace()

    df = df.rename(columns={"classification": "label"})
    df['document_index'] = df.index

    number_of_folds = 5
    kf = KFold(n_splits = number_of_folds, shuffle = True, random_state = 2)
    fold = kf.split(df)
    result = next(fold, None)
    train = df.iloc[result[0]]
    test =  df.iloc[result[1]]
    # import pdb;pdb.set_trace()
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    train['subset'] = 'train'
    test['subset'] = 'test'
    frames = [train, test]
    dataset = pd.concat(frames)
    return dataset, train, test

def clean_text(text):
    import nltk
    nltk.download('stopwords',quiet=True)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)

    re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                        .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                        re.MULTILINE | re.UNICODE)
    re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

    text = re_url.sub("URL", text)

    text = re_ip.sub("IPADDRESS", text)

    text = text.lower().split()

    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)
    return text

def calculate_tf_idf(df):
    df = df[['text','label','document_index']]
    df['text']= df['text'].apply(str)

    #uncommand this line when run satdin4sources dataset
    # df['text']=df['text'].map(lambda x: clean_text(x))

    # df['document_index'] = df.index

    #create bag of words
    all_text = ' '.join(df['text'].astype(str))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([all_text])

    # Get feature names and their frequencies
    features = vectorizer.get_feature_names_out()
    term_frequencies = X.toarray()[0]

    # Filter out words with term frequency less than 4
    filtered_words = [word for word, freq in zip(features, term_frequencies) if freq >= 4]

    # Update the DataFrame to keep only the words with frequency >= 4
    df['text'] = df['text'].apply(lambda text: ' '.join([word for word in str(text).split() if word in filtered_words]))
    df = df.dropna(subset=['label'])
    # df = df[df['text'].apply(lambda x: len(str(x).split(" ")) > 4)]

    # Calculate TF-IDF for the remaining words for full dataset
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(df['text'])
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())

    # import pdb;pdb.set_trace()


    word_dict = {k: v for v, k in enumerate(filtered_words)}

    #create data for train and test
    document_word_df = pd.DataFrame(columns=['document_index','word_index','tfidf'])
    for index, row in df.iterrows():
        i = 0
        list_of_word = row["text"].split()
        for word in list_of_word:
            try:
                # new_edge = {'document_index': 'u' + str(int(row["document_index"])), 'word_index': 'i' + str(int(word_dict[word])), 'tfidf': df_tfidf.iloc[row["document_index"]][word]}
                new_edge = {'document_index': 'u' + str(int(row["document_index"])), 'word_index': 'i' + str(int(word_dict[word])), 'tfidf': df_tfidf.iloc[i][word]}
            except:
                import pdb;pdb.set_trace()
                print("debug")
            document_word_df = document_word_df._append(new_edge, ignore_index=True)
        i += 1
    
    return df, document_word_df

def data_preprocessing(path, rawfilename): #build graph for document-word graph
    file_name = os.path.join(path, rawfilename)
    df = pd.read_csv(file_name)
    df, train_df, test_df  = get_dataset(df)

    # df['document_index'] = df.index
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # import pdb;pdb.set_trace()

    #Save doc-word graph
    save_folder_path = "data/" + file_name.split("/")[2].split(".csv")[0]

    #unprocessed train dataset
    unprocessed_train_file_name = os.path.join(save_folder_path, "unprocessed_train_set.csv") 
    train_df.to_csv(unprocessed_train_file_name, sep=',', index=False)

    #unprocessed test dataset
    unprocessed_test_file_name = os.path.join(save_folder_path, "unprocessed_test_set.csv") 
    test_df.to_csv(unprocessed_test_file_name, sep=',', index=False)

    train_df = pd.read_csv(unprocessed_train_file_name)
    test_df = pd.read_csv(unprocessed_test_file_name)


    # df = df[['text','classification']]
    # df['text']= df['text'].apply(str)
    # df['text']=df['text'].map(lambda x: clean_text(x))
    # df['document_index'] = df.index

    # #create bag of words
    # all_text = ' '.join(df['text'].astype(str))
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform([all_text])

    # # Get feature names and their frequencies
    # features = vectorizer.get_feature_names_out()
    # term_frequencies = X.toarray()[0]

    # # Filter out words with term frequency less than 4
    # filtered_words = [word for word, freq in zip(features, term_frequencies) if freq >= 4]

    # # Update the DataFrame to keep only the words with frequency >= 4
    # df['text'] = df['text'].apply(lambda text: ' '.join([word for word in str(text).split() if word in filtered_words]))
    # df = df.dropna(subset=['classification'])
    # # df = df[df['text'].apply(lambda x: len(str(x).split(" ")) > 4)]

    # # Calculate TF-IDF for the remaining words for full dataset
    # vectorizer_tfidf = TfidfVectorizer()
    # X_tfidf = vectorizer_tfidf.fit_transform(df['text'])
    # df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())

    # word_dict = {k: v for v, k in enumerate(filtered_words)}

    # #create data for train and test
    # document_word_df = pd.DataFrame(columns=['document_index','word_index','tfidf'])
    # for index, row in df.iterrows():
    #     list_of_word = row["text"].split()
    #     for word in list_of_word:
    #         new_edge = {'document_index': 'u' + str(int(row["document_index"])), 'word_index': 'i' + str(int(word_dict[word])), 'tfidf': df_tfidf.iloc[row["document_index"]][word]}
    #         document_word_df = document_word_df._append(new_edge, ignore_index=True)
    # import pdb;pdb.set_trace()


    df, document_word_df = calculate_tf_idf(df)
    train_df, train_document_word_df = calculate_tf_idf(train_df)
    test_df, test_document_word_df = calculate_tf_idf(test_df)

    # import pdb;pdb.set_trace()


    
    # Check whether the specified path exists or not
    isExist = os.path.exists(save_folder_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_folder_path)
        print("The new directory is created!")

    #full dataset
    graph_file_name = os.path.join(save_folder_path, "doc_word_graph.csv") 
    document_index_file_name = os.path.join(save_folder_path, "document_index.csv") 
    document_word_df.to_csv(graph_file_name, sep='\t', index=False, header=False)
    df.to_csv(document_index_file_name, sep=',', index=False)


    # #train set
    # train_graph_file_name = os.path.join(save_folder_path, "train_doc_word_graph.csv") 
    # train_document_index_file_name = os.path.join(save_folder_path, "train_document_index.csv") 
    # train_document_word_df.to_csv(train_graph_file_name, sep='\t', index=False, header=False)
    # train_df.to_csv(train_document_index_file_name, sep=',', index=False)

    # #test_set
    # test_graph_file_name = os.path.join(save_folder_path, "test_doc_word_graph.csv") 
    # test_document_index_file_name = os.path.join(save_folder_path, "test_document_index.csv") 
    # test_document_word_df.to_csv(test_graph_file_name, sep='\t', index=False, header=False)
    # test_df.to_csv(test_document_index_file_name, sep=',', index=False)    


    # df_word = pd.DataFrame(filtered_words, columns=['word'])
    # df_word['word_index'] = df_word.index
    # df_word.to_csv(word_index_file_name, sep=',', index=False)

    # freq = X.sum(axis=0).A1
    # # Create a DataFrame with term frequencies
    # term_freq_df = pd.DataFrame({'Term': features, 'Frequency': freq})
    # term_freq_df = term_freq_df[term_freq_df['Frequency'] > 3]
    # print(term_freq_df.info())
    # df = df_total.rename(columns={'text': 'comment', 'classification': 'debt'})
    # new_df = df.copy()
    # def condition(label):
    #     if label  == 'code_debt' or label == 'design_debt':
    #         return 'code/design_debt'
    #     else:
    #         return label
    # new_df = new_df[new_df.debt != 'defect_debt']
    # new_df = new_df[new_df.debt != 'architecture_debt']
    # new_df = new_df[new_df.debt != 'build_debt']
    # new_df['debt'] = new_df['debt'].map(lambda x: condition(x))
    # output_file_name = os.path.join(output_folder_path, file_name)
    # df.to_csv(output_file_name, sep=',', index=False)
    # import pdb;pdb.set_trace()
    print(file_name," DONE")



def walk_generator(gul,args):
    """
    walk generator
    :param gul:
    :param args:
    :return:
    """
    gul.calculate_centrality(args.mode)
    if args.large == 0:
        gul.homogeneous_graph_random_walks(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 1:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 2:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(datafile=args.train_data,percentage=args.p,maxT=args.maxT, minT=args.minT)
    return gul


def train_by_sampling(args):
    data_preprocessing(args.path, args.rawfilename)
    model_path = os.path.join('../', args.model_name)
    # import pdb;pdb.set_trace()
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter, args.d))
    print('========== processing data ===========')
    # dul = DataUtils(model_path)
    # if args.rec:
    #     test_user, test_item, test_rate = dul.read_data(args.test_data)
    print("constructing graph....")
    gul = GraphUtils(model_path) #init object from graph_utils
    gul.construct_training_graph(args.train_data)
    # edge_dict_u = gul.edge_dict_u # dict {'u5999': {'i10': 1.0, 'i653': 1.0}, 'u6000': {'i10': 1.0}}
    # edge_list = gul.edge_list # list edge [('u5999', 'i653', 1.0), ('u6000', 'i10', 1.0)]
    walk_generator(gul,args)


def main():
    parser = ArgumentParser("BiNE",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--train-data', default=r'data/satd-dataset-r-comment-without-nondebt',
                        help='Input graph file.')

    # parser.add_argument('--test-data', default=r'../data/rating_test.dat')

    parser.add_argument('--model-name', default='satd-dataset-r-comment-without-nondebt',
                        help='name of model.')

    # parser.add_argument('--vectors-u', default=r'../data/vectors_u.dat',
    #                     help="file of embedding vectors of U")

    # parser.add_argument('--vectors-v', default=r'../data/vectors_v.dat',
    #                     help="file of embedding vectors of V")

    # parser.add_argument('--case-train', default=r'../data/wiki/case_train.dat',
    #                     help="file of training data for LR")

    # parser.add_argument('--case-test', default=r'../data/wiki/case_test.dat',
    #                     help="file of testing data for LR")

    parser.add_argument('--ws', default=5, type=int,
                        help='window size.')

    parser.add_argument('--ns', default=4, type=int,
                        help='number of negative samples.')

    parser.add_argument('--d', default=128, type=int,
                        help='embedding size.')

    parser.add_argument('--maxT', default=32, type=int,
                        help='maximal walks per vertex.')

    parser.add_argument('--minT', default=1, type=int,
                        help='minimal walks per vertex.')

    parser.add_argument('--p', default=0.15, type=float,
                        help='walk stopping probability.')

    parser.add_argument('--alpha', default=0.01, type=float,
                        help='trade-off parameter alpha.')

    parser.add_argument('--beta', default=0.01, type=float,
                        help='trade-off parameter beta.')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='trade-off parameter gamma.')

    parser.add_argument('--lam', default=0.025, type=float,
                        help='learning rate lambda.')
    parser.add_argument('--max-iter', default=2, type=int,
                        help='maximal number of iterations.')

    parser.add_argument('--top-n', default=10, type=int,
                        help='recommend top-n items for each user.')

    parser.add_argument('--rec', default=1, type=int,
                        help='calculate the recommendation metrics.')

    parser.add_argument('--lip', default=0, type=int,
                        help='calculate the link prediction metrics.')

    parser.add_argument('--large', default=0, type=int,
                        help='for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph')

    parser.add_argument('--mode', default='hits', type=str,
                        help='metrics of centrality')
    # parser.add_argument('--word-dict', default=r'../data/satd-dataset-/word_index.csv',
    #                     help='Input word dictionary file.')
    parser.add_argument('--document-dict', default=r'../data/satd-dataset-r-comment-without-nondebt/document_index.csv',
                        help='Input document file.')
    parser.add_argument('--path', default=r'data/satd-java-raw',
                        help='Input path to raw file.')
    parser.add_argument('--rawfilename', default=r'satd-dataset-r-comment-without-nondebt.csv',
                        help='Input raw dataset filename.')                        


    args = parser.parse_args()
    train_by_sampling(args)

if __name__ == "__main__":
    sys.exit(main())

# python train.py 
# --train-data ../data/dblp/rating_train.dat 
# --test-data ../data/dblp/rating_test.dat 
# --lam 0.025  #learning rate
# --max-iter 100 
# --model-name dblp 
# --rec 1 #calculate the recommendation metrics
# --large 2  #for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph
# --vectors-u ../data/dblp/vectors_u.dat 
# --vectors-v ../data/dblp/vectors_v.dat
# --word-dict ../data/satd-dataset-r-comment/word_index.json
# --document-dict ../data/satd-dataset-r-comment/document_index.csv


# python train.py --train-data ../data/dblp/rating_train.dat --test-data ../data/dblp/rating_test.dat --lam 0.025 --max-iter 1 --model-name dblp --rec 1 --large 2 --vectors-u ../data/dblp/vectors_u.dat --vectors-v ../data/dblp/vectors_v.dat
# data/satd-dataset-r-comment

# python train.py --train-data ../data/satd-dataset-r-comment/train.csv --test-data ../data/satd-dataset-r-comment/test.csv --lam 0.025 --max-iter 2 --model-name r-comment --rec 1 --large 2 --vectors-u ../data/satd-dataset-r-comment/vectors_u.dat --vectors-v ../data/satd-dataset-r-comment/vectors_v.dat