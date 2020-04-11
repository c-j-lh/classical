#!/usr/bin/env python
# coding: utf-8

# In[127]:


##################
# Classification #
##################
composers = ("Bach", "Mozart", "Beethoven", "Debussy")


# In[128]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import svm, linear_model, naive_bayes, neural_network, neighbors, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import random, math
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from collections import Counter
from itertools import combinations
from tqdm.notebook import *


# In[129]:


with open('mxl-list.txt', 'r') as f:
    dataset = [piece.strip() for piece in f.readlines()]
    
    composer_datas = []
    for composer in composers:
        composer_data = [f for f in dataset if f.replace('-', '_').split('_')[0] == composer]
        composer_datas.append(composer_data)
        
    '''bach_data = [f for f in dataset if f.replace('-', '_').split('_')[0] == 'bach']
    beethoven_data = [f for f in dataset if f.replace('-', '_').split('_')[0] == 'beethoven']
    debussy_data = [f for f in dataset if f.replace('-', '_').split('_')[0] == 'debussy']
    scarlatti_data = [f for f in dataset if f.replace('-', '_').split('_')[0] == 'scarlatti']
    victoria_data = [f for f in dataset if f.replace('-', '_').split('_')[0] == 'victoria']'''


# In[130]:


COMPOSERs = []
for composer, composer_data in zip(composers, composer_datas):
    with open(f'{composer}-chordsequence.txt', 'r') as f:
        COMPOSER = [' '.join(piece.strip('[]\n').split(', ')) for piece in f.readlines()]
        COMPOSER = [(COMPOSER[i], composer_data[i]) for i in range(len(COMPOSER))]
        COMPOSERs.append(COMPOSER)
    
'''with open('bach-chordsequence.txt', 'r') as f:
    BACH = [' '.join(piece.strip('[]\n').split(', ')) for piece in f.readlines()]
    BACH = [(BACH[i], bach_data[i]) for i in range(len(BACH))]
with open('beethoven-chordsequence.txt', 'r') as f:
    BEETHOVEN = [' '.join(piece.strip('[]\n').split(', ')) for piece in f.readlines()]
    BEETHOVEN = [(BEETHOVEN[i], beethoven_data[i]) for i in range(len(BEETHOVEN))]
with open('debussy-chordsequence.txt', 'r') as f:
    DEBUSSY = [' '.join(piece.strip('[]\n').split(', ')) for piece in f.readlines()]
    DEBUSSY = [(DEBUSSY[i], debussy_data[i]) for i in range(len(DEBUSSY))]
with open('scarlatti-chordsequence.txt', 'r') as f:
    SCARLATTI = [' '.join(piece.strip('[]\n').split(', ')) for piece in f.readlines()]
    SCARLATTI = [(SCARLATTI[i], scarlatti_data[i]) for i in range(len(SCARLATTI))]
with open('victoria-chordsequence.txt', 'r') as f:
    VICTORIA = [' '.join(piece.strip('[]\n').split(', ')) for piece in f.readlines()]
    VICTORIA = [(VICTORIA[i], victoria_data[i]) for i in range(len(VICTORIA))]'''


# In[131]:


def find_ngrams(input_list, N=4):
    return [' '.join(input_list[i:i+N]) for i in range(len(input_list)-N+1)]

def ngrams_by_composer(composer): 
    for i in range(1,5):
        ngrams = []
        for piece in composer:
            ngrams += find_ngrams(piece[0].split(' '), i)
        print(len(ngrams), '{}-grams total;'.format(str(i)), len(set(ngrams)), 'unique')
    print('-')

def show_ngrams(composer_data, composer_name):
    print(composer_name, ':', len(composer_data), 'pieces')
    ngrams_by_composer(composer_data)


# In[132]:


for COMPOSER, composer in zip(COMPOSERs, composers):
    show_ngrams(COMPOSER, composer)
show_ngrams(sum(COMPOSERs, []), 'all composers')

'''show_ngrams(BACH,'bach')
show_ngrams(BEETHOVEN,'beethoven')
show_ngrams(DEBUSSY,'debussy')
show_ngrams(SCARLATTI,'scarlatti')
show_ngrams(VICTORIA, 'victoria')
show_ngrams(BACH+BEETHOVEN+DEBUSSY+SCARLATTI+VICTORIA, 'all composers')'''


# In[133]:


def build_Xy(composers, size=1):
    if size >= 1: # use every rows
        indices = [range(len(composer)) for composer in composers]
    else:
        indices = [random.sample(range(len(composer)), math.ceil(size*len(composer))) for composer in composers]

    y = []
    for i in range(len(composers)):
        y += [i for n in range(len(indices[i]))]
    
    X = []
    for i in range(len(composers)):
        X += [composers[i][j] for j in indices[i]]
    
    return X, np.array(y, dtype='int16')


# In[213]:


def crossvalidate(X_tuple, y, classifiers, vectorizer, NGRAMRANGE, K=10, set_=False):    
    for clf in classifiers:
        clf.cm_sum = np.zeros([len(set(y)),len(set(y))], dtype='int16')                      if set_ else np.zeros([len(composers), len(composers)], dtype='int16')
        clf.accuracies, clf.fones, clf.misclassified, clf.runningtime = [], [], [], []
        clf.fones_micro, clf.fones_macro = [], []
        clf.name = str(clf).split('(')[0]

    X = np.array([piece[0] for piece in X_tuple])
    
    import pickle #
    ids = [int(piece[1][:-4].split('_')[1]) for piece in X_tuple] #
    with open('indices.pickle', 'rb') as file:
        df = pickle.load(file)
    indices = np.asarray([df.loc[id_] for id_ in ids])
    
    filenames = np.array([piece[1] for piece in X_tuple])
    kf = KFold(n_splits=min(K,len(y)), shuffle=True)
    for train_index, test_index in tqdm(kf.split(y), unit='fold', total=10, leave=False):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        vct = vectorizer.set_params(lowercase=False, token_pattern=u"(?u)\\b\\w+\\b", ngram_range=NGRAMRANGE)
        X_train_tfidf = vct.fit_transform(X_train)
#         X_test_tfidf = tfidf.transform(X_test)  # i think this computes tf-idf values using the whole test set, but i want each piece to be treated separately
        X_test_tfidf = sp.vstack([vct.transform(np.array([piece])) for piece in X_test])
    
        #print(type(X_train_tfidf), X_train_tfidf.shape)
        X_train_tfidf = np.array([list(ai) + list(bi) for ai, bi in zip(X_train_tfidf.toarray(), indices)])
        X_test_tfidf = np.array([list(ai) + list(bi) for ai, bi in zip(X_test_tfidf.toarray(), indices)])
        #print(X_train_tfidf)
        for clf in tqdm(classifiers, unit='classifier', leave=False):
            t = datetime.now()
            clf.fit(X_train_tfidf, y_train)
            y_pred = clf.predict(X_test_tfidf)
            clf.runningtime.append((datetime.now()-t).total_seconds())
            clf.cm_sum += confusion_matrix(y_test, y_pred) #, labels=list(set(y_test)) if set_ else range(1, len(composers)+1)
            clf.misclassified.append(test_index[np.where(y_test != y_pred)]) # http://stackoverflow.com/a/25570632
            clf.accuracies.append(accuracy_score(y_test, y_pred))
            clf.fones.append(f1_score(y_test, y_pred, average='weighted'))
            clf.fones_micro.append(f1_score(y_test, y_pred, average='micro'))
            clf.fones_macro.append(f1_score(y_test, y_pred, average='macro'))

    result = dict()
    for clf in classifiers:
        clf.misclassified = np.sort(np.hstack(clf.misclassified))
        result[clf.name] = [clf.cm_sum, clf.accuracies, clf.fones, clf.misclassified, filenames[clf.misclassified], clf.runningtime, clf.fones_micro, clf.fones_macro]
    return result


# In[205]:


def benchmark_classifiers(composers, NGRAMRANGES, classifiers, vectorizer, n=1, retrieve_title=True, set_=False):
    misclassified_list = []
    for NGRAMRANGE in tqdm(NGRAMRANGES, unit='NGRAMRANGE', leave=False):
        print('n-gram range', NGRAMRANGE)
        X, y = build_Xy(composers, size=n)
        cv_result = crossvalidate(X, y, classifiers, vectorizer, NGRAMRANGE, set_=set_)
        for clf, results in cv_result.items():
            print(clf)
            cm = results[0]
            print(cm)
            acc = results[1] # using two different f-measures, don't need this
#             print('accuracy', round(np.mean(acc)*100,2), '({})'.format(round(np.std(acc, ddof=1)*100,2)))
            fones = results[2] # weighted average, don't need this
#             print('f1', round(np.mean(fones)*100,2), '({})'.format(round(np.std(fones, ddof=1)*100,2)), fones)
            misclassified = results[3]
            misclassified_filenames = results[4]
            misclassified_list += list(misclassified_filenames)
#             print('misclassified',[(misclassified[i], misclassified_filenames[i]) for i in range(len(misclassified))])
            runningtime = results[5]
#             print('running time', np.sum(runningtime))
            fones_micro = results[6]
            fones_macro = results[7]
            print('micro-averaged f-score (std) & macro-averaged f-score (std)')
            print(round(np.mean(fones_micro),4), '({})'.format(round(np.std(fones_micro, ddof=1),4)), '&', round(np.mean(fones_macro),4), '({})'.format(round(np.std(fones_macro, ddof=1),4)))
    print('-----')
    return misclassified_list


# In[228]:


'''COMPOSERS = [BACH, BEETHOVEN, DEBUSSY, SCARLATTI, VICTORIA]'''
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
NGRAMRANGES = [(1,2), (1,3)] #[(1,1),(2,2),(3,3),(4,4),(1,2),(3,4),(1,4)]
CLASSIFIERS = [
    svm.LinearSVC(),#penalty='l2', C=5, loss='hinge'),
    linear_model.LogisticRegression(), #penalty='l2', C=100, tol=1, multi_class='multinomial', solver='sag'),
    neighbors.KNeighborsClassifier(),#weights='distance'),
    #naive_bayes.MultinomialNB(alpha=0.00001, fit_prior=False),
    neural_network.MLPClassifier(),#solver='lbfgs', hidden_layer_sizes=(100,)),
    RandomForestClassifier(),#n_estimators=100, random_state=42),
    GradientBoostingClassifier(),#n_estimators=100, max_leaf_nodes=4, max_depth=None, random_state=2,
                   #min_samples_split=5),
]


# In[ ]:


VECTORIZER = TfidfVectorizer(sublinear_tf=True)
benchmark_classifiers(COMPOSERs,NGRAMRANGES,CLASSIFIERS,VECTORIZER)


# In[ ]:


# Compare different methods of vectorizing


# In[ ]:


VECTORIZER = CountVectorizer(binary=True)
benchmark_classifiers(COMPOSERs,NGRAMRANGES,CLASSIFIERS,VECTORIZER)


# In[ ]:


VECTORIZER = CountVectorizer()
benchmark_classifiers(COMPOSERs,NGRAMRANGES,CLASSIFIERS,VECTORIZER)


# In[ ]:


# test pairwise classification 


# In[214]:


if True:
    NGRAMRANGES = [(1,2)]
    VECTORIZER = TfidfVectorizer(sublinear_tf=True)
    for indices in tqdm(list(combinations(range(len(composers)),2))):
        print(f'comparing {composers[indices[0]]} and {composers[indices[1]]} (composers {indices[0]} and {indices[1]})',) # COMPOSERs = [BACH, BEETHOVEN, DEBUSSY, SCARLATTI, VICTORIA]
        benchmark_classifiers([COMPOSERs[i] for i in indices],NGRAMRANGES,CLASSIFIERS,VECTORIZER, set_=True)


# In[ ]:


# Identify the often-misclassified files
# do the experiment 100 times with the best classifier, SVM
# then find pieces that are misclassified more than 50% of the time
'''COMPOSERS = [BACH, BEETHOVEN, DEBUSSY, SCARLATTI, VICTORIA]'''
CLASSIFIERS = [svm.LinearSVC(penalty='l2', C=5, loss='hinge')] # 
appendix = []
for i in trange(100):
    appendix += benchmark_classifiers(COMPOSERs,NGRAMRANGES,CLASSIFIERS,VECTORIZER)
Counter(appendix).most_common()


# In[ ]:


# Use both chord sequences and duration data


# In[ ]:


flatten = lambda l: [item for sublist in l for item in sublist]
with open('Bach-durations.txt', 'r') as f:
    BA = [line.strip() for line in f.readlines()]
with open('Mozart-durations.txt', 'r') as f:
    MO = [line.strip() for line in f.readlines()]
with open('Beethoven-durations.txt', 'r') as f:
    BE = [line.strip() for line in f.readlines()]
with open('Debussy-durations.txt', 'r') as f:
    DE = [line.strip() for line in f.readlines()]
'''with open('victoria-durations.txt', 'r') as f:
    VD = [line.strip() for line in f.readlines()]'''
    
#BD2_TYPELENGTH = [piece.split(';') for piece in BD2]
BA_TYPELENGTH = [piece.split(';') for piece in BA]
MO_TYPELENGTH = [piece.split(';') for piece in MO]
BE_TYPELENGTH = [piece.split(';') for piece in BE]
DE_TYPELENGTH = [piece.split(';') for piece in DE]

typelengths = list(set(flatten(BA_TYPELENGTH + MO_TYPELENGTH + BE_TYPELENGTH + DE_TYPELENGTH)))
typelength_dict = {typelengths[i]: str(i+300) for i in range(len(typelengths))}
BA_T = [(' '.join([typelength_dict[dur] for dur in piece]),'temp') for piece in BA_TYPELENGTH]
MO_T = [(' '.join([typelength_dict[dur] for dur in piece]),'temp') for piece in MO_TYPELENGTH]
BE_T = [(' '.join([typelength_dict[dur] for dur in piece]),'temp') for piece in BE_TYPELENGTH]
DE_T = [(' '.join([typelength_dict[dur] for dur in piece]),'temp') for piece in DE_TYPELENGTH]
#VD_T = [(' '.join([typelength_dict[dur] for dur in piece]),'temp') for piece in VD_TYPELENGTH]


# In[ ]:


# print most common duration by composer, regardless of element type(chord/note/rest)

# BD_LENGTHONLY = [[string.split('|')[1] for string in piece.split(';')] for piece in BD]
# SD_LENGTHONLY = [[string.split('|')[1] for string in piece.split(';')] for piece in SD]
# BD2_LENGTHONLY = [[string.split('|')[1] for string in piece.split(';')] for piece in BD2]
# DD_LENGTHONLY = [[string.split('|')[1] for string in piece.split(';')] for piece in DD]
# VD_LENGTHONLY = [[string.split('|')[1] for string in piece.split(';')] for piece in VD]
# lengths = list(set(flatten(BD2_LENGTHONLY+BD_LENGTHONLY+DD_LENGTHONLY+SD_LENGTHONLY+VD_LENGTHONLY)))
# length_dict = {lengths[i]: str(i+200) for i in range(len(lengths))}
# BD_L = [(' '.join([length_dict[dur] for dur in piece]),'temp') for piece in BD_LENGTHONLY]
# BD2_L = [(' '.join([length_dict[dur] for dur in piece]),'temp') for piece in BD2_LENGTHONLY]
# SD_L = [(' '.join([length_dict[dur] for dur in piece]),'temp') for piece in SD_LENGTHONLY]
# DD_L = [(' '.join([length_dict[dur] for dur in piece]),'temp') for piece in DD_LENGTHONLY]
# VD_L = [(' '.join([length_dict[dur] for dur in piece]),'temp') for piece in VD_LENGTHONLY]
# duration_all       = flatten(BD2_LENGTHONLY+BD_LENGTHONLY+DD_LENGTHONLY+SD_LENGTHONLY+VD_LENGTHONLY)
# duration_bach      = flatten(BD2_LENGTHONLY)
# duration_beethoven = flatten(BD_LENGTHONLY)
# duration_debussy   = flatten(DD_LENGTHONLY)
# duration_scarlatti = flatten(SD_LENGTHONLY)
# duration_victoria  = flatten(VD_LENGTHONLY)

# for l in [duration_all,duration_bach,duration_beethoven,duration_debussy,duration_scarlatti,duration_victoria]:
#     for key, value in Counter(l).most_common(10):
#         print(key, '&', round(100*value/len(l),2))
#     print('')


# In[221]:


def crossvalidate_twofeaturesets(X_tuple1, X_tuple2, y, classifiers, vectorizer, range1, range2, K=10):    
    for clf in classifiers:
        clf.cm_sum = np.zeros([len(set(y)),len(set(y))], dtype='int16')
        clf.accuracies, clf.fones, clf.misclassified, clf.runningtime = [], [], [], []
        clf.fones_micro, clf.fones_macro = [], []
        clf.name = str(clf).split('(')[0]

    X1 = np.array([piece[0] for piece in X_tuple1])
    X2 = np.array([piece[0] for piece in X_tuple2])
    filenames = np.array([piece[1] for piece in X_tuple2])
    kf = KFold(n_splits=K, shuffle=True)
    for train_index, test_index in tqdm(kf.split(y), total=10, unit='fold', leave=False):
        y_train, y_test = y[train_index], y[test_index]
        X_train_new, X_test_new = X1[train_index], X1[test_index]
        vct1 = vectorizer.set_params(ngram_range=range1)
        X_train, X_test = X2[train_index], X2[test_index] 
        vct2 = vectorizer.set_params(ngram_range=range2)
   
        X_train_new_tfidf = vct1.fit_transform(X_train_new) # use two separate vectorizers for each feature set
        X_test_new_tfidf = sp.vstack([vct1.transform(np.array([piece])) for piece in X_test_new])
        X_train_tfidf = vct2.fit_transform(X_train)
        X_test_tfidf = sp.vstack([vct2.transform(np.array([piece])) for piece in X_test])
        
        X_train_tfidf = sp.hstack((X_train_tfidf, X_train_new_tfidf)) # Merge the two feature sets
        X_test_tfidf = sp.hstack((X_test_tfidf, X_test_new_tfidf))
        
        for clf in tqdm(classifiers, unit='classifier', leave=False):
            t = datetime.now()
            clf.fit(X_train_tfidf, y_train)
            y_pred = clf.predict(X_test_tfidf)
            clf.runningtime.append((datetime.now()-t).total_seconds())
            clf.cm_sum += confusion_matrix(y_test, y_pred)#, labels=range(1, len(composers)+1))
            clf.misclassified.append(test_index[np.where(y_test != y_pred)]) # http://stackoverflow.com/a/25570632
            clf.accuracies.append(accuracy_score(y_test, y_pred))
            clf.fones.append(f1_score(y_test, y_pred, average='weighted'))
            clf.fones_micro.append(f1_score(y_test, y_pred, average='micro'))
            clf.fones_macro.append(f1_score(y_test, y_pred, average='macro'))

    result = dict()
    for clf in classifiers:
        clf.misclassified = np.sort(np.hstack(clf.misclassified))
        result[clf.name] = [clf.cm_sum, clf.accuracies, clf.fones, clf.misclassified, filenames[clf.misclassified], clf.runningtime, clf.fones_micro, clf.fones_macro]
    return result


# In[222]:


def benchmark_classifiers_twofeaturesets(composers1, composers2, range1, range2, classifiers, vectorizer, n=1, retrieve_title=True):
    misclassified_list = []
    print('duration n-gram range:', range1, 'chord n-gram range:', range2)
    X1, y = build_Xy(composers1, size=n)
    X2, y = build_Xy(composers2, size=n)
    cv_result = crossvalidate_twofeaturesets(X1, X2, y, classifiers, vectorizer, range1, range2)
    for clf, results in cv_result.items():
        print(clf)
        cm = results[0]
        print(cm)
        acc = results[1]
        fones = results[2]
        misclassified = results[3]
        misclassified_filenames = results[4]
        misclassified_list += list(misclassified_filenames)
#             print('misclassified',[(misclassified[i], misclassified_filenames[i]) for i in range(len(misclassified))])
        runningtime = results[5]
#         print('running time', np.sum(runningtime))
        fones_micro = results[6]
        fones_macro = results[7]
        print('F-measures')
        print(round(np.mean(fones_micro),4), '({})'.format(round(np.std(fones_micro, ddof=1),4)), '&', round(np.mean(fones_macro),4), '({})'.format(round(np.std(fones_macro, ddof=1),4)))
    print('-----')
    return misclassified_list


# In[225]:


CLASSIFIERS = [
    svm.LinearSVC(penalty='l2', C=5, loss='hinge'),
    linear_model.LogisticRegression(penalty='l2', C=100, tol=1, multi_class='multinomial', solver='sag'),
    neighbors.KNeighborsClassifier(weights='distance'),
    naive_bayes.MultinomialNB(alpha=0.00001, fit_prior=False),
    neural_network.MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,)),
    RandomForestClassifier(n_estimators=500, random_state=42),
    GradientBoostingClassifier(n_estimators=500, max_leaf_nodes=4, max_depth=None, random_state=2,
                   min_samples_split=5),
              ]
VECTORIZER = TfidfVectorizer(sublinear_tf=True, lowercase=False, token_pattern=u"(?u)\\b\\w+\\b")

COMPOSERS1 = [BA_T, MO_T, BE_T, DE_T]


# In[226]:


benchmark_classifiers_twofeaturesets(COMPOSERS1, COMPOSERs, (1,2), (1,2), CLASSIFIERS, VECTORIZER)


# In[216]:


for indices in combinations(range(len(composers)),2):
    print('composer indices', [i for i in indices]) 
    benchmark_classifiers_twofeaturesets([COMPOSERS1[i] for i in indices],[COMPOSERs[i] for i in indices],(1,1),(1,2),CLASSIFIERS,VECTORIZER)


# In[217]:


# Identify the often-misclassified files, using both feature sets
# do the experiment 100 times with the best classifier, SVM
# then find pieces that are misclassified more than 50% of the time
COMPOSERS1 = [BA_T, MO_T, BE_T, DE_T]
#COMPOSERS2 = [BACH, BEETHOVEN, DEBUSSY, SCARLATTI, VICTORIA]
CLASSIFIERS = [svm.LinearSVC(penalty='l2', C=5, loss='hinge')] # 
appendix = []
for i in trange(100):
    appendix += benchmark_classifiers_twofeaturesets(COMPOSERS1, COMPOSERs, (1,1), (1,2), CLASSIFIERS, VECTORIZER)
Counter(appendix).most_common()


# In[ ]:


X, y = build_Xy(COMPOSERs, size=1)
print(X)
y

