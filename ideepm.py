import sys
import os
import numpy as np
import random
import pdb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
import keras.layers.core as core
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization, Lambda, GlobalMaxPooling2D
from keras.layers import LSTM, Bidirectional, Reshape, Layer
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import merge, Input, TimeDistributed
from keras.regularizers import WeightRegularizer, l1, l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import class_weight
from keras import objectives, initializations
from keras import backend as K
#import utils

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label  


def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:] #discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict


def read_fasta_file_new(fasta_file = '../data/UTR_hg19.fasta'):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:].split()[0]  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    return seq_dict

def focal_loss(gamma=2, alpha=2):
	def focal_loss_fixed(y_true, y_pred):
		if(K.backend()=="tensorflow"):
			import tensorflow as tf
			pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
		if(K.backend()=="theano"):
			import theano.tensor as T
			pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
	return focal_loss_fixed

def perform_metrics(out, y_test):
    threshold = np.arange(0.1, 0.9, 0.1)

    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:, i])
        for j in threshold:
            y_pred = [1 if prob >= j else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(y_test[:, i], y_pred))
        acc = np.array(acc)
        index = np.where(acc == acc.max())
        accuracies.append(acc.max())
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    print "best thresholds", best_threshold
    y_pred = np.array(
        [[1 if out[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

    print("-" * 40)
    print("Matthews Correlation Coefficient")
    print("Class wise accuracies")
    print(accuracies)

    print("other statistics\n")
    total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i] == y_pred[i]).sum() == 5])
    print("Fully correct output")
    print(total_correctly_predicted)
    print(total_correctly_predicted / out.shape[0])


def load_rnacomend_data(datadir='../data/'):
    pair_file = datadir + 'interactions_HT.txt'
    # rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'

    rna_seq_dict = read_fasta_file(rna_seq_file)
    protein_set = set()
    inter_pair = {}
    new_pair = {}
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            protein_set.add(protein)
            rna = values[1]
            inter_pair.setdefault(rna, []).append(protein)
            new_pair.setdefault(protein, []).append(rna)

    for protein, rna in new_pair.iteritems():
        print protein, len(rna)
    return inter_pair, rna_seq_dict, protein_set


def get_rnarecommend(inter_pair_dict, rna_seq_dict, protein_list):
    data = {}
    labels = []
    rna_seqs = []
    protein_list.append("negative")
    all_hg19_utrs = read_fasta_file_new()
    remained_rnas = list(set(all_hg19_utrs.keys()) - set(inter_pair_dict.keys()))
    #pdb.set_trace()
    for rna, protein in inter_pair_dict.iteritems():
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        init_labels = np.array([0]*len(protein_list))
        inds = []
        for pro in protein:
            inds.append(protein_list.index(pro))
        init_labels[np.array(inds)] = 1
        labels.append(init_labels)
        rna_seqs.append(rna_seq)
    #pdb.set_trace()
    max_num_targets = np.sum(labels, axis =0).max()
    # negatives

    random.shuffle(remained_rnas)
    #pdb.set_trace()
    for rna in remained_rnas[:max_num_targets]:
        rna_seq = all_hg19_utrs[rna]
        rna_seq = rna_seq.replace('T', 'U')
        rna_seqs.append(rna_seq)
        init_labels = np.array([0] * (len(protein_list) - 1) + [1])
        labels.append(init_labels)

    data["seq"] = rna_seqs
    data["Y"] = np.array(labels)

    return data


def get_RNA_seq_concolutional_array(seq, motif_len = 10):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    # for seq in seqs:
    # for key, seq in seqs.iteritems():
    half_len = motif_len/2
    row = (len(seq) + 2 * half_len)
    new_array = np.zeros((row, 4))
    for i in range(half_len):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - half_len, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + half_len
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
            # data[key] = new_array
    return new_array

def padding_sequence(seq, max_len = 2695, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_class_weight(df_y):
    y_classes = df_y.idxmax(1, skipna=False)

    from sklearn.preprocessing import LabelEncoder

    # Instantiate the label encoder
    le = LabelEncoder()

    # Fit the label encoder to our label series
    le.fit(list(y_classes))

    # Create integer based labels Series
    y_integers = le.transform(list(y_classes))

    # Create dict of labels : integer representation
    labels_and_integers = dict(zip(y_classes, y_integers))

    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    sample_weights = compute_sample_weight('balanced', y_integers)

    class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))

    return class_weights_dict


def get_all_rna_mildata(seqs, labels, training_val_indice, train_val_label, test_indice, test_label, max_len = 2695):
    index = 0
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])
    train_bags, train_labels = get_bag_data_1_channel(train_seqs, train_val_label, max_len= max_len)

    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])

    test_bags, test_labels = get_bag_data_1_channel(test_seqs, test_label, max_len=max_len)

    return train_bags, train_labels, test_bags, test_labels


def set_cnn_model(input_dim=4, input_length=2705, nbfilter = 101):
    model = Sequential()
    # model.add(brnn)

    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=10,
                     border_mode="valid",
                     # activation="relu",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_length=3))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(nbfilter * 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    return model

def get_bag_data_1_channel(seqs, labels, max_len = 2695):
    bags = []
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        #bag_subt.append(tri_fea.T)
        bags.append(np.array(tri_fea))
    return bags, labels

def get_domain_features(in_file = 'rbps_HT.txt'):
    protein_list = []
    with open('protein_list', 'r') as fp:
        for line in fp:
            protein_list.append(line[1:-1])
    domain_dict = {}
    fp = open(in_file, 'r')
    index = 0
    for line in fp:
        values = line.rstrip().split()
        vals = [float(val) for val in values]
        domain_dict[protein_list[index]] = vals
        index = index + 1
    fp.close()

    return domain_dict

def get_all_data():
    inter_pair_dict, rna_seq_dict, protein_set = load_rnacomend_data()
    #pdb.set_trace()
    #protein_set = set(inter_pair_dict.values())
    protein_list = []
    for protein in protein_set:
        protein_list.append(protein)

    data = get_rnarecommend(inter_pair_dict, rna_seq_dict, protein_list)
    labels = data["Y"]
    seqs = data["seq"]
    #training_val_indice, train_val_label, test_indice, test_label = split_training_validation(labels)
    x_index = range(len(labels))
    #pdb.set_trace()
    training_val_indice, test_indice, train_val_label, test_label = train_test_split(x_index, labels, train_size = 0.8, stratify= labels)
    train_bags, train_labels, test_bags, test_labels = get_all_rna_mildata(seqs, labels, training_val_indice, train_val_label, test_indice, test_label)
    return np.array(train_bags), np.array(train_labels), np.array(test_bags), np.array(test_labels), protein_list

def run_mlcnn():
    train_bags, train_labels, test_bags, test_labels, protein_list = get_all_data()
    print train_bags.shape, train_labels.shape, test_bags.shape, test_labels.shape
    #pdb.set_trace()
    clf = set_cnn_model(nbfilter = 101)
    clf.add(Dense(68, activation='sigmoid'))
    #pdb.set_trace()
    clf.add(Reshape((68, 1), input_shape=(68,)))

    clf.add(LSTM(68, return_sequences= True))
    clf.add(Flatten())
    clf.add(Dense(68, activation='sigmoid'))

    clf.compile(optimizer=Adam(), loss='binary_crossentropy') #'mean_squared_error'



    clf.fit(train_bags, train_labels, batch_size=64, nb_epoch=20, verbose=0, class_weight= 'auto')

    preds = clf.predict(test_bags)

    print 'Macro-AUC', roc_auc_score(test_labels, preds, average='macro')
    print 'Micro-AUC',roc_auc_score(test_labels, preds, average='micro')
    print 'weight-AUC',roc_auc_score(test_labels, preds, average='weighted')


    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0

    print f1_score(test_labels, preds, average='macro')
    print f1_score(test_labels, preds, average='micro')
    print f1_score(test_labels, preds, average='weighted')



run_mlcnn()


