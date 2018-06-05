# iDeepM
RNA-binding proteins (RBPs) play crucial roles in both transcriptional and post-transcriptional gene regulation. RNAs interact with RBPs to conduct their functions and their dysregulation are associated with many diseases. How to computationally detect RNA-binding proteins based on collected data are needed. Previous prediction approaches learns a binary classification model for a target-specific RBP to predict whether a RNA binds to a RBP or not.  However, a RNA can bind to multiple RBPs. Thus, when given a RNA sequence,  it would be more useful in real applications to predict its binding proteins if the RNA-specific prediction is formulated as a multi-label classification problem, whose inputs are RNA sequences and labels are available RBPs. <br>

In this study, we present a multi-label deep learning method called iDeepM to predict a set of binding RBPs for RNA sequences. We first collect a set of RBPs and their RNA targets from CLIP-seq data. For each RNA, we consider its binding RBPs as its labels. Next, we train a joint model based on multi-label deep learning to predict how a RNA is attached by a set of RBPs. iDeepM uses a convolutional neural network to learn shared high-level features across RBPs, and long short term memory network (LSTM) under multi-label learning framework to infer the dependency and combinations of RBPs. <br><br>

# Dependency:
python 2.7 <br>
Keras 1.2 (https://github.com/keras-team/keras ) and its backend is theano 0.9 <br>
Sklearn (https://github.com/scikit-learn/scikit-learn)

# Data
The data is downloaded from https://github.com/gianlucacorrado/RNAcommender/tree/master/examples, it contains 67 proteins and binding UTR sequences
