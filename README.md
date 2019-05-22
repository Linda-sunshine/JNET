# JNET
This is the implementation for the Joint Network Embedding and Topic Embedding (JNET). We provide all the source codes for the algorithm and related baselines.

## Quick Start (For Linux and Mac)
* Download the [JNET repo](https://github.com/Linda-sunshine/JNET.git) to your local machine.
* Download the [data](http://www.cs.virginia.edu/~lg5bt/files/jnet_data.zip) to the directory that **./src** lies in.
* Compile the whole project with [complie file](https://github.com/Linda-sunshine/JNET/blob/master/compile).
```
./compile
```
* Run the algorithm with default setting with [run file](https://github.com/Linda-sunshine/JNET/blob/master/run).
```
./run
```
## Questions regarding running JNET
### Q1: What's inside the ./data folder?
**./data** folder has all the data in differnet settings for running experiments, including both Yelp data (./data/YelpNew/) and StackOverflow data (./data/StackOverflow/). 
```
./ColdStart4Docs
./ColdStart4Edges
./CV4Edges
./Model
./output
./StackOverflow
./YelpNew
```
* **ColdStart4Docs** is used for document modeling in cold-start setting. We reserve three different sets of users regarding to their connectivity and name them as light, medium and heavy users. We preserve these users' all documents for testing and only utilized their connectivity to learn user embedding. We performed 5-fold cross validation. 
* **ColdStart4Edges** is used for link prediction in cold-start setting. We reserve three different sets of users regarding to their document size and name them as light, medium and heavy users. We preserve these users' all social connections for testing and only utilized their doucments to learn user embedding. We performed 5-fold cross validation.
* **CV4Edges** is used for link prediction. We reserve three different sets of users regarding to their document size and name them as light, medium and heavy users. We split users' connections into 5 folds. Each time, we preserve one fold for testing and use these other four folds for training user embeddings, together with their textul documents. The learned user embeddings are fed into link prediction for performance eveluation.
* **Model** provides configuration files used for running JNET.
* **StackOverflow** contains users's text posts on StackOverflow and constructed network based on "rely-to" information. 
* **YelpNew** contains users's texts reviews from Yelp dataset and facebook friendship data. 

### Q2: How to run the algorithm JNET with different parameters?
The following table lists the major parameters for JNET:
```
Usage: java execution [options] training_folder options:
-prefix: the directory of the data (default ./data/)
-data: specific dataset used for training (default YelpNew) ption: YelpNew, StackOverflow
-emIter: the number of iterations in variationla EM algorithm (default 50)
-nuTopics: the number of topics (default 30)
-dim: the dimension of user embeddings and topic embeddings (default 10)
-multi: run the algorithm in multi-threading or not (default true)
-saveDir: directory for saving the learned user embeddings and topic embeddings (default ./data/output/)
-kFold: speicfy which fold to train (default 0), option: 0, 1, 2, 3, 4
-mode: specify the experimental setting (default cv4doc)option: cv4doc, cv4edge
-coldStart: whether we perform experiments in cold-start setting or not (default: false)
--------------------------------------------------------------------------------
```
Especially, the parameter **mode** specifies the setting of the experiments, **cv4doc** is for the task of documenet modeling and **cv4edge** is for the task of link prediction. Both of them learn user embeddings, which are saved in **saveDir**. If the **saveDir** is not specified, its default directory is **./data/output/**. **coldStart** specififies whether the current setting is cold-start setting or not. More details of the experimental setting can be found in the paper.

### Q3: What's inside the output folder?
A lot of information is saved for verifying the model.

### Q4: What does the output mean?
In running the algorithm, a lot of information is printed out for reference, as follows. The first line is the information of the parameters used in the current run, which is the same as introduced in the first part. We also print out the confusion matrix, together with Micro F1 and Macro F1 for both classes. Micro F1 performance is calculated by aggregating all users’ reviews while Macro F1 is the average performance over all the users. Class 0 is negative class and Class 1 is positive class. Due to the randomness of initialization of model weights, the final sentiment performance may vary in different runs. More detail about the experimental setting can be found in our paper.
```
Load 5000 2-gram new features from /zf8/lg5bt/DataSigir/StackOverflow/StackOverflowSelectedVocab.txt...
10808 users/244360 docs are loaded from /zf8/lg5bt/DataSigir/StackOverflow/Users...
[Stat]Stat as follow:
48838	48834	48802	48879	49007	
[Info]Total user size: 10805, total doc size: 244360, users with friends: 10145, total edges: 62394.000.
[Info]Training size: 0(0.00), adaptation size: 0(0.00), and testing size: 0(0.00)
[EUB]Mode: CV4DOC, Dim: 10, Topic number: 40, EmIter: 130, VarIter: 10, TrainInferIter: 1, TestInferIter: 1500, ParamIter: 20.
[Info]Construct network with interactions and non-interactions....Finish network construction!
==========Start 0-fold cross validation=========
[Info]Initializing topics, documents and users...
[ModelParam]alpha:1.000, gamma:0.100,tau:1.000,xi:2.000, rho:1.00000
----------Start EM 0 iteraction----------
[Multi-E-step] 0 iteration, likelihood(d:t:u)=(-201570933.04, -2310.97, -243770268.40), converge to 1.00000000
[Multi-E-step] 1 iteration, likelihood(d:t:u)=(-4290787.24, -1287.47, -240353105.84), converge to 0.45065961
[Multi-E-step] 2 iteration, likelihood(d:t:u)=(-2904760.94, -1554.37, -237554957.03), converge to 0.01710194
[Multi-E-step] 3 iteration, likelihood(d:t:u)=(-2877074.05, -1471.47, -236862749.17), converge to 0.00299415
[Multi-E-step] 4 iteration, likelihood(d:t:u)=(-2853381.60, -1510.61, -236189439.40), converge to 0.00290715
[Multi-E-step] 5 iteration, likelihood(d:t:u)=(-2828475.45, -1525.13, -235522478.87), converge to 0.00289424
[Multi-E-step] 6 iteration, likelihood(d:t:u)=(-2806558.76, -1525.07, -234860435.06), converge to 0.00286953
[Multi-E-step] 7 iteration, likelihood(d:t:u)=(-2785307.87, -1524.94, -234203540.26), converge to 0.00285333
[Multi-E-step] 8 iteration, likelihood(d:t:u)=(-2764490.17, -1524.95, -233551802.00), converge to 0.00283790
[Multi-E-step] 9 iteration, likelihood(d:t:u)=(-2744090.84, -1524.92, -232905165.37), converge to 0.00282262
[Multi-E-step] 10 iteration, likelihood(d:t:u)=(-2724098.83, -1524.88, -232263580.66), converge to 0.00280745
[Info]Finish E-step: loglikelihood is: -234989204.37311.
[M-step]Estimate alpha....
[M-step]Estimate beta....
[ModelParam]alpha:16.057, gamma:0.100,tau:1.000,xi:2.000, rho:1.00000
```
