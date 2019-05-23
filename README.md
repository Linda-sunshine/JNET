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
-data: specific dataset used for training (default YelpNew) option: YelpNew, StackOverflow
-emIter: the number of iterations in variationla EM algorithm (default 50)
-nuTopics: the number of topics (default 30)
-dim: the dimension of user embeddings and topic embeddings (default 10)
-multi: run the algorithm in multi-threading or not (default true)
-saveDir: directory for saving the learned user embeddings and topic embeddings (default ./data/output/)
-kFold: speicfy which fold to train (default 0), option: 0, 1, 2, 3, 4
-mode: specify the experimental setting (default cv4doc) option: cv4doc, cv4edge
-coldStart: whether we perform experiments in cold-start setting or not (default: false)
--------------------------------------------------------------------------------
```
Especially, the parameter **mode** specifies the setting of the experiments, **cv4doc** is for the task of documenet modeling and **cv4edge** is for the task of link prediction. Both of them learn user embeddings, which are saved in **saveDir**. If the **saveDir** is not specified, its default directory is **./data/output/**. The parameter **coldStart** specififies whether the current setting is cold-start setting or not. More details of the experimental setting can be found in the paper.

### Q3: What's inside the output folder?
A lot of information learned from the training process is saved for other applications. In the folder **./data/output**, we further create sub-directory for each run and the sub-directory is named in the fashion **YelpNew_nuTopics_30_dim_10_fold_0** to record the current parameters. We have the following files printed out for further use. 
```
Beta.txt
EUB_embedding_dim_10_fold_0.txt
EUB_EmbeddingWithDetails_dim_10_fold_0.txt
Perplexity.txt
TopicEmbedding.txt
```
* **Beta.txt** is the topics learned from topic modeling. The dimension of each topic is the vocabulary size, i.e., 5000, and the value is in log space.
* **EUB_embedding_dim_10_fold_0.txt** is the learned user embeddings. The first line specify the number of users and the latent dimension.
* **EUB_EmbeddingWithDetails_dim_10_fold_0.txt** is the learned user embeddings in detail. Each user has three sets of parameters printed, user id, variational mean of user embedding and variational sigma of user embedding.
* **Perplexity.txt** is the perplexity result performed on testing documents and will be available for mode "cv4doc".
* **TopicEmbedding.txt** is the learned topic embeddings in detail. Each topic also has three sets of parameters printed, topic id, variational mean of topic embedding and variational sigma of topic embedding.
With the learned user embeddings, we can perform other tasks such as link prediction.

### Q4: What does the output mean?
In running the algorithm, a lot of information is printed out for reference, as follows. Line 1-5 is basic information of the input data, especially the **[stat]** the number of documents in each fold since we are performing 5-fold cross validation. Line 6 describes the basic information of the model. From Line 8, the algorithm starts performing EM algorithm. In E-step, the decomposed likelihood regarding to document, topic and user are printed out for tracking. In M-step, model parameters are estimated as shown in Line 24-25. E-step and M-step are repeated until convergence. Starting from Line 28, the algorithm tests the perplexity on the held-out set of documents, and the performance is reported in Line 32. It also prints out the save directory and the time used to run.
```
1 Load 5000 2-gram new features from /zf8/lg5bt/DataSigir/StackOverflow/StackOverflowSelectedVocab.txt...
2 10808 users/244360 docs are loaded from /zf8/lg5bt/DataSigir/StackOverflow/Users...
3 [Stat]Stat as follow:
4 48838	48834	48802	48879	49007	
5 [Info]Total user size: 10805, total doc size: 244360, users with friends: 10145, total edges: 62394.000.
6 [JNET]Mode: CV4DOC, Dim: 10, Topic number: 40, EmIter: 100, VarIter: 10, TrainInferIter: 10, TestInferIter: 100, ParamIter: 20.
7 [Info]Construct network with interactions and non-interactions....Finish network construction!
8 ==========Start 0-fold cross validation=========
9 [Info]Initializing topics, documents and users...
10 [ModelParam]alpha:1.000, gamma:0.100,tau:1.000,xi:2.000, rho:1.00000
11 ----------Start EM 0 iteraction----------
12 [Multi-E-step] 0 iteration, likelihood(d:t:u)=(-201570933.04, -2310.97, -243770268.40), converge to 1.00000000
13 [Multi-E-step] 1 iteration, likelihood(d:t:u)=(-4290787.24, -1287.47, -240353105.84), converge to 0.45065961
14 [Multi-E-step] 2 iteration, likelihood(d:t:u)=(-2904760.94, -1554.37, -237554957.03), converge to 0.01710194
15 [Multi-E-step] 3 iteration, likelihood(d:t:u)=(-2877074.05, -1471.47, -236862749.17), converge to 0.00299415
16 [Multi-E-step] 4 iteration, likelihood(d:t:u)=(-2853381.60, -1510.61, -236189439.40), converge to 0.00290715
17 [Multi-E-step] 5 iteration, likelihood(d:t:u)=(-2828475.45, -1525.13, -235522478.87), converge to 0.00289424
18 [Multi-E-step] 6 iteration, likelihood(d:t:u)=(-2806558.76, -1525.07, -234860435.06), converge to 0.00286953
19 [Multi-E-step] 7 iteration, likelihood(d:t:u)=(-2785307.87, -1524.94, -234203540.26), converge to 0.00285333
20 [Multi-E-step] 8 iteration, likelihood(d:t:u)=(-2764490.17, -1524.95, -233551802.00), converge to 0.00283790
21 [Multi-E-step] 9 iteration, likelihood(d:t:u)=(-2744090.84, -1524.92, -232905165.37), converge to 0.00282262
22 [Multi-E-step] 10 iteration, likelihood(d:t:u)=(-2724098.83, -1524.88, -232263580.66), converge to 0.00280745
23 [Info]Finish E-step: loglikelihood is: -234989204.37311.
24 [M-step]Estimate alpha....
25 [M-step]Estimate beta....
26 [ModelParam]alpha:16.057, gamma:0.100,tau:1.000,xi:2.000, rho:1.00000
27 ....
28 In one fold, (train: test)=(150179 : 37558)
29 [Info]Current mode is cv for docs, start evaluation....
30 [Inference]Likelihood: -9068817.34
31 [Inference]Likelihood: -9068817.46
32 [Stat]TestInferIter=100, perplexity=3364.7239, totalWords=1116698, allLogLikelihood=-9068817.4570, avgLogLikelihood=-241.4617
33 [Info]Finish training, start saving data...
34 TopWord FilePath:./data/output/TopKWords.txt
35 [Time]This training+testing process took 0.85 hours.
```


