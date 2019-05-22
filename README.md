# JNET
This is the implementation for the Joint Network Embedding and Topic Embedding (JNET). We provide all the source codes for the algorithm and related baselines.

## Quick Start (For Linux and Mac)
* Download the [JNET repo](https://github.com/Linda-sunshine/JNET.git) to your local machine.
* Download the [data](http://www.cs.virginia.edu/~lg5bt/files/data.zip) to the directory that **./src** lies in.
* Compile the whole project with [complie file](https://github.com/Linda-sunshine/JNET/blob/master/compile).
```
./compile
```
* Run the algorithm with default setting with [run file](https://github.com/Linda-sunshine/JNET/blob/master/run).
```
./run
```
## Questions regarding running JNET and Baselines
### Q1: What's inside the ./data folder?
**./data** folder has all the data needed for the experiments reported in the paper, including both Yelp data (./data/CoLinAdapt/YelpNew/) and StackOverflow data (./data/CoLinAdapt/StackOverflow/). For example, **./data/CoLinAdapt/Yelp** contains the following files which are needed for running experiments with Yelp dataset:
```
CrossGroups_800.txt
AmazonFriends.txt
fv_lm_DF_1000.txt
GlobalWeights.txt
SelectedVocab.csv
./Users
```
* **CrossGroups_800.txt** contains the feature indexes for 800 feature groups.
* **fv_lm_DF_1000.txt** contains the 1000 textual features selected for training language models.
* **GlobalWeights.txt** contains the weights for sentiment features trained on a separate data, which serves as a base model.
* **SelectedVocab.csv** contains the 5000 sentiment features used for training sentiment models.
* **./Users** folder contains 9760 users.
* **AmazonFriends.txt** contains the friendship information. In each line, the first string is the user ID and the following strings are his/her friends' IDs.

### Q2: How to run the algorithm HUB with different parameters?
We use **-model** to select different algorithms and the default one is HUB.
The following table lists all the parameters for HUB:
```
Usage: java execution [options] training_folder
options:
-data: specify the dataset used for training (default YelpNew)
option: Amazon, YelpNew
-eta1: coefficient for the scaling in each user group's regularization (default 0.05)
-eta2: coefficient for the shifting in each user group's regularization (default 0.05)
-eta3: coefficient for the scaling in super user's regularization (default 0.05)
-eta4: coefficient for the shifting in super user's regularization (default 0.05)
-model: specific training model,
option: Base-base, MT-SVM-mtsvm, GBSSL-gbssl, HUB-hubMTLinAdapt+kMeans-mtclinkmeans,  cLinAdapt-mtclindp, cLinAdapt+HDP-mtclinhdp (default hub)
-M: the size of the auxiliary variables in the posterior inference of the group indicator (default 6)
-alpha: concentraction parameter for the first-layer DP, i.e., collective identities (default 0.01)
-beta: concentraction parameter for the prior Dirichlet Distribution of language model (default 0.05)
-eta: concentraction parameter for the second-layer DP, i.e., user mixture (default 0.05)
-nuI: number of iterations for sampling (default 30)
-sdA: variance for the normal distribution for the prior of shifting parameters (default 0.0425)
-sdB: variance for the normal distribution for the prior of scaling parameters (default 0.0425)
-burn: number of iterations in burn-in period (default 5)
-thin: thinning of sampling chain (default 5)
-rho: the sparsity parameter for block model (default 0.05)
--------------------------------------------------------------------------------
```

### Q3: How to run baselines?
As reported in the paper, we have six baselines. We can use "-model" to select baselines, the corresponding parameters are specified in the above table. For example, I would like to run MT-SVM on Amazon dataset, then after the compiling, input the following command line:
```
./run -model mtsvm -data Amazon
```

### Q4: What does the output mean?
The final output of the algorithm is the sentiment classification performance of HUB, which is shown as follows. The first line is the information of the parameters used in the current run, which is the same as introduced in the first part. We also print out the confusion matrix, together with Micro F1 and Macro F1 for both classes. Micro F1 performance is calculated by aggregating all users’ reviews while Macro F1 is the average performance over all the users. Class 0 is negative class and Class 1 is positive class. Due to the randomness of initialization of model weights, the final sentiment performance may vary in different runs. More detail about the experimental setting can be found in our paper.
```
MTCLinAdaptWithMMB[dim:801,dimSup:3072,lmDim:1000,M:6,rho:0.09000,alpha:0.0010,eta:0.0500     ,beta:0.0100,nScale:(0.050,0.050),#Iter:60,N1(0.000,0.043),N2(1.000,0.043)]
Micro confusion matrix:
         0       1
0       21702   2909
1       3480    78755
Micro F1:
Class 0: 0.8717         Class 1: 0.9610
Macro F1:
Class 0: 0.8635+0.2611  Class 1: 0.9590+0.1031
```
## Citing HUB
We encourage you to cite our work if you have referred it in your work. You can use the following BibTeX citation:
```
@inproceedings{gong2018sentiment,
  title={When Sentiment Analysis Meets Social Network: A Holistic User Behavior Modeling in Opinionated Data},
  author={Gong, Lin and Wang, Hongning},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1455--1464},
  year={2018},
  organization={ACM}
}
```
