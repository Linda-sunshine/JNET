package mains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Analyzer.MultiThreadedStackOverflowAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.UserEmbedding.EUB4ColdStart_multithreading;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The main entrance for calling EUB model
 */
public class EUBMain {

    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "YelpNew"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/JNET/";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

        int kFold = 5;
        int time = 2;
        int k = 0;

        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
        String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", prefix, dataset, dataset);
//        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, dataset, k);
        String cvIndexFile4NonInteraction = String.format("%s/%s/%sCVIndex4NonInteraction_time_%d.txt", prefix, dataset, dataset, time);

        MultiThreadedNetworkAnalyzer analyzer = null;

        if (dataset.equals("StackOverflow")) {
            analyzer = new MultiThreadedStackOverflowAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, true);
        } else
            analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        /***Our algorithm JNET****/
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();

        String mode = "cv4doc"; // "cv4edge" "cs4doc"--"cold start for doc" "cs4edge"--"cold start for edge"
        boolean coldStartFlag = false;

        //if it is cv for doc, use all the interactions + part of docs
        if (mode.equals("cv4doc") && !coldStartFlag) {
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        }
        // if it is cv for edge, use all the docs + part of edges
        else if (mode.equals("cv4edge") && !coldStartFlag) {
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        // cold start for doc, use all edges, test doc perplexity on light/medium/heavy users
        else if (mode.equals("cv4doc") && coldStartFlag) {
            cvIndexFile = String.format("./data/DataEUB/ColdStart4Docs/%s_cold_start_4docs_fold_%d.txt", dataset, k);
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        }
        // cold start for edge, use all edges, learn user embedding for light/medium/heavy users
        else if (mode.equals("cv4edge") && coldStartFlag) {
            cvIndexFile4Interaction = String.format("./data/DataEUB/ColdStart4Edges/%s_cold_start_4edges_fold_%d_interactions.txt", dataset, k);
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        // answerer recommendation for stackoverflow data only
        else if (mode.equals("ansrec")) {
            cvIndexFile = String.format("%s/%s/AnswerRecommendation/StackOverflowCVIndex4Recommendation.txt", prefix, dataset);
            friendFile = String.format("%s/%s/AnswerRecommendation/StackOverflowFriends4Recommendation.txt", prefix, dataset);
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        }


        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 50, number_of_topics = 20, varMaxIter = 10, embeddingDim = 10, trainIter = 1, testIter = 1500;
        //these two parameters must be larger than 1!!!
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6, stepSize = 0.001;
        boolean alphaFlag = true, gammaFlag = true, betaFlag = true, tauFlag = false, xiFlag = false, rhoFlag = false;
        boolean multiFlag = true, adaFlag = false;
        _Corpus corpus = analyzer.getCorpus();

        long start = System.currentTimeMillis();
        LDA_Variational tModel = null;

        if(multiFlag && coldStartFlag)
            tModel = new EUB4ColdStart_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        else if(multiFlag && !coldStartFlag)
            tModel = new EUB_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        else
            tModel = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);

        ((EUB) tModel).initLookupTables(analyzer.getUsers());
        ((EUB) tModel).setModelParamsUpdateFlags(alphaFlag, gammaFlag, betaFlag, tauFlag, xiFlag, rhoFlag);
        ((EUB) tModel).setMode(mode);

        ((EUB) tModel).setTrainInferMaxIter(trainIter);
        ((EUB) tModel).setTestInferMaxIter(testIter);
        ((EUB) tModel).setStepSize(stepSize);

        if(mode.equals("ansrec")){
            String questionIds = String.format("%s/%s/AnswerRecommendation/StackOverflowSelectedQuestions.txt",
                    prefix, dataset);
            ((EUB) tModel).loadQuestionIds(questionIds);
        }

        long current = System.currentTimeMillis();
        String saveDir = String.format("./data/embeddingExp/eub/%s_emIter_%d_nuTopics_%d_varIter_%d_trainIter_%d_testIter_%d_dim_%d_ada_%b/" +
                "fold_%d_%d", dataset, emMaxIter, number_of_topics, varMaxIter, trainIter, testIter, embeddingDim, adaFlag, k, current);

        if(multiFlag && coldStartFlag)
            ((EUB4ColdStart_multithreading) tModel).fixedCrossValidation(k, saveDir);
        else
            ((EUB) tModel).fixedCrossValidation(k, saveDir);

        tModel.printBeta(saveDir);
        if(mode.equals("ansrec")){
            ((EUB) tModel).printTopicEmbeddingAsMtx(saveDir);
            ((EUB) tModel).printTheta(saveDir);
        }
        long end = System.currentTimeMillis();

        // the total time of training and testing in the unit of hours
        double hours = (end - start)/((1000*60*60) * 1.0);
        System.out.print(String.format("[Time]This training+testing process took %.4f hours.\n", hours));
    }
}
