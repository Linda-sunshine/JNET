package mains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Analyzer.MultiThreadedStackOverflowAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures.EmbeddingParameter;
import structures._Corpus;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.UserEmbedding.EUB4ColdStart_multithreading;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EUBExecution {


    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        EmbeddingParameter param = new EmbeddingParameter(args);

        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", param.m_prefix, param.m_data, param.m_data);
        String userFolder = String.format("%s/%s/Users", param.m_prefix, param.m_data);
        String friendFile = String.format("%s/%s/%sFriends.txt", param.m_prefix, param.m_data, param.m_data);
        String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", param.m_prefix, param.m_data, param.m_data);
        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", param.m_prefix, param.m_data, param.m_data, param.m_kFold);

        MultiThreadedNetworkAnalyzer analyzer = null;
        int kFold = 5;

        if(param.m_data.equals("StackOverflow")){
            analyzer = new MultiThreadedStackOverflowAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, true);
        } else
            analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        // we store the interaction information before-hand, load them directly
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();

        // if it is cv for doc, use all the interactions + part of docs
        if(param.m_mode.equals("cv4doc") && !param.m_coldStartFlag){
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        }
        // if it is cv for edge, use all the docs + part of edges
        else if(param.m_mode.equals("cv4edge") && !param.m_coldStartFlag){
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        // cold start for doc, use all edges, test doc perplexity on light/medium/heavy users
        else if(param.m_mode.equals("cv4doc") && param.m_coldStartFlag) {
            cvIndexFile = String.format("%s/%s/ColdStart/%s_cold_start_4docs_fold_%d.txt", param.m_prefix, param.m_data,
                    param.m_data, param.m_kFold);
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        }
        // cold start for edge, use all edges, learn user embedding for light/medium/heavy users
        else if(param.m_mode.equals("cv4edge") && param.m_coldStartFlag){
            cvIndexFile4Interaction = String.format("%s/%s/ColdStart/%s_cold_start_4edges_fold_%d_interactions.txt",
                    param.m_prefix, param.m_data, param.m_data, param.m_kFold);
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        // answerer recommendation for stackoverflow data only
        else if(param.m_mode.equals("ansrec")){
            cvIndexFile = String.format("%s/%s/AnswerRecommendation/%sCVIndex4Recommendation.txt",
                    param.m_prefix, param.m_data, param.m_data);
            friendFile = String.format("%s/%s/AnswerRecommendation/%sFriends4Recommendation.txt",
                    param.m_prefix, param.m_data, param.m_data);
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        }
        /***Start running joint modeling of user embedding and topic embedding****/
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6;//these two parameters must be larger than 1!!!
        _Corpus corpus = analyzer.getCorpus();

        long start = System.currentTimeMillis();
        LDA_Variational tModel = null;

        if(param.m_multiFlag && param.m_coldStartFlag) {
            tModel = new EUB4ColdStart_multithreading(param.m_emIter, emConverge, beta, corpus, lambda, param.m_number_of_topics,
                    alpha, param.m_varIter, varConverge, param.m_embeddingDim);
        } else if(param.m_multiFlag && !param.m_coldStartFlag){
            tModel = new EUB_multithreading(param.m_emIter, emConverge, beta, corpus, lambda, param.m_number_of_topics,
                    alpha, param.m_varIter, varConverge, param.m_embeddingDim);
        } else{
            tModel = new EUB(param.m_emIter, emConverge, beta, corpus, lambda, param.m_number_of_topics, alpha,
                    param.m_varIter, varConverge, param.m_embeddingDim);
        }

        ((EUB) tModel).initLookupTables(analyzer.getUsers());
        ((EUB) tModel).setModelParamsUpdateFlags(param.m_alphaFlag, param.m_gammaFlag, param.m_betaFlag,
                param.m_tauFlag, param.m_xiFlag, param.m_rhoFlag);
        ((EUB) tModel).setMode(param.m_mode);

        ((EUB) tModel).setTrainInferMaxIter(param.m_trainInferIter);
        ((EUB) tModel).setTestInferMaxIter(param.m_testInferIter);
        ((EUB) tModel).setParamMaxIter(param.m_paramIter);
        ((EUB) tModel).setStepSize(param.m_stepSize);
        ((EUB) tModel).setGamma(param.m_gamma);
        ((EUB) tModel).setData(param.m_data);

        if(param.m_mode.equals("ansrec")){
            String questionIds = String.format("%s/%s/AnswerRecommendation/%sSelectedQuestions.txt",
                    param.m_prefix, param.m_data, param.m_data);
            ((EUB) tModel).loadQuestionIds(questionIds);
        }

        if(param.m_multiFlag && param.m_coldStartFlag){
            ((EUB4ColdStart_multithreading) tModel).fixedCrossValidation(param.m_kFold, param.m_saveDir);
        } else{
            ((EUB) tModel).fixedCrossValidation(param.m_kFold, param.m_saveDir);
        }

        String saveDir = param.m_saveDir + String.format("%s_nuTopics_%d_dim_%d_fold_%d_%d", param.m_data,
                param.m_number_of_topics, param.m_embeddingDim, param.m_kFold);
        File dir = new File(saveDir);
        if(!dir.exists())
            dir.mkdir();

        tModel.printBeta(param.m_saveDir);
        if(param.m_mode.equals("ansrec")){
            ((EUB) tModel).printTopicEmbeddingAsMtx(param.m_saveDir);
            ((EUB) tModel).printTheta(param.m_saveDir);
        }
        long end = System.currentTimeMillis();

        // the total time of training and testing in the unit of hours
        double hours = (end - start)/((1000*60*60) * 1.0);
        System.out.print(String.format("[Time]This training+testing process took %.2f hours.\n", hours));

    }
}
