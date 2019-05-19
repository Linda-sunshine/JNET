package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import structures._Doc;
import structures._Review;
import structures._User;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * The analyzer is used to further analyze each post in StackOverflow, i.e., question and answer
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class MultiThreadedStackOverflowAnalyzer  extends MultiThreadedNetworkAnalyzer {

    public MultiThreadedStackOverflowAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // one map for indexing all questions
    HashMap<Integer, _Review> m_questionMap = new HashMap<>();
    HashMap<Integer, _Review> m_answerMap = new HashMap<>();
    HashMap<Integer, ArrayList<Integer>> m_questionAnswersMap = new HashMap<>();

    // assume we already have the cv index for all the documents
    public void buildQuestionAnswerMap() {

        // step 1: find all the questions in the data first
        for (_User u : m_users) {
            for (_Review r : u.getReviews()) {
                if (r.getParentId() == -1) {
                    m_questionMap.put(r.getPostId(), r);
                } else{
                    m_answerMap.put(r.getPostId(), r);
                }
            }
        }

        System.out.format("Total number of users %d.\n", m_users.size());
        System.out.format("Network map size is %d. \n", m_networkMap.size());

        // step 2: find all the answers to the corresponding questions
        HashSet<String> users = new HashSet<>();
        HashMap<String, HashSet<String>> net = new HashMap<>();
        for (_User u : m_users) {
            String ui = u.getUserID();
            // find all the questions in the data first
            for (_Review r : u.getReviews()) {
                int questionId = r.getParentId();
                if (questionId != -1 && m_questionMap.containsKey(questionId)) {
                    if (!m_questionAnswersMap.containsKey(questionId))
                        m_questionAnswersMap.put(questionId, new ArrayList<Integer>());
                    m_questionAnswersMap.get(questionId).add(r.getPostId());
                }
            }
        }

        // step 3: calculate stat of the answers to the questions
        double avg = 0;
        int lgFive = 0;
        for (int qId : m_questionAnswersMap.keySet()) {
            avg += m_questionAnswersMap.get(qId).size();
            if (m_questionAnswersMap.get(qId).size() > 5)
                lgFive++;
        }
        avg /= m_questionAnswersMap.keySet().size();
        System.out.format("[stat] Total questions: %d, questions with answers: %d,questions with > 5 answers: %d, avg anser: %.2f\n",
                m_questionMap.size(), m_questionAnswersMap.size(), lgFive, avg);
    }


    // Load one file as a user here.
    protected void loadUser(String filename, int core){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            String userID = extractUserID(file.getName()); //UserId is contained in the filename.

            // Skip the first line since it is user name.
            reader.readLine();
            int postId, parentId, score;
            String source;
            ArrayList<_Review> reviews = new ArrayList<_Review>();

            _Review review;
            int ylabel, index = 0;
            long timestamp;
            while((line = reader.readLine()) != null){
                postId = Integer.valueOf(line.trim());
                source = reader.readLine(); // review content
                parentId = Integer.valueOf(reader.readLine().trim()); // parentId
                ylabel = Integer.valueOf(reader.readLine()); // ylabel
                timestamp = Long.valueOf(reader.readLine());

                // Construct the new review.
                if(ylabel != 3){
                    ylabel = (ylabel >= 4) ? 1:0;
                    review = new _Review(-1, postId, source, ylabel, parentId, userID, timestamp);
                    if(AnalyzeDoc(review, core)) { //Create the sparse vector for the review.
                        reviews.add(review);
                        review.setID(index++);
                    }
                }
            }
            if(reviews.size() > 1){//at least one for adaptation and one for testing
                synchronized (m_allocReviewLock) {
                    if(m_allocateFlag)
                        allocateReviews(reviews);
                    m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
                    m_corpus.addDocs(reviews);
                }
            } else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
                review = reviews.get(0);
                synchronized (m_rollbackLock) {
                    rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
                }
            }

            reader.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // the function selects questions for candidate recommendation
    ArrayList<Integer> m_selectedQuestions = new ArrayList<>();
    ArrayList<Integer> m_selectedAnswers = new ArrayList<>();

    public void selectQuestions4Recommendation(){
        super.constructUserIDIndex();
        for(int qId: m_questionAnswersMap.keySet()){
            String uiId = m_questionMap.get(qId).getUserID();
            if(!m_networkMap.containsKey(uiId)) continue;

            ArrayList<Integer> answers = m_questionAnswersMap.get(qId);
            int nuOfAns = m_questionAnswersMap.get(qId).size();
            if(nuOfAns > 1 && nuOfAns <= 5){
                boolean flag = true;
                for(int aId: answers){
                    _User uj = m_users.get(m_userIDIndex.get(m_answerMap.get(aId).getUserID()));
                    // the user will not be trained in the training
                    if(!containsQuestion(uj)){
                        flag = false;
                    }
                }
                if(flag) {
                    m_selectedQuestions.add(qId);
                }
            }
        }

        // further filter the selected question based on network
        for(int qId: m_selectedQuestions){
            String uiId = m_questionMap.get(qId).getUserID();
            HashSet<String> uiFrds = m_networkMap.get(uiId);
            if(uiFrds == null){
                System.out.println("The user does not have any friends!");
                continue;
            }
            // access answers of one question
            for(int aId: m_questionAnswersMap.get(qId)){
                String ujId = m_answerMap.get(aId).getUserID();
                HashSet<String> ujFrds = m_networkMap.get(ujId);
                if(uiFrds != null && ujFrds != null && uiFrds.contains(ujId) && ujFrds.contains(uiId)){
                    if(!m_testInteractions.containsKey(qId))
                        m_testInteractions.put(qId, new HashSet<String>());
                    m_testInteractions.get(qId).add(ujId);
                    m_selectedAnswers.add(aId);
                }
            }
        }

        // assign the selected questions to the array list
        m_selectedQuestions.clear();
        for(int qId: m_testInteractions.keySet())
            m_selectedQuestions.add(qId);
        System.out.println("Total number of valid questions: " + m_selectedQuestions.size());
        System.out.println("Total number of answers: " + m_selectedAnswers.size());
    }

    // remove connections based on selected questions and answers
    // key: question id, value: user ids that answered this question
    HashMap<Integer, HashSet<String>> m_testInteractions = new HashMap<>();
    // key: question id, value: user ids that did not answer this question
    HashMap<Integer, HashSet<Integer>> m_testNonInteractions = new HashMap<>();

    public void refineNetwork4Recommendation(int time, String prefix){

        int remove = 0;

        // sample the testing non-interactions
        sampleNonInteractions(time);
        saveNonInteractions(prefix, time);
        saveInteractions(prefix);

        // remove the testing interactions from the training network
        for(int qId: m_testInteractions.keySet()){
            String ui = m_questionMap.get(qId).getUserID();
            for(String uj: m_testInteractions.get(qId)){
                if(m_networkMap.get(ui).contains(uj)){
                    m_networkMap.get(ui).remove(uj);
                    remove++;
                }
                if(m_networkMap.get(uj).contains(ui)){
                    m_networkMap.get(uj).remove(ui);
                    remove++;
                }
            }
        }
        System.out.println(remove + " edges are removed!!!");
    }

    public void sampleNonInteractions(int time) {
        HashSet<String> interactions = new HashSet<>();
        ArrayList<Integer> nonInteractions = new ArrayList<Integer>();

        for (int qId : m_testInteractions.keySet()) {
            String uiId = m_questionMap.get(qId).getUserID();
            int i = m_userIDIndex.get(uiId);
            interactions = m_testInteractions.get(qId);
            nonInteractions.clear();

            for (int j = 0; j < m_users.size(); j++) {
                if (i == j) continue;
                if (interactions.contains(j)) continue;
                nonInteractions.add(j);
            }

            int number = time * m_testInteractions.get(qId).size();
            m_testNonInteractions.put(qId, sampleNonInteractions(nonInteractions, number));
        }
    }

    public void saveInteractions(String prefix){
        try{
            String filename = prefix + "Interactions4Recommendations_test.txt";
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int qId: m_testInteractions.keySet()){
                writer.write(qId + "\t");
                for(String uj: m_testInteractions.get(qId)){
                    writer.write(uj + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' interactions are written in %s.\n", m_testInteractions.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void saveNonInteractions(String prefix, int time){
        try{
            String filename = String.format("%sNonInteractions_time_%d_Recommendations.txt", prefix, time);
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int qId: m_testNonInteractions.keySet()){
                writer.write(qId + "\t");
                for(int nonIdx: m_testNonInteractions.get(qId)){
                    String nonId = m_users.get(nonIdx).getUserID();
                    writer.write(nonId + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' non-interactions are written in %s.\n", m_testNonInteractions.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public boolean containsQuestion(_User u){
        for(_Review r: u.getReviews()){
            if(r.getParentId() == -1)
                return true;
        }
        return false;
    }

    // we assign cv index for all the reviews in the corpus
    public void assignCVIndex4AnswerRecommendation(){
        int unseen = 0, seen = 0;
        for(_Doc d: m_corpus.getCollection()){
            _Review r = (_Review) d;
            if(m_selectedAnswers.contains(r.getPostId())){
                r.setMask4CV(0);
                unseen++;
            } else{
                r.setMask4CV(1);
                seen++;
            }
        }
        System.out.format("Train doc size: %d, test doc size: %d\n", seen, unseen);
    }

    public void printSelectedQuestionIds(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int qId: m_selectedQuestions){
                String uId = m_questionMap.get(qId).getUserID();
                writer.format("%s\t%d\n", uId, qId);
            }
            writer.close();
            System.out.format("Finish writing %d selected questions!\n", m_selectedQuestions.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {
        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "StackOverflow"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

        int time = 10;
        // load the interaction, remove the connections built based on the selected answers
        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
        String friendFile4Recommendation = String.format("%s/%s/%sFriends4Recommendation.txt", prefix, dataset, dataset);
        String questionFile = String.format("%s/%s/%sSelectedQuestions.txt", prefix, dataset, dataset);
        String prefix4Rec = String.format("%s/%s/%s", prefix, dataset, dataset);

//        MultiThreadedStackOverflowAnalyzer analyzer = new MultiThreadedStackOverflowAnalyzer(tokenModel, classNumber, providedCV,
//                Ngram, lengthThreshold, numberOfCores, true);
//
//        analyzer.setAllocateReviewFlag(false); // do not allocate reviews
//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.buildQuestionAnswerMap();
//        analyzer.loadInteractions(friendFile);
//        analyzer.selectQuestions4Recommendation();
//
//        // assign cv index for training and testing documents
        String cvIndexFile = String.format("%s/%s/%sCVIndex4Recommendation.txt", prefix, dataset, dataset);
//        analyzer.assignCVIndex4AnswerRecommendation();
//        analyzer.saveCVIndex(cvIndexFile);
//
//        analyzer.refineNetwork4Recommendation(time, prefix4Rec);
//        analyzer.saveNetwork(friendFile4Recommendation);
//        analyzer.printSelectedQuestionIds(questionFile);

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        String mode = "cv4doc"; // "cv4edge" "cs4doc"--"cold start for doc" "cs4edge"--"cold start for edge"

        int kFold = 5, k = 0;

        //if it is cv for doc, use all the interactions + part of docs
        analyzer.loadCVIndex(cvIndexFile, kFold);
        analyzer.loadInteractions(friendFile4Recommendation);

        _Corpus corpus = analyzer.getCorpus();

        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 50, number_of_topics = 20, varMaxIter = 10, embeddingDim = 10, trainIter = 1, testIter = 1500;
        //these two parameters must be larger than 1!!!
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6, stepSize = 0.001;
        boolean alphaFlag = true, gammaFlag = false, betaFlag = true, tauFlag = false, xiFlag = true, rhoFlag = false;
        boolean multiFlag = true, adaFlag = false;

        long start = System.currentTimeMillis();

        EUB_multithreading tModel = new EUB_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);

        tModel.initLookupTables(analyzer.getUsers());
        tModel.setModelParamsUpdateFlags(alphaFlag, gammaFlag, betaFlag, tauFlag, xiFlag, rhoFlag);
        tModel.setMode(mode);

        tModel.setTrainInferMaxIter(trainIter);
        tModel.setTestInferMaxIter(testIter);
        tModel.setStepSize(stepSize);

        long current = System.currentTimeMillis();
        String saveDir = String.format("./data/embeddingExp/eub/%s_emIter_%d_nuTopics_%d_varIter_%d_trainIter_%d_testIter_%d_dim_%d_ada_%b/" +
                    "fold_%d_%d", dataset, emMaxIter, number_of_topics, varMaxIter, trainIter, testIter, embeddingDim, adaFlag, k, current);

        tModel.fixedCrossValidation(k, saveDir);
//        tModel.printGamma("./data/embeddingExp/EUB");
//        tModel.printBeta("./data/embeddingExp/EUB");
        long end = System.currentTimeMillis();
        System.out.println("\n[Info]Start time: " + start);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/((1000*60*60) * 1.0);
        System.out.print(String.format("[Time]This training+testing process took %.4f hours.\n", hours));
    }
}
