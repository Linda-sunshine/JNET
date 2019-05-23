package topicmodels.UserEmbedding;

import Jama.Matrix;
import structures.*;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

import java.io.*;
import java.util.*;


/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The joint modeling of user embedding (U*M) and topic embedding (K*M)
 */

public class EUB extends LDA_Variational {

    public enum modelType {
        CV4DOC, // cross validation for testing document perplexity
        CV4EDGE, // cross validation for testing link prediction
    }

    // Default is CV4Doc
    protected modelType m_mType = modelType.CV4DOC;

    protected _Topic4EUB[] m_topics;
    protected ArrayList<_User4EUB> m_users;
    protected ArrayList<_Doc4EUB> m_docs;

    protected int m_embeddingDim;

    protected HashMap<String, Integer> m_usersIndex;
    // key: user index, value: document index array
    protected HashMap<Integer, ArrayList<Integer>> m_userDocMap;
    // key: user index, value: interaction indexes
    protected HashMap<Integer, HashSet<Integer>> m_networkMap;

    // key: doc index, value: user index
    protected HashMap<Integer, Integer> m_docUserMap;

    /*****variational parameters*****/
    protected double t_mu = 0.1, t_sigma = 1;
    protected double d_mu = 0.1, d_sigma = 1;
    protected double u_mu = 0.1, u_sigma = 1;

    /*****model parameters*****/
    // this alpha is differnet from alpha in LDA
    // alpha is precision parameter for topic embedding in EUB
    // alpha is a vector parameter for dirichlet distribution
    protected double m_alpha_s = 1;
    protected double m_tau = 1;
    protected double m_gamma = 1;
    protected double m_xi = 2.0;

    /*****Sparsity parameter******/
    static public double m_rho = 1;

    protected int m_displayLv = 0;
    protected double m_stepSize = 1e-3;

    protected boolean m_alphaFlag = false;
    protected boolean m_gammaFlag = false;
    protected boolean m_betaFlag = true;
    protected boolean m_tauFlag = false;
    protected boolean m_xiFlag = false;
    protected boolean m_rhoFlag = false;

    // whehter we use adam grad to optimize or not
    protected boolean m_adaFlag = false;

    protected int m_trainInferMaxIter = 1;
    protected int m_paramMaxIter = 20;
    protected int m_testInferMaxIter = 1500;

    protected String m_data = "YelpNew";

    public EUB(int number_of_iteration, double converge, double beta,
               _Corpus c, double lambda, int number_of_topics, double alpha,
               int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);

        m_embeddingDim = m;
        m_topics = new _Topic4EUB[number_of_topics];
        m_users = new ArrayList<_User4EUB>();
        m_docs = new ArrayList<_Doc4EUB>();
        m_networkMap = new HashMap<Integer, HashSet<Integer>>();

        m_trainSet = new ArrayList<>();
        m_testSet = new ArrayList<>();
    }

    public String toString() {
        return String.format("[JNET]Mode: %s, Dim: %d, Topic number: %d, EmIter: %d, VarIter: %d, TrainInferIter: %d, TestInferIter: %d, ParamIter: %d.",
                m_mType.toString(), m_embeddingDim, number_of_topics, number_of_iteration, m_varMaxIter, m_trainInferMaxIter, m_testInferMaxIter, m_paramMaxIter);
    }

    public void setModelParamsUpdateFlags(boolean alphaFlag, boolean gammaFlag, boolean betaFlag,
                                          boolean tauFlag, boolean xiFlag, boolean rhoFlag) {
        m_alphaFlag = alphaFlag;
        m_gammaFlag = gammaFlag;
        m_betaFlag = betaFlag;
        m_tauFlag = tauFlag;
        m_xiFlag = xiFlag;
        m_rhoFlag = rhoFlag;
    }

    public void setMode(String mode) {
        if (mode.equals("cv4edge")) {
            m_mType = modelType.CV4EDGE;
        } else if (mode.equals("cv4doc")) {
            m_mType = modelType.CV4DOC;
        }
    }

    public void setAdaFlag(boolean b) {
        m_adaFlag = b;
    }
    public void setData(String data){
        m_data = data;
    }

    public void setGamma(double g){
        m_gamma = g;
    }
    public void setStepSize(double s) {
        if (s > 0)
            m_stepSize = s;
        else {
            System.err.format("[Error]Step size has to be positive! Set to the default value 1e-3...\n");
            m_stepSize = 1e-3;
        }

    }

    public void setTrainInferMaxIter(int it) {
        if (it > 0)
            m_trainInferMaxIter = it;
        else {
            System.err.format("[Error]Maximum iteration for inference iteration has to be positive! Set to the default value 1...\n");
            m_trainInferMaxIter = 1;
        }
    }

    public void setTestInferMaxIter(int it) {
        if (it > 0)
            m_testInferMaxIter = it;
        else {
            System.err.format("[Error]Maximum iteration for inference iteration has to be positive! Set to the default value 1...\n");
            m_testInferMaxIter = 1;
        }
    }

    public void setParamMaxIter(int it) {
        m_paramMaxIter = it;
    }

    // Load the data for later user
    public void initLookupTables(ArrayList<_User> users) {
        m_usersIndex = new HashMap<String, Integer>();
        m_userDocMap = new HashMap<Integer, ArrayList<Integer>>();
        m_docUserMap = new HashMap<Integer, Integer>();

        // build the lookup table for user/doc/topic
        for (int i = 0; i < users.size(); i++) {
            _User4EUB user = new _User4EUB(users.get(i));
            m_users.add(user);
            m_usersIndex.put(user.getUserID(), i);

            initUserDocs(i, user, users.get(i).getReviews());
        }

        for (int k = 0; k < number_of_topics; k++)
            m_topics[k] = new _Topic4EUB(k);
    }

    protected void initUserDocs(int i, _User4EUB user, ArrayList<_Review> reviews) {
        ArrayList<_Doc4EUB> docs = new ArrayList<_Doc4EUB>();
        for (_Review r : reviews) {
            int dIndex = m_docs.size();
            _Doc4EUB doc = new _Doc4EUB(r, dIndex);
            docs.add(doc);
            m_docs.add(doc);
            m_docUserMap.put(dIndex, i);
        }
        user.setReviews(docs);
    }

    public void loadTopicTermProbability(String betaFilename) {

        try {
            // load beta for the whole corpus first
            File betaFile = new File(betaFilename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(betaFile), "UTF-8"));
            String line = reader.readLine();
            String[] strs = line.split("\t");
            int nuTopics = Integer.valueOf(strs[0]);
            int vocabSize = Integer.valueOf(strs[1]);
            if (nuTopics != number_of_topics || vocabSize != vocabulary_size) {
                System.out.println("Wrong input files for EUB!");
            }
            int count = 0;
            double[][] topic_term_probability = new double[number_of_topics][vocabulary_size];
            while ((line = reader.readLine()) != null) {
                strs = line.trim().split("\t");
                if (strs.length != vocabulary_size)
                    System.out.println("Wrong vocabulary size!!");
                double[] oneTopic = new double[vocabulary_size];
                for (int i = 0; i < strs.length; i++) {
                    oneTopic[i] = Double.valueOf(strs[i]);
                }
                topic_term_probability[count++] = oneTopic;
            }
            // assign the topics to the model
            this.topic_term_probabilty = topic_term_probability;
            System.out.format("[Info]Finish loading beta from %s\n", betaFilename);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void loadDocsPhi(String phiFilename) {

        try {
            // load beta for the whole corpus first
            File betaFile = new File(phiFilename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(betaFile), "UTF-8"));
            String line;
            String[] strs;
            while ((line = reader.readLine()) != null) {
                // start reading one user's data
                if (line.equals("-----")) {
                    strs = reader.readLine().trim().split("\t");
                    if (strs.length != 4)
                        System.out.println("[error]Wrong format!!!");
                    String uid = strs[0];
                    int id = Integer.valueOf(strs[1]);
                    int fvSize = Integer.valueOf(strs[2]);
                    int nuTopics = Integer.valueOf(strs[3]);
                    // read phi
                    double[][] phi = new double[fvSize][nuTopics];
                    for (int i = 0; i < fvSize; i++) {
                        strs = reader.readLine().trim().split("\t");
                        if (strs.length != number_of_topics) {
                            System.out.println("[error]Wrong dimension for the phi!");
                        }
                        double[] oneFv = new double[number_of_topics];
                        for (int j = 0; j < strs.length; j++) {
                            oneFv[j] = Double.valueOf(strs[j]);
                        }
                        phi[i] = oneFv;
                    }
                    _Doc4EUB r = (_Doc4EUB) m_users.get(m_usersIndex.get(uid)).getReviewByID(id);
                    r.setPhi(phi);
                }
            }
            System.out.format("[Info]Finish loading beta from %s\n", phiFilename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // key: user index, value: the
    protected void buildUserDocMap() {
        m_userDocMap.clear();

        for (int i = 0; i < m_users.size(); i++) {
            ArrayList<_Review> reviews = m_users.get(i).getReviews();
            for (_Review r : reviews) {
                if (r.getType() == _Doc.rType.TRAIN) {
                    if (!m_userDocMap.containsKey(i))
                        m_userDocMap.put(i, new ArrayList<Integer>());
                    _Doc4EUB doc = (_Doc4EUB) r;
                    m_userDocMap.get(i).add(doc.getIndex());
                }
            }
        }
    }

    public void setDisplayLv(int d) {
        m_displayLv = d;
    }

    @Override
    public void EMonCorpus() {
        constructNetwork();
        m_trainSet = new ArrayList<_Doc>();
        // collect all training reviews
        for (_User u : m_users) {
            for (_Review r : u.getReviews()) {
                m_trainSet.add(r);
            }
        }
        EM();
    }

    @Override
    public void EM() {

        // sample non-interactions first before E-step
        initialize_probability(m_trainSet);

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge;
        do {
            System.out.format(String.format("\n----------Start EM %d iteraction----------\n", iter));


            if (m_multithread)
                currentAllLikelihood = multithread_E_step();
            else
                currentAllLikelihood = E_step();

            System.out.format("[Info]Finish E-step: loglikelihood is: %.5f.\n", currentAllLikelihood);
            if (iter > 0)
                converge = Math.abs((lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood);
            else
                converge = 1.0;

            calculate_M_step(++iter);

            if (iter % 10 == 0) {
                printTopWords(30);
            }

            lastAllLikelihood = currentAllLikelihood;
        } while (iter < number_of_iteration && converge > m_converge);
    }

    // put the user interaction and non-interaction information in the four hashmaps
    protected void constructNetwork() {
        System.out.print("[Info]Construct network with interactions and non-interactions....");
        for (String uiId : m_usersIndex.keySet()) {
            int uiIdx = m_usersIndex.get(uiId);
            // interactions
            _User4EUB ui = m_users.get(uiIdx);
            if (ui.getFriends() != null && ui.getFriends().length > 0) {
                if (!m_networkMap.containsKey(uiIdx))
                    m_networkMap.put(uiIdx, new HashSet<Integer>());
                for (String ujId : ui.getFriends()) {
                    int ujIdx = m_usersIndex.get(ujId);
                    if (!m_networkMap.containsKey(ujIdx))
                        m_networkMap.put(ujIdx, new HashSet<Integer>());
                    m_networkMap.get(uiIdx).add(ujIdx);
                    m_networkMap.get(ujIdx).add(uiIdx);
                }
            }
        }

        System.out.print("Finish network construction!");
    }

    @Override
    protected void initialize_probability(Collection<_Doc> docs) {
        System.out.println("[Info]Initializing topics, documents and users...");
        for (_Topic4EUB t : m_topics)
            t.setTopics4Variational(m_embeddingDim, t_mu, t_sigma);

        for (_Doc d : m_trainSet) {
            ((_Doc4EUB) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma);
        }

        for (_User4EUB u : m_users)
            u.setTopics4Variational(m_embeddingDim, m_users.size(), u_mu, u_sigma);

        init();
        for (_Doc doc : m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        calculate_M_step(0);
    }

    // Update the stats related with doc
    protected void updateStats4Doc(_Doc4EUB doc) {
        _SparseFeature[] fv = doc.getSparse();
        int wid;
        double v;
        for (int n = 0; n < fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for (int i = 0; i < number_of_topics; i++)
                word_topic_sstat[i][wid] += v * doc.m_phi[n][i];
        }
    }

    // update variational parameters of latent variables
    protected double E_step() {
        int iter = 0;
        double last = -1.0, converge;

        init();
        do {
            double docLikelihood = 0.0, userLikelihood = 0.0, topicLikelihood = 0.0;

            for (_Doc doc : m_trainSet) {
                docLikelihood += varInference4Doc((_Doc4EUB) doc);
                if (Double.isNaN(docLikelihood) || Double.isInfinite(docLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }

            for (_Topic4EUB topic : m_topics) {
                topicLikelihood += varInference4Topic(topic);
                if (Double.isNaN(topicLikelihood) || Double.isInfinite(topicLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }

            for (_User4EUB user : m_users) {
                userLikelihood += varInference4User(user);
                if (Double.isNaN(userLikelihood) || Double.isInfinite(userLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }

            if (iter > 0)
                converge = Math.abs((docLikelihood + topicLikelihood + userLikelihood - last) / last);
            else
                converge = 1.0;

            last = docLikelihood + topicLikelihood + userLikelihood;
            System.out.format("[Multi-E-step] %d iteration, likelihood(d:t:u)=(%.2f, %.2f, %.2f), converge to %.8f\n",
                    iter, docLikelihood, topicLikelihood, userLikelihood, converge);

        } while (iter++ < m_varMaxIter && converge > m_varConverge);

        for (_Doc doc : m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        return last;
    }

    @Override
    public void calculate_M_step(int iter) {
        if (iter>0) {
            if (m_alphaFlag)
                est_alpha();// precision for topic embedding
            if (m_gammaFlag)
                est_gamma(); // precision for user embedding
            if (m_betaFlag)
                est_beta(); // topic-word distribution
            if (m_tauFlag)
                est_tau(); // precision for topic proportion
            if (m_xiFlag)
                est_xi(); // sigma for the affinity \delta_{ij}
            if (m_rhoFlag)
                est_rho();
        }

        System.out.format("[ModelParam]alpha:%.3f, gamma:%.3f,tau:%.3f,xi:%.3f, rho:%.5f\n",
                m_alpha_s, m_gamma, m_tau, m_xi, m_rho);

        finalEst();
    }

    protected void est_alpha() {
        System.out.println("[M-step]Estimate alpha....");
        double denominator = 0;
        for (int k = 0; k < number_of_topics; k++) {
            _Topic4EUB topic = m_topics[k];
            denominator += sumSigmaDiagAddMuTransposeMu(topic.m_sigma_phi, topic.m_mu_phi);
        }
        if (denominator>0)
            m_alpha_s = number_of_topics * m_embeddingDim / denominator;
        //otherwise keep the current value
    }

    protected void est_gamma() {
        System.out.println("[M-step]Estimate gamma....");
        double denominator = 0;
        for (int uIndex : m_userDocMap.keySet()) {
            _User4EUB user = m_users.get(uIndex);
            denominator += sumSigmaDiagAddMuTransposeMu(user.m_sigma_u, user.m_mu_u);
        }

        if (denominator>0)
            m_gamma = m_users.size() * m_embeddingDim / denominator;
        //otherwise keep the current value
    }

    protected void est_beta() {
        System.out.println("[M-step]Estimate beta....");
        for (int k = 0; k < number_of_topics; k++) {
            double sum = Utils.sumOfArray(word_topic_sstat[k]);
            for (int v = 0; v < vocabulary_size; v++) //will be in the log scale!!
                topic_term_probabilty[k][v] = Math.log(word_topic_sstat[k][v] / sum);
        }
    }

    protected void est_tau() {
        System.out.println("[M-step]Estimate tau....");
        double denominator = 0;
        double D = m_trainSet.size(); // total number of documents
        for (_Doc d : m_trainSet) {
            _Doc4EUB doc = (_Doc4EUB) d;
            int uIndex = m_docUserMap.get(doc.getIndex());
            int uIdx = m_usersIndex.get(doc.getUserID());
            if (uIndex != uIdx)
                System.out.println("Error!!!");
            _User4EUB user = m_users.get(uIndex);
            denominator += calculateStat4Tau(user, doc);
        }

        if (denominator>0)
            m_tau = D * number_of_topics / denominator;
        //otherwise keep the current value
    }

    protected void est_xi() {
        System.out.println("[M-step]Estimate xi....");
        double xiSquare = 0;
        // sum over all interactions and non-interactions
        // i->p and p'
        for (int i = 0; i < m_users.size(); i++) {
            _User4EUB ui = m_users.get(i);

            for (int j = 0; j < m_users.size(); j++) {
                if (i == j) continue;
                xiSquare += calculateStat4Xi(ui, m_users.get(j), j);
            }
        }

        if (xiSquare>0) {
            double totalConnection = m_users.size() * (m_users.size() - 1);
            m_xi = Math.sqrt(xiSquare/totalConnection);
        }
        //otherwise keep the current value
    }

    protected void est_rho() {
        System.out.println("[M-step]Estimate rho....");
        double numerator = 0, denominator = 0;
        int eij;

        for (int i = 0; i < m_users.size(); i++) {
            _User4EUB ui = m_users.get(i);
            HashSet<Integer> interactions = m_networkMap.containsKey(i) ? m_networkMap.get(i) : null;
            for (int j = 0; j < m_users.size(); j++) {
                if (i == j) continue;
                eij = interactions != null && interactions.contains(j) ? 1 : 0;
                numerator += eij;
                denominator += (1 - eij) * Math.exp(ui.m_mu_delta[j] + 0.5 * ui.m_sigma_delta[j] *
                        ui.m_sigma_delta[j]) / ui.m_epsilon_prime[j];
            }
        }
        m_rho = numerator / denominator;
    }

    protected double calculateStat4Xi(_User4EUB ui, _User4EUB uj, int j) {
        double val = ui.m_mu_delta[j] * ui.m_mu_delta[j] + ui.m_sigma_delta[j] * ui.m_sigma_delta[j];
        for (int m = 0; m < m_embeddingDim; m++) {
            val -= 2 * ui.m_mu_delta[j] * ui.m_mu_u[m] * uj.m_mu_u[m];
            for (int l = 0; l < m_embeddingDim; l++) {
                val += (ui.m_sigma_u[m][l] + ui.m_mu_u[m] * uj.m_mu_u[l]) *
                        (uj.m_sigma_u[m][l] + uj.m_mu_u[m] * uj.m_mu_u[l]);
            }
        }
        return val;
    }

    protected double calculateStat4Tau(_User4EUB user, _Doc4EUB doc) {
        double term1 = 0, term2 = 0, term3 = 0;

        for (int k = 0; k < number_of_topics; k++) {
            _Topic4EUB topic = m_topics[k];
            term1 += doc.m_sigma_theta[k] + doc.m_mu_theta[k] * doc.m_mu_theta[k];
            for (int m = 0; m < m_embeddingDim; m++) {
                term2 += doc.m_mu_theta[k] * topic.m_mu_phi[m] * user.m_mu_u[m];
                for (int l = 0; l < m_embeddingDim; l++) {
                    term3 += (user.m_sigma_u[m][l] + user.m_mu_u[m] * user.m_mu_u[l])
                            * (topic.m_sigma_phi[m][l] + topic.m_mu_phi[m] * topic.m_mu_phi[l]);
                }
            }
        }
        return term1 - 2*term2 + term3;
    }

    protected double varInference4Topic(_Topic4EUB topic) {
        // update the mu and sigma for each topic \phi_k -- Eq(65) and Eq(67)
        update_phi_k(topic);

        return calc_log_likelihood_per_topic(topic);
    }

    protected double varInference4User(_User4EUB user) {
        double curLoglikelihood = 0.0, lastLoglikelihood = 1.0, converge = 0.0;
        int iter = 0;

        do {

            // update the variational parameters for user embedding ui -- Eq(70) and Eq(72)
            update_u_i(user);
            // update the mean and variance for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(75)
            update_delta_ij_mu(user);
            update_delta_ij_sigma(user);
            // update the taylor parameter epsilon
            update_epsilon(user);
            // update the taylor parameter epsilon_prime
            update_epsilon_prime(user);

            curLoglikelihood = calc_log_likelihood_per_user(user);

            if (Double.isNaN(curLoglikelihood) || Double.isInfinite(curLoglikelihood))
                break;

            if (iter > 0)
                converge = (lastLoglikelihood - curLoglikelihood) / lastLoglikelihood;
            else
                converge = 1.0;

            lastLoglikelihood = curLoglikelihood;
        } while (++iter < m_trainInferMaxIter && Math.abs(converge) > m_varConverge);

        return lastLoglikelihood;
    }

    protected double varInference4Doc(_Doc4EUB doc) {
        int maxIter = (doc.getType() == _Doc.rType.TEST) ? m_testInferMaxIter : m_trainInferMaxIter;
        double curLoglikelihood = 0.0, lastLoglikelihood = 1.0, converge = 0.0;
        int iter = 0;
        boolean warning;

        do {

            // variational parameters for word indicator z_{idn}
            update_eta_id(doc);

            // variational parameters for topic distribution \theta_{id}
            update_theta_id_mu(doc);
            update_zeta(doc);
            update_theta_id_sigma(doc);
            update_zeta(doc);

            curLoglikelihood = calc_log_likelihood_per_doc(doc);

            if (Double.isNaN(curLoglikelihood) || Double.isInfinite(curLoglikelihood))
                break;

            if (iter > 0)
                converge = (lastLoglikelihood - curLoglikelihood) / lastLoglikelihood;
            else
                converge = 1.0;

            lastLoglikelihood = curLoglikelihood;
        } while (++iter < maxIter && Math.abs(converge) > m_varConverge);

        return curLoglikelihood;
    }

    // update the mu and sigma for each topic \phi_k
    protected void update_phi_k(_Topic4EUB topic){

        double[][] CovMat = new double[m_embeddingDim][m_embeddingDim];
        double[] term = new double[m_embeddingDim];

        int tIndex = topic.getIndex();
        for(int i=0; i<m_embeddingDim; i++)
            CovMat[i][i] = m_alpha_s;

        // \tau * \sum_u|sum_d(\sigma + \mu * \mu^T)
        for(int uIndex: m_userDocMap.keySet()){
            _User4EUB user = m_users.get(uIndex);
            // \sigma + mu * mu^T
            for(int dIndex: m_userDocMap.get(uIndex)){
                _Doc4EUB doc = m_docs.get(dIndex);
                Utils.add2Array(term, user.m_mu_u, m_tau * doc.m_mu_theta[tIndex]);
            }
            sigmaAddMuMuTranspose(CovMat, user.m_sigma_u, user.m_mu_u, m_tau * m_userDocMap.get(uIndex).size());
        }

        Matrix invsMtx = (new Matrix(CovMat)).inverse();

        topic.m_sigma_phi = invsMtx.getArray();
        topic.m_mu_phi = Utils.matrixMultVector(topic.m_sigma_phi, term);
    }
    // \sum_{mm}\simga[m][m] + \mu^T * \mu
    protected double sumSigmaDiagAddMuTransposeMu(double[][] sigma, double[] mu){
        if(sigma.length != mu.length)
            return 0;
        int dim = mu.length;
        double sum = 0;
        for(int m=0; m<dim; m++)
            sum += sigma[m][m] + mu[m] * mu[m];
        return sum;
    }

    // \sum_{mm}\simga[m][m] + \mu^T * \mu
    protected double sumSigmaDiagAddMuTransposeMu(double[] sigma, double[] mu){
        if(sigma.length != mu.length)
            return 0;
        int dim = mu.length;
        double sum = 0;
        for(int m=0; m<dim; m++){
            sum += sigma[m] + mu[m] * mu[m];
        }
        return sum;
    }

    // \simga + \mu * \mu^T
    protected double[][] sigmaAddMuMuTranspose(double[][] sigma, double[] mu){
        int dim = mu.length;
        double[][] res = new double[dim][dim];
        for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
                res[i][j] = sigma[i][j] + mu[i]*mu[j];
            }
        }
        return res;
    }

    //added by Hongning Wang
    protected void sigmaAddMuMuTranspose(double[][] res, double[][] sigma, double[] mu, double weight){
        int dim = mu.length;
        for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
                res[i][j] += weight*(sigma[i][j] + mu[i]*mu[j]);
            }
        }
    }

    // update the variational parameters for user embedding ui -- Eq(44) and Eq(45)
    protected void update_u_i(_User4EUB ui) {
        // termT: the part related with user's topics
        double[][] sigma_termT = new double[m_embeddingDim][m_embeddingDim];
        double[] mu_termT = new double[m_embeddingDim];

        int i = m_usersIndex.get(ui.getUserID());
        double xiSqRecp = 1.0/m_xi /m_xi;

        //associated documents
        if(m_userDocMap.containsKey(i)){
            ArrayList<Integer> uDocs = m_userDocMap.get(i);
            double tauD = m_tau * uDocs.size();
            for(int k=0; k<number_of_topics; k++){
                // stat for updating sigma
                _Topic4EUB topic = m_topics[k];
                sigmaAddMuMuTranspose(sigma_termT, topic.m_sigma_phi, topic.m_mu_phi, tauD);

                // stat for updating mu
                for(int dIndex: uDocs){
                    _Doc4EUB doc = m_docs.get(dIndex);
                    Utils.add2Array(mu_termT, topic.m_mu_phi, m_tau * doc.m_mu_theta[k]);
                }
            }
        }

        // \gamma * I
        for(int a=0; a<m_embeddingDim; a++)
            sigma_termT[a][a] += m_gamma;

        //associated edges
        for(int j=0; j<m_users.size(); j++){
            if(j == i)
                continue;

            _User4EUB uj = m_users.get(j);
            sigmaAddMuMuTranspose(sigma_termT, uj.m_sigma_u, uj.m_mu_u, xiSqRecp);
            Utils.add2Array(mu_termT, uj.m_mu_u, ui.m_mu_delta[j]*xiSqRecp);
        }

        Matrix invsMtx = (new Matrix(sigma_termT)).inverse();
        ui.m_sigma_u = invsMtx.getArray();
        ui.m_mu_u = Utils.matrixMultVector(ui.m_sigma_u, mu_termT);
    }

    // update mean for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(47)
    protected void update_delta_ij_mu(_User4EUB ui) {
        int i = m_usersIndex.get(ui.getUserID());

        double muG = 0, muH[] = new double[m_users.size()], xiSqRecp = 1.0/m_xi /m_xi;


        double fValue, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 30, iter = 0;
        HashSet<Integer> interactions = m_networkMap.containsKey(i) ? m_networkMap.get(i) : null;

        do {
            fValue = 0;

            //this has to be an undirected graph

            for(int j=0; j<m_users.size(); j++){
                if(i == j)
                    continue;

                int eij = (interactions != null && interactions.contains(j)) ? 1 : 0;
                double dotProd = Utils.dotProduct(ui.m_mu_u, m_users.get(j).m_mu_u);
                double term = ((1-eij)*(1-m_rho)/ui.m_epsilon_prime[j] - 1/ui.m_epsilon[j]) *
                        Math.exp(ui.m_mu_delta[j] + 0.5 * ui.m_sigma_delta[j] * ui.m_sigma_delta[j]);

                fValue += eij == 1 ? ui.m_mu_delta[j] : 0 + term
                        -0.5*xiSqRecp * ui.m_mu_delta[j] * (ui.m_mu_delta[j] - 2 * dotProd);
                muG = eij + term - xiSqRecp * (ui.m_mu_delta[j] - dotProd);

                if(m_adaFlag)
                    ui.m_mu_delta[j] += m_stepSize/Math.sqrt(muH[j]) * muG;
                else
                    ui.m_mu_delta[j] += m_stepSize * muG;
                muH[j] += muG * muG;
            }

            printFValue(lastFValue, fValue);
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;

        } while(iter++ < iterMax && Math.abs(diff) > cvg);
        if(m_displayLv != 0)
            System.out.println("------------------------");
    }

    protected double[] calcFGValueDeltaMu(_User4EUB ui, double[] mu_delta, int eij, int j) {
        double[] sigma_delta = ui.m_sigma_delta;
        double dotProd = Utils.dotProduct(ui.m_mu_u, m_users.get(j).m_mu_u);

        double fValue = eij == 1 ? mu_delta[j] : 0;
        double gValue = eij;
        double term1 = ((1 - eij) * (1 - m_rho) / ui.m_epsilon_prime[j] - 1 / ui.m_epsilon[j]) *
                Math.exp(mu_delta[j] + 0.5 * sigma_delta[j] * sigma_delta[j]);

        fValue += term1 - 0.5 / m_xi / m_xi * (mu_delta[j] * mu_delta[j] - 2 * mu_delta[j] * dotProd);
        gValue += term1 - 1 / m_xi / m_xi * (mu_delta[j] - dotProd);
        return new double[]{fValue, gValue};
    }
    // update variance for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(48)
    protected void update_delta_ij_sigma(_User4EUB ui){
        int i = m_usersIndex.get(ui.getUserID());

        double fValue, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 30, iter = 0;
        double sigmaG, sigmaH[] = new double[m_users.size()], xiSqRecp = 1.0/m_xi /m_xi;
        HashSet<Integer> interactions = m_networkMap.containsKey(i) ? m_networkMap.get(i) : null;

        do {
            fValue = 0;
            for(int j=0; j<m_users.size(); j++){
                if(i == j)
                    continue;

                int eij = interactions != null && interactions.contains(j) ? 1 : 0;
                double term = ((1-eij)*(1-m_rho)/ui.m_epsilon_prime[j] - 1/ui.m_epsilon[j]) *
                        Math.exp(ui.m_mu_delta[j] + 0.5 * ui.m_sigma_delta[j] * ui.m_sigma_delta[j]);

                fValue += term - 0.5 * xiSqRecp * (ui.m_sigma_delta[j] * ui.m_sigma_delta[j]) + Math.log(Math.abs(ui.m_sigma_delta[j]));
                sigmaG = ui.m_sigma_delta[j] * (term - xiSqRecp) + 1.0 / ui.m_sigma_delta[j];

                if(m_adaFlag)
                    ui.m_sigma_delta[j] += m_stepSize/Math.sqrt(sigmaH[j]) * sigmaG;
                else
                    ui.m_sigma_delta[j] += m_stepSize * sigmaG;

                sigmaH[j] += sigmaG * sigmaG;

                if(Double.isNaN(ui.m_sigma_delta[j]) || Double.isInfinite(ui.m_sigma_delta[j]))
                    System.out.println("[error] Sigma_delta is Nan or Infinity!!");
            }
            printFValue(lastFValue, fValue);
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;

        } while(iter++ < iterMax && Math.abs(diff) > cvg);
        if(m_displayLv != 0)
            System.out.println("------------------------");
    }

    protected double[] calcFGValueDeltaSigma(_User4EUB ui, double[] sigma, int eij, int j) {

        double term1 = ((1 - eij) * (1 - m_rho) / ui.m_epsilon_prime[j] - 1 / ui.m_epsilon[j])
                * Math.exp(ui.m_mu_delta[j] + 0.5 * sigma[j] * sigma[j]);
        double fValue = term1 - 0.5 / m_xi / m_xi * (sigma[j] * sigma[j]) + Math.log(Math.abs(sigma[j]));
        double gValue = sigma[j] * term1 - sigma[j] / m_xi / m_xi + 1 / sigma[j];
        return new double[]{fValue, gValue};
    }

    // update the taylor parameter epsilon
    protected void update_epsilon(_User4EUB ui) {
        int i = m_usersIndex.get(ui.getUserID());
        for (int j = 0; j < ui.m_epsilon.length; j++) {
            if (j != i)
                ui.m_epsilon[j] = Math.exp(ui.m_mu_delta[j] + 0.5 * ui.m_sigma_delta[j] * ui.m_sigma_delta[j]) + 1;
        }
    }

    // update the taylor parameter epsilon_prime
    protected void update_epsilon_prime(_User4EUB ui) {
        int i = m_usersIndex.get(ui.getUserID());
        for (int j = 0; j < ui.m_epsilon_prime.length; j++) {
            if (j != i)
                ui.m_epsilon_prime[j] = (1 - m_rho) * Math.exp(ui.m_mu_delta[j] +
                        0.5 * ui.m_sigma_delta[j] * ui.m_sigma_delta[j]) + 1;
        }
    }

    protected void update_eta_id(_Doc4EUB doc) {
        double logSum;
        _SparseFeature[] fvs = doc.getSparse();
        Arrays.fill(doc.m_sstat, 0);

        for (int n = 0; n < fvs.length; n++) {
            int v = fvs[n].getIndex();
            for (int k = 0; k < number_of_topics; k++) {
                // Eq(57) update eta of each document
                doc.m_phi[n][k] = doc.m_mu_theta[k] + topic_term_probabilty[k][v];
            }

            // normalize
            logSum = Utils.logSum(doc.m_phi[n]);
            for (int k = 0; k < number_of_topics; k++) {
                doc.m_phi[n][k] = Math.exp(doc.m_phi[n][k] - logSum);
                // update \sum_\eta_{vk}, only related with topic index k
                doc.m_sstat[k] += fvs[n].getValue() * doc.m_phi[n][k];
            }
        }
    }
    // variational parameters for topic distribution \theta_{id}
    protected void update_theta_id_mu(_Doc4EUB doc){
        // user index of the current doc
        int i = m_docUserMap.get(doc.getIndex());
        int N = doc.getTotalDocLength();

        double fValue, dotProd, moment;
        double lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 30, iter = 0;
        double mu_u[] = m_users.get(i).m_mu_u;
        double muG, muH[] = new double[number_of_topics];

        do{
            fValue = 0;
            for(int k=0; k<number_of_topics; k++){
                // function value
                moment = N * Math.exp(doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k] - doc.m_logZeta);
                fValue += -0.5 * m_tau * doc.m_mu_theta[k] * doc.m_mu_theta[k];
                dotProd = Utils.dotProduct(m_topics[k].m_mu_phi, mu_u);
                fValue += m_tau * doc.m_mu_theta[k] * dotProd + doc.m_mu_theta[k] * doc.m_sstat[k] - moment;

                // gradient
                muG = -m_tau * doc.m_mu_theta[k] + m_tau * dotProd + doc.m_sstat[k] - moment;
                if(m_adaFlag)
                    doc.m_mu_theta[k] += m_stepSize/Math.sqrt(muH[k]) * muG;
                else
                    doc.m_mu_theta[k] += m_stepSize * muG;

                muH[k] += muG * muG;
            }
            printFValue(lastFValue, fValue);
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;

        } while(iter++ < iterMax && Math.abs(diff) > cvg);
        if(m_displayLv != 0)
            System.out.println("------------------------");
    }

    protected void printFValue(double oldFValue, double fValue) {

        if (m_displayLv == 2) {
            System.out.println("Fvalue is " + fValue);
        } else if (m_displayLv == 1) {
            if (fValue < oldFValue)
                System.out.print("o");
            else
                System.out.print("x");
        } else if (m_displayLv == 0)
            return;
    }

    protected void update_theta_id_sigma(_Doc4EUB doc){
        int N = doc.getTotalDocLength();
        double fValue, moment;
        double lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 30, iter = 0;

        double sigmaSqrtG = 0, sigmaSqrtH[] = new double[number_of_topics];
        do{
            fValue = 0;
            for(int k=0; k<number_of_topics; k++) {
                // function value
                moment = N * Math.exp(doc.m_mu_theta[k] + 0.5 * doc.m_sigma_sqrt_theta[k] * doc.m_sigma_sqrt_theta[k]
                        - doc.m_logZeta);
                fValue += -0.5 * m_tau * doc.m_sigma_sqrt_theta[k] * doc.m_sigma_sqrt_theta[k] - moment
                        + 0.5 * Math.log(doc.m_sigma_sqrt_theta[k] * doc.m_sigma_sqrt_theta[k]);
                // gradient
                sigmaSqrtG = -m_tau * doc.m_sigma_sqrt_theta[k] - doc.m_sigma_sqrt_theta[k] * moment
                        + 1 / doc.m_sigma_sqrt_theta[k];

                if (m_adaFlag)
                    doc.m_sigma_sqrt_theta[k] += m_stepSize / Math.sqrt(sigmaSqrtH[k]) * sigmaSqrtG;
                else
                    doc.m_sigma_sqrt_theta[k] += m_stepSize * sigmaSqrtG;

                sigmaSqrtH[k] += sigmaSqrtG * sigmaSqrtG;

                if (Double.isNaN(doc.m_sigma_sqrt_theta[k]) || Double.isInfinite(doc.m_sigma_sqrt_theta[k])) {
                    System.out.println("Doc: sigma_sqrt_theta[k] is Nan or Infinity!!");
                    break;
                }
            }

            printFValue(lastFValue, fValue);
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
        } while(iter++ < iterMax && Math.abs(diff) > cvg);

        for(int k=0; k<number_of_topics; k++)
            doc.m_sigma_theta[k] = doc.m_sigma_sqrt_theta[k] * doc.m_sigma_sqrt_theta[k];
        if(m_displayLv != 0)
            System.out.println("------------------------");
    }

    // taylor parameter
    protected void update_zeta(_Doc4EUB doc) {
        doc.m_logZeta = doc.m_mu_theta[0] + 0.5 * doc.m_sigma_theta[0];

        for (int k = 1; k < number_of_topics; k++) {
            doc.m_logZeta = Utils.logSum(doc.m_logZeta, doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k]);
        }
    }

    protected double calc_log_likelihood_per_topic(_Topic4EUB topic) {

        double logLikelihood = 0.5 * m_embeddingDim * (Math.log(m_alpha_s) + 1);
        double determinant = new Matrix(topic.m_sigma_phi).det();
        if (determinant < 0)
            System.out.println("[error]Negative determinant in likelihood for topic!");
        logLikelihood += 0.5 * Math.log(Math.abs(determinant)) - 0.5 * m_alpha_s *
                sumSigmaDiagAddMuTransposeMu(topic.m_sigma_phi, topic.m_mu_phi);
        return logLikelihood;
    }

    protected double calc_log_likelihood_per_doc(_Doc4EUB doc) {
        // the first part
        int dLength = doc.getTotalDocLength();
        double logLikelihood = 0.5 * number_of_topics * (Math.log(m_tau) + 1)
                - dLength * (doc.m_logZeta -1)
                - 0.5 * m_tau * sumSigmaDiagAddMuTransposeMu(doc.m_sigma_theta, doc.m_mu_theta);

        if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] Doc: loglikelihood is Nan or Infinity!!");

        double determinant = 1;
        for (int k = 0; k < number_of_topics; k++) {
            determinant *= doc.m_sigma_theta[k];
        }

        if (determinant <= 0)
            System.out.println("[error]Negative determinant in likelihood for doc!");
        logLikelihood += 0.5 * Math.log(Math.abs(determinant));

        if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] Doc: loglikelihood is Nan or Infinity!!");
        double term1 = 0, term2 = 0;
        _User4EUB user = m_users.get(m_docUserMap.get(doc.getIndex()));
        for (int k = 0; k < number_of_topics; k++) {
            _Topic4EUB topic = m_topics[k];

            for (int m = 0; m < m_embeddingDim; m++) {
                term1 += doc.m_mu_theta[k] * topic.m_mu_phi[m] * user.m_mu_u[m];
                for (int l = 0; l < m_embeddingDim; l++) {
                    term2 += (user.m_sigma_u[m][l] + user.m_mu_u[m] * user.m_mu_u[l])
                            * (topic.m_sigma_phi[m][l] + topic.m_mu_phi[m] * topic.m_mu_phi[l]);
                }
            }
        }
        logLikelihood += m_tau * term1 - 0.5 * m_tau * term2;
        if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] Doc: loglikelihood is Nan or Infinity!!");

        // the second part which involves with words
        _SparseFeature[] fv = doc.getSparse();
        for (int k = 0; k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                int wid = fv[n].getIndex();
                double v = fv[n].getValue() * doc.m_phi[n][k];
                logLikelihood += v * (doc.m_mu_theta[k] - Math.log(doc.m_phi[n][k]) + topic_term_probabilty[k][wid]);
                if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                    System.out.println("[error] Doc: loglikelihood is Nan or Infinity!!");
            }
            logLikelihood -= dLength * Math.exp(doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k] - doc.m_logZeta);
            if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                System.out.println("[error] Doc: loglikelihood is Nan or Infinity!!");
        }
        return logLikelihood;
    }

    protected double calc_log_likelihood_per_user(_User4EUB ui) {
        double logLikelihood = 0.5 * m_embeddingDim * (Math.log(m_gamma) + 1) -
                0.5 * m_gamma * sumSigmaDiagAddMuTransposeMu(ui.m_sigma_u, ui.m_mu_u);

        // calculate the determinant of the matrix
        double determinant = new Matrix(ui.m_sigma_u).det();
        if (determinant < 0)
            System.out.println("[error]Negative determinant in likelihood for user!");
        logLikelihood += 0.5 * Math.log(Math.abs(determinant));
        if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] User: the likelihood is Nan or Infinity!!");

        int i = m_usersIndex.get(ui.getUserID());
        HashSet<Integer> interactions = m_networkMap.containsKey(i) ? m_networkMap.get(i) : null;

        for (int j = 0; j < m_users.size(); j++) {
            if (i == j) continue;
            _User4EUB uj = m_users.get(j);
            int eij = interactions != null && interactions.contains(j) ? 1 : 0;
            logLikelihood += calcLogLikelihoodOneEdge(ui, uj, i, j, eij);
            if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                System.out.println("[error] User: the likelihood is Nan or Infinity!!");

        }
        return logLikelihood;
    }

    protected double calcLogLikelihoodOneEdge(_User4EUB ui, _User4EUB uj, int i, int j, int eij) {
        double muDelta = ui.m_mu_delta[j], sigmaDelta = ui.m_sigma_delta[j];
        double epsilon = ui.m_epsilon[j], epsilon_prime = ui.m_epsilon_prime[j];
        double xi2 = m_xi * m_xi;

        double logLikelihood = eij * (muDelta + Math.log(m_rho))
                + ((1 - m_rho) * (1 - eij) / epsilon_prime - 1 / epsilon) * Math.exp(muDelta + 0.5 * sigmaDelta * sigmaDelta)
                -(muDelta * muDelta + sigmaDelta * sigmaDelta)/(2 * xi2) + Math.log(Math.abs(sigmaDelta))
                -1/epsilon - Math.log(epsilon) + eij + (1-eij) * (Math.log(epsilon_prime) + 1/epsilon_prime)
                - Math.log(m_xi) + 0.5;

        if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] Edge: the likelihood is Nan or Infinity!!");
        for (int m = 0; m < m_embeddingDim; m++) {
            logLikelihood += muDelta /xi2 * ui.m_mu_u[m] * uj.m_mu_u[m];
            if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                System.out.println("[error] Edge: the likelihood is Nan or Infinity!!");
            for (int l = 0; l < m_embeddingDim; l++) {
                logLikelihood -= 1 / (2*xi2) * (ui.m_sigma_u[m][l] + ui.m_mu_u[m] * ui.m_mu_u[l]) *
                        (uj.m_sigma_u[m][l] + uj.m_mu_u[m] * uj.m_mu_u[l]);
                if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                    System.out.println("[error] Edge: the likelihood is Nan or Infinity!!");
            }
        }
        if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] Edge: the likelihood is Nan or Infinity!!");
        return logLikelihood;
    }


    // fixed cross validation with specified fold number
    public void fixedCrossValidation(int k, String saveDir) {

        System.out.println(toString());

        double perplexity = 0;
        constructNetwork();
        System.out.format("\n==========Start %d-fold cross validation=========\n", k);
        m_trainSet.clear();
        m_testSet.clear();
        for (int i = 0; i < m_users.size(); i++) {
            for (_Review r : m_users.get(i).getReviews()) {
                if (r.getMask4CV() == k) {
                    r.setType(_Doc.rType.TEST);
                    m_testSet.add(r);
                } else {
                    r.setType(_Doc.rType.TRAIN);
                    m_trainSet.add(r);
                }
            }
        }
        buildUserDocMap();
        EM();
        System.out.format("In one fold, (train: test)=(%d : %d)\n", m_trainSet.size(), m_testSet.size());
        if (m_mType == modelType.CV4DOC) {
            System.out.println("[Info]Current mode is cv for docs, start evaluation....");
            perplexity = evaluation();

        } else if (m_mType == modelType.CV4EDGE) {
            System.out.println("[Info]Current mode is cv for edges, link predication is performed later.");
        } else {
            System.out.println("[error]Please specify the correct mode for evaluation!");
        }
        printStat4OneFold(k, saveDir, perplexity);
    }

    public void printStat4OneFold(int k, String saveDir, double perplexity) {
        // record related information
        System.out.println("[Info]Finish training, start saving data...");
        File fileDir = new File(saveDir);
        if (!fileDir.exists())
            fileDir.mkdirs();

        printOneFoldPerplexity(saveDir, perplexity);
        printTopWords(30, String.format("%s/TopKWords.txt", saveDir));
        printTopicEmbedding(String.format("%s/TopicEmbedding.txt", saveDir));
        printUserEmbedding(String.format("%s/EUB_embedding_dim_%d_fold_%d.txt", saveDir, m_embeddingDim, k));
        printUserEmbeddingWithDetails(String.format("%s/EUB_EmbeddingWithDetails_dim_%d_fold_%d.txt", saveDir, m_embeddingDim, k));
    }

    public void printOneFoldPerplexity(String saveDir, double perplexity) {
        try {
            PrintWriter writer = new PrintWriter(new File(String.format("%s/Perplexity.txt", saveDir)));
            writer.format("perplexity: %.5f\n", perplexity);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // evaluation in single-thread
    public double evaluation() {

        double allLoglikelihood = 0;
        int totalWords = 0;
        for (_Doc d : m_testSet) {
            double likelihood = inference(d);
            allLoglikelihood += likelihood;
            totalWords += d.getTotalDocLength();
        }

        double perplexity = Math.exp(-allLoglikelihood / totalWords);
        double avgLoglikelihood = allLoglikelihood / m_testSet.size();

        System.out.format("[Stat]InferIter=%d, perplexity=%.4f, total words=%d, all_log-likelihood=%.4f, " +
                        "avg_log-likelihood=%.4f\n\n",
                m_testInferMaxIter, perplexity, totalWords, allLoglikelihood, avgLoglikelihood);
        return perplexity;
    }

    @Override
    public double calculate_E_step(_Doc d) {
        _Doc4EUB doc = (_Doc4EUB) d;
        double cur = varInference4Doc(doc);
        updateStats4Doc(doc);
        return cur;
    }

    @Override
    protected void initTestDoc(_Doc d) {
        ((_Doc4EUB) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma);
    }

    @Override
    public double inference(_Doc d) {
        initTestDoc(d);
        double likelihood = calculate_E_step(d);
        estThetaInDoc(d);
        return likelihood;
    }

    public void printTopicEmbedding(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for (int i = 0; i < m_topics.length; i++) {
                _Topic4EUB topic = m_topics[i];
                writer.format("Topic %d\nmu_phi:\n", i);
                for (double mu : topic.m_mu_phi) {
                    writer.format("%.3f\t", mu);
                }
                writer.write("\nsimga_phi:\n");
                for (double sigma[] : topic.m_sigma_phi) {
                    for (double s : sigma)
                        writer.format("%.5f\t", s);
                    writer.write("\n");
                }
                writer.write("----------------------------\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // print out user embedding with both mean and variance
    public void printUserEmbeddingWithDetails(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for (int i = 0; i < m_users.size(); i++) {
                _User4EUB user = m_users.get(i);
                writer.format("User %s\nmu_u:\n", user.getUserID());
                for (double mu : user.m_mu_u) {
                    writer.format("%.3f\t", mu);
                }
                writer.write("\nsimga_u:\n");
                for (double sigma[] : user.m_sigma_u) {
                    for (double s : sigma)
                        writer.format("%.5f\t", s);
                    writer.write("\n");
                }
                writer.write("----------------------------\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // print out user embedding with mean
    public void printUserEmbedding(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", m_users.size(), m_embeddingDim);
            for (int i = 0; i < m_users.size(); i++) {
                _User4EUB user = m_users.get(i);
                writer.format("%s\t", user.getUserID());
                for (double mu : user.m_mu_u) {
                    writer.format("%.4f\t", mu);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadQuestionIds(String filename){
        if(m_data.equals("YelpNew"))
            loadQuestionIds4Yelp(filename);
        else if(m_data.equals("StackOverflow"));
            loadQuestionIds4Stack(filename);
    }

    /*****The following codes are used in answerer recommendation********/
    HashMap<Integer, _Doc4EUB> m_questionMap = new HashMap<>();
    public void loadQuestionIds4Yelp(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            while ((line = reader.readLine()) != null) {
                // the first is uid, the second is qid
                String[] strs = line.trim().split("\\s+");
                String uId = strs[0];
                int qId = Integer.valueOf(strs[1]);
                // find the user first
                _User4EUB user = m_users.get(m_usersIndex.get(uId));
                // find the ref of the review document
                for(_Review r: user.getReviews()){
                    _Doc4EUB d = (_Doc4EUB) r;
                    if(d.getPostId() == qId){
                        m_questionMap.put(qId, d);
                        break;
                    }
                }
            }
            reader.close();
            System.out.format("Finish loading %d question ids from %s.\n", m_questionMap.size(), filename);

        }catch(IOException e){
            e.printStackTrace();
        }
    }

    /*****The following codes are used in answerer recommendation********/
    HashMap<String, _Doc4EUB> m_reviewMap = new HashMap<>();
    public void loadQuestionIds4Stack(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            while ((line = reader.readLine()) != null) {
                // the first is uid, the second is qid
                String[] strs = line.trim().split("\\s+");
                String uId = strs[0];
                int qId = Integer.valueOf(strs[1]);
                // find the user first
                _User4EUB user = m_users.get(m_usersIndex.get(uId));
                // find the ref of the review document
                for(_Review r: user.getReviews()){
                    _Doc4EUB d = (_Doc4EUB) r;
                    if(d.getID() == qId){
                        String rId = String.format("%s#%d", uId, qId);
                        m_reviewMap.put(rId, d);
                        break;
                    }
                }
            }
            reader.close();
            System.out.format("Finish loading %d question ids from %s.\n", m_reviewMap.size(), filename);

        }catch(IOException e){
            e.printStackTrace();
        }
    }
    // print out the mu of topic embedding as K*M matrix
    public void printTopicEmbeddingAsMtx(String dir){
        try {
            String filename = String.format("%s/EUB_Phi_dim_%d.txt", dir, m_embeddingDim);
            PrintWriter writer = new PrintWriter(new File(filename+""));
            writer.format("%d\t%d", m_topics.length, m_embeddingDim);
            for (int i = 0; i < m_topics.length; i++) {
                _Topic4EUB topic = m_topics[i];
                for (double mu : topic.m_mu_phi) {
                    writer.format("%.4f\t", mu);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void printTheta(String dir){
        if(m_data.equals("YelpNew"))
            printTheta4Yelp(dir);
        else if(m_data.equals("StackOverflow"))
            printTheta4Stack(dir);
    }

    public void printTheta4Yelp(String dir){
        try {
            String filename = String.format("%s/EUB_theta_dim_%d.txt", dir, m_embeddingDim);
            PrintWriter writer = new PrintWriter(new File(filename+""));
            writer.format("%d\t%d", m_reviewMap.size(), m_embeddingDim);
            for(String rId: m_reviewMap.keySet()){
                writer.write( rId + "\t");
                for(double v: m_reviewMap.get(rId).m_mu_theta)
                    writer.format("%.4f\t", v);
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void printTheta4Stack(String dir){
        try {
            String filename = String.format("%s/EUB_theta_dim_%d.txt", dir, m_embeddingDim);
            PrintWriter writer = new PrintWriter(new File(filename+""));
            writer.format("%d\t%d", m_questionMap.size(), m_embeddingDim);
            for(int qId: m_questionMap.keySet()){
                writer.write(qId + "\t");
                for(double v: m_questionMap.get(qId).m_mu_theta)
                    writer.format("%.4f\t", v);
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
