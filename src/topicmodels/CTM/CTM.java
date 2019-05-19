package topicmodels.CTM;

import java.io.*;
import java.util.*;

import Jama.Matrix;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures.*;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

public class CTM extends LDA_Variational {
    public double[] mu;
    public double[][] inv_cov;
    public double[][] cov;
    public double log_det_cov;

    public double[] muStat;
    public double[][] covStat;

    public int len1;
    public int len2;

    public CTM(int emMaxIter, double emConverge,
               double beta, _Corpus corpus, double lambda,
               int number_of_topics, double alpha, int varMaxIter, double varConverge){
        super(emMaxIter, emConverge,
                beta, corpus, lambda,
                number_of_topics, alpha, varMaxIter, varConverge);
        len1 = number_of_topics;
        len2 = number_of_topics-1;
        m_logSpace = true;
    }

    @Override
    public String toString(){
        return String.format("CTM[k:%d]\n",
                number_of_topics);
    }

    public void initModel(){
        System.out.println("[Info]Initializing CTM Model...");

        mu = new double[len2];
        inv_cov = new double[len2][len2];
        cov = new double[len2][len2];
        topic_term_probabilty = new double[len1][vocabulary_size];
        log_det_cov = 0.0;

        muStat = new double[len2];
        covStat = new double[len2][len2];
        word_topic_sstat = new double[len1][vocabulary_size];

        m_sstat = new double[len1];

        int initialFlag = 0;
        Random r = new Random();

        if(initialFlag ==0){

            for(int i=0; i<len2; i++){
                mu[i] = 0;
                cov[i][i] = 1.0;

            }

            double val = 0.0;
            for(int i=0; i<len1; i++){
                double sum = 0.0;
                for(int j=0; j<vocabulary_size; j++){
                    val = r.nextDouble()+1.0/100.0;
                    //val = r.nextDouble();

                    sum += val;
                    topic_term_probabilty[i][j] = val;

                }

                for(int j=0; j<vocabulary_size; j++){
                    topic_term_probabilty[i][j] = Math.log(topic_term_probabilty[i][j])-Math.log(sum);

                }
            }
        }
        else{
            // corpus initial
            long seed = 1115574245;
            r.setSeed(seed);

            for(int i=0; i<len2; i++){
                mu[i] = 0;
                cov[i][i] = 1.0;
            }

            int corpusSize = m_trainSet.size();

            for(int i=0; i<len1; i++){
                double sum = 0;
                for(int j=0; j<1; j++){
                    double d = corpusSize*r.nextDouble();
                    int randomDocID = (int) d;
                    System.out.println("docID"+randomDocID);

                    _Doc doc = m_trainSet.get(randomDocID);
                    _SparseFeature[] fv = doc.getSparse();
                    for(int n=0; n<fv.length; n++){
                        int wid = fv[n].getIndex();
                        double value = fv[n].getValue();
                        topic_term_probabilty[i][wid] += value;
                    }
                }

                for(int n=0; n<vocabulary_size; n++){
                    topic_term_probabilty[i][n] += 1.0 + r.nextDouble();
                    sum += topic_term_probabilty[i][n];
                }

                for(int n=0; n<vocabulary_size; n++){
                    topic_term_probabilty[i][n] = Math.log((double)topic_term_probabilty[i][n]/sum);
                }
            }
        }
        Matrix covMatrix = new Matrix(cov);
        Matrix inv_covMatrix = covMatrix.inverse();
        double det_cov = covMatrix.det();
        log_det_cov = Math.log(det_cov);
        inv_cov = inv_covMatrix.getArray();
    }

    public void initStats(){
        Arrays.fill(muStat, 0);

        for(int i=0; i<len2; i++){
            Arrays.fill(covStat[i], 0);
        }

        for(int i=0; i<len1; i++){
            Arrays.fill(word_topic_sstat[i], 1e-2);
        }
    }

    public void initDoc(_Doc d){
        double phiArg = (double)1/len1;
        //System.out.println("phiArg"+phiArg);
        double zetaArg = 10;
        double nuArg = 10;
        double lambdaArg = 0;

        _Doc4ETBIR doc = (_Doc4ETBIR) d;
        doc.m_phi = new double[doc.getSparse().length][len1];
        for (int n = 0; n <doc.getSparse().length; n++) {
            Arrays.fill(doc.m_phi[n], phiArg);
        }

        // here m_mu is the mean lambda in CTM paper
        doc.m_mu = new double[len1];
        Arrays.fill(doc.m_mu, lambdaArg);
        doc.m_mu[len2] = 0;
        // m_Sigma is the variance nu in CTM paper
        doc.m_Sigma = new double[len1];
        Arrays.fill(doc.m_Sigma, nuArg);
        doc.m_Sigma[len2] = 0;

        doc.m_logZeta = zetaArg;
        doc.m_topics = new double[len1];
    }

    @Override
    protected void initTestDoc(_Doc d) {
        initDoc(d);
    }

    public double calculate_E_step(_Doc d) {
        ArrayList<_Doc> lineSearchFailDoc = new ArrayList<_Doc>();

        double curLikelihood = varInference(d, lineSearchFailDoc);
        updateStats(d);
        return curLikelihood;
    }

    public double varInference(_Doc doc, ArrayList<_Doc> lineSearchFailDoc){
        int iter = 0;
        double curLikelihood = 0.0;
        double converge = 0.0;
        double oldLikelihood = 0.0;

        if(m_varConverge>0)
            oldLikelihood = calLikelihood(doc);

        boolean fail = false;
        do{
            iter += 1;
            opt_zeta(doc);
            //fail is true
            if(opt_lambda(doc)){
                fail = true;
            }

            opt_zeta(doc);
            opt_nu(doc);

            opt_zeta(doc);
            opt_phi(doc);

            if(m_varConverge>0){
                curLikelihood = calLikelihood(doc);

                converge = (oldLikelihood-curLikelihood)/oldLikelihood;
                oldLikelihood = curLikelihood;

            }
            //System.out.println("iter"+iter);
        }while((iter<m_varMaxIter)&&(Math.abs(converge)>m_varConverge));

        if(fail){
            lineSearchFailDoc.add(doc);
        }

        return curLikelihood;

    }

    public double calLikelihood(_Doc d){
        double likelihood = 0.0;
        _Doc4ETBIR doc = (_Doc4ETBIR) d;
        likelihood += -0.5*log_det_cov;
        likelihood += 0.5*(len2+doc.m_Sigma[len2]);

        for(int i=0; i<len2; i++){
            likelihood += -(0.5) * doc.m_Sigma[i] * inv_cov[i][i];
        }


        for(int i=0; i<len2; i++){
            for(int j=0; j<len2; j++){
                likelihood += -(0.5) * (doc.m_mu[i]-mu[i])*inv_cov[i][j]*(doc.m_mu[j] - mu[j]);
            }
        }

        for(int i=0; i<len2; i++){
            likelihood += 0.5 * Math.log(doc.m_Sigma[i]);
        }

        likelihood += -expect_mult_norm(doc)*doc.getTotalDocLength();

        _SparseFeature[] fv = doc.getSparse();
        for(int n=0; n<fv.length; n++){
            int wid = fv[n].getIndex();
            double v = fv[n].getValue();

            for(int i=0; i<len1; i++){
                likelihood += doc.m_phi[n][i]*v*(doc.m_mu[i]+topic_term_probabilty[i][wid]-Math
                        .log(doc.m_phi[n][i]));
            }
        }

        return likelihood;

    }

    public double expect_mult_norm(_Doc d) {
        double sum_exp = 0.0;
        double mult_zeta = 0.0;

        _Doc4ETBIR doc = (_Doc4ETBIR)d;

        for (int i = 0; i <len1; i++) {
            sum_exp += Math.exp(doc.m_mu[i] + 0.5 * doc.m_Sigma[i]);
        }

        mult_zeta =  (1.0 / doc.m_logZeta) * sum_exp - 1 + Math.log(doc.m_logZeta);
        return mult_zeta;
    }

    public void updateStats(_Doc d){
        _Doc4ETBIR doc = (_Doc4ETBIR)d;
        for(int i=0; i<len2; i++){
            muStat[i] += doc.m_mu[i];
            for(int j=0; j<len2; j++){
                double lilj = doc.m_mu[i]*doc.m_mu[j];

                if(i==j){
                    covStat[i][j] += doc.m_Sigma[i] + lilj;
                }else{
                    covStat[i][j] += lilj;
                }
            }

        }

        _SparseFeature[] fv = doc.getSparse();
        for(int n=0; n<fv.length; n++){
            int wid = fv[n].getIndex();
            double v = fv[n].getValue();

            for(int i=0; i<len1; i++){
                word_topic_sstat[i][wid] += v*doc.m_phi[n][i];

            }
        }
    }

    public void opt_zeta(_Doc d){
        _Doc4ETBIR doc = (_Doc4ETBIR)d;
//        doc.m_logZeta = doc.m_mu[0] + 0.5 * doc.m_Sigma[0];
//        for (int k = 1; k < len2; k++)
//            doc.m_logZeta = Utils.logSum(doc.m_logZeta, doc.m_mu[k] + 0.5 * doc.m_Sigma[k]);
        doc.m_logZeta= 1.0;
        for(int i=0; i<len2; i++){
            doc.m_logZeta += Math.exp(doc.m_mu[i] + 0.5 * doc.m_Sigma[i]);
        }
    }

    public void opt_phi(_Doc d){
        double logSum = 0.0;
        _Doc4ETBIR doc = (_Doc4ETBIR)d;

        _SparseFeature[] fv = doc.getSparse();
        for(int n=0; n<fv.length; n++){
            int wid = fv[n].getIndex();
            double v = fv[n].getValue();

            for(int i=0; i<len1; i++){
                doc.m_phi[n][i] = topic_term_probabilty[i][wid]+doc.m_mu[i];

            }

            logSum = Utils.logSum(doc.m_phi[n]);
            for(int i=0; i<len1; i++){
                doc.m_phi[n][i] = Math.exp(doc.m_phi[n][i]-logSum);

            }

        }
    }


    public boolean opt_lambda(_Doc d){
        boolean failSearch = false;
        int[] iflag = {0}, iprint={-1, 3};
        double fValue=0.0;
        int xSize = len2;
        double[] x = new double[len2];
        double[] x_g = new double[len2];
        double[] x_diag = new double[len2];
        _Doc4ETBIR doc = (_Doc4ETBIR)d;

        for(int i=0; i<len2; i++){
            x[i] = doc.m_mu[i];
            x_diag[i] = 0;
            x_g[i] = 0;
        }

        int iter = 0;
        double eps = 1e-3;
        try{
            do{
                fValue = calcLambdaFuncGradient(doc, x, x_g);
                LBFGS.lbfgs(xSize, 4, x, fValue, x_g, false, x_diag, iprint, eps, 1e-16, iflag);
            }while(iflag[0]!=0 && iter++<15);

        }catch(ExceptionWithIflag e){
            failSearch = true;
            e.printStackTrace();
        }
        if(iflag[0]==-1)
            System.err.println("[Warning] LBFGS fail docID: "+doc.getID());
        for(int i=0; i<len2; i++){
            doc.m_mu[i] = x[i];
        }
        doc.m_mu[len2] = 0;
        return failSearch;
    }

    public double calcLambdaFuncGradient(_Doc d, double[] x, double[] x_g){
        double[] sum_phi = new double[len2];
        _Doc4ETBIR doc = (_Doc4ETBIR)d;
        _SparseFeature[] fv = doc.getSparse();
        for(int i=0; i<len2; i++){
            for(int n=0; n<fv.length; n++){
                int wid = fv[n].getIndex();
                double v= fv[n].getValue();

                sum_phi[i] += v*doc.m_phi[n][i];
            }
        }

        double term1 = 0.0;
        double[] gTerm1 = new double[len2];
        for(int i=0; i<len2; i++){
            term1 += x[i]*sum_phi[i];
            gTerm1[i] = sum_phi[i];
        }

        double term2 = 0.0;
        double[] gTerm2 = new double[len2];
        double[] lambda_mu = new double[len2];
        for(int i=0; i<len2; i++){
            lambda_mu[i] = x[i]-mu[i];
        }

        Matrix inv_covMatrix = new Matrix(inv_cov);
        Matrix lambda_muMatrix = new Matrix(lambda_mu, len2);
        term2 = -0.5* (lambda_muMatrix.transpose().times(
                inv_covMatrix.times(lambda_muMatrix)).get(0, 0));
        for(int i=0; i<len2; i++){
            for(int j=0; j<len2; j++){
                gTerm2[i] -= inv_cov[i][j]*lambda_mu[j];
            }
        }

        double term3=0;
        double[] gTerm3 = new double[len2];
        for(int i=0; i<len2; i++){
            term3 += Math.exp(x[i]+0.5*doc.m_Sigma[i]);
            gTerm3[i] = -doc.getTotalDocLength()*Math.exp(x[i]+doc.m_Sigma[i]*0.5)/doc.m_logZeta;
        }
        term3 = -doc.getTotalDocLength()*(term3/doc.m_logZeta);


        double lambdaLikelihood = -(term1+term2+term3);
        for(int i=0; i<len2; i++){
            x_g[i] = -(gTerm1[i]+gTerm2[i]+gTerm3[i]);
        }
        return lambdaLikelihood;
    }

    public void opt_nu(_Doc d){
        int[] iflag = {0}, iprint={-1, 3};
        double fValue = 0.0;
        int xSize = len2;
        double[] x = new double[len2];
        double[] x_g = new double[len2];
        double[] x_diag = new double[len2];
        _Doc4ETBIR doc = (_Doc4ETBIR)d;

        for(int i=0; i<len2; i++){
            x[i] = Math.log(doc.m_Sigma[i]);
            x_diag[i] = 0;
            x_g[i] = 0;
        }

        int iter = 0;
        try{
            do{
                fValue = calcNuFuncGradient(doc, x, x_g);
                LBFGS.lbfgs(xSize, 4, x, fValue, x_g, false, x_diag, iprint, 1e-6, 1e-32, iflag);
            }while(iflag[0]!=0 && iter++<15);

        }catch(ExceptionWithIflag e){
            e.printStackTrace();
        }

        for(int i=0; i<len2; i++){
            doc.m_Sigma[i] = Math.exp(x[i]);
        }
    }

    public double calcNuFuncGradient(_Doc d, double[] x, double[] x_g){
        double likelihood = 0.0;
        _Doc4ETBIR doc = (_Doc4ETBIR)d;

        double term1 = 0.0;
        double term2 = 0.0;
        double term3 = 0.0;

        double[] gTerm1 = new double[len2];
        double[] gTerm2 = new double[len2];
        double[] gTerm3 = new double[len2];

        for(int i=0; i<len2; i++){
            term1 += Math.exp(x[i])*inv_cov[i][i];
            gTerm1[i] = -0.5*Math.exp(x[i])*inv_cov[i][i];
        }
        term1 = -0.5*term1;

        for(int i=0; i<len2; i++){
            term2 += Math.exp(doc.m_mu[i]+Math.exp(x[i])/2);
            gTerm2[i] = -0.5*Math.exp(x[i])*Math.exp(doc.m_mu[i]+Math.exp(x[i])/2)*doc.getTotalDocLength()/doc.m_logZeta;
        }
        term2 = -doc.getDocLength()*term2/doc.m_logZeta;

        for(int i=0; i<len2; i++){
            term3 += 0.5*x[i];
            gTerm3[i] = 0.5;
        }

        likelihood = -(term1+term2+term3);
        for(int i=0; i<len2; i++){
            x_g[i] = -(gTerm1[i]+gTerm2[i]+gTerm3[i]);
        }
        return likelihood;
    }

    @Override
    public void calculate_M_step(int iter){
        //mu
        for(int i=0; i<len2; i++){
            mu[i] = muStat[i]/m_trainSet.size();
        }
        //cov
        for(int i=0; i<len2; i++){
            for(int j=0; j<len2; j++){
                cov[i][j] = covStat[i][j] + m_trainSet.size() * mu[i] * mu[j]- mu[i]* muStat[j] - mu[j] * muStat[i];
                cov[i][j] = cov[i][j] / m_trainSet.size();
            }
        }
        Matrix covMatrix = new Matrix(cov);
        Matrix inv_covMatrix = covMatrix.inverse();
        double det_cov = covMatrix.det();
        log_det_cov = Math.log(det_cov);
        inv_cov = inv_covMatrix.getArray();
        //beta
        for(int i=0; i<len1; i++){
            double sum = Utils.sumOfArray(word_topic_sstat[i]);
            for (int n = 0; n < vocabulary_size; n++) {
                topic_term_probabilty[i][n] = Math.log(word_topic_sstat[i][n])-Math.log(sum);
            }
        }
    }

    @Override
    public void EM(){
        initModel();
        initStats();
        for(_Doc d: m_trainSet){
            initDoc(d);
        }

        int iter = 0;
        double oldTotal = 0.0;
        double curTotal = 0.0;
        double converge = 0.0;

        //boolean resetDocParam = true;
        do{
            initStats();
            curTotal = 0.0;
            for(_Doc d:m_trainSet) {
                curTotal += calculate_E_step(d);
            }
            if(Double.isNaN(curTotal)){
                System.err.println("[Error]E_step produces NaN likelihood...");
                break;
            }
            if(iter >0)
                converge = Math.abs((oldTotal-curTotal)/oldTotal);
            else
                converge = 1.0;

            calculate_M_step(iter);
            oldTotal = curTotal;
            System.out.format("[Info]EM iteration %d: likelihood is %.2f, converges to %.4f...\n",
                    iter, curTotal, converge);
        }while((iter++<number_of_iteration)&&(converge>m_converge));

        finalEst();
    }

    //k-fold Cross Validation.
    @Override
    public void crossValidation(int k) {
        m_trainSet = new ArrayList<_Doc>();
        m_testSet = new ArrayList<_Doc>();

        double[] perp = new double[k];
        double[] like = new double[k];
        m_corpus.shuffle(k);
        int[] masks = m_corpus.getMasks();
        ArrayList<_Doc> docs = m_corpus.getCollection();
        //Use this loop to iterate all the ten folders, set the train set and test set.
        System.out.println("[Info]Start RANDOM cross validation...");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < masks.length; j++) {
                if( masks[j]==i )
                    m_testSet.add(docs.get(j));
                else
                    m_trainSet.add(docs.get(j));
            }

            System.out.format("====================\n[Info]Fold No. %d: train size = %d, test size = %d....\n", i, m_trainSet.size(), m_testSet.size());

            long start = System.currentTimeMillis();
            //train
            EM();

            //test
            double[] results = EvaluationMultipleMetrics();
            perp[i] = results[0];
            like[i] = results[1];

            System.out.format("[Info]Train/Test finished in %.2f seconds...\n", (System.currentTimeMillis()-start)/1000.0);
            m_trainSet.clear();
            m_testSet.clear();
        }

        //output the performance statistics
        double mean = Utils.sumOfArray(perp)/k, var = 0;
        for(int i=0; i<perp.length; i++)
            var += (perp[i]-mean) * (perp[i]-mean);
        var = Math.sqrt(var/k);
        System.out.format("[Stat]Perplexity %.3f+/-%.3f\n", mean, var);

        mean = Utils.sumOfArray(like)/k;
        var = 0;
        for(int i=0; i<like.length; i++)
            var += (like[i]-mean) * (like[i]-mean);
        var = Math.sqrt(var/k);
        System.out.format("[Stat]Loglikelihood %.3f+/-%.3f\n", mean, var);
    }

    @Override
    public void printAggreTopWords(int k, String topWordPath, HashMap<String, List<_Doc>> docCluster) {
        File file = new File(topWordPath);
        try{
            file.getParentFile().mkdirs();
            file.createNewFile();
        } catch(IOException e){
            e.printStackTrace();
        }
        try{
            PrintWriter topWordWriter = new PrintWriter(file);

            for(Map.Entry<String, List<_Doc>> entryU : docCluster.entrySet()) {
                double[] gamma = new double[number_of_topics];
                Arrays.fill(gamma, 0);
                for(_Doc d:entryU.getValue()) {
                    double sum = Utils.logSum(((_Doc4ETBIR)d).m_mu);
                    for (int i = 0; i < number_of_topics; i++) {
                        gamma[i] += Math.exp(((_Doc4ETBIR)d).m_mu[i] - sum);
                    }
                }
                for(int i = 0; i < number_of_topics; i++){
                    gamma[i] /= entryU.getValue().size();
                }

                topWordWriter.format("ID %s(%d reviews)\n", entryU.getKey(), entryU.getValue().size());
                for (int i = 0; i < topic_term_probabilty.length; i++) {
                    MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
                    for (int j = 0; j < vocabulary_size; j++)
                        fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));

                    topWordWriter.format("-- Topic %d(%.5f):\t", i, gamma[i]);
                    for (_RankItem it : fVector)
                        topWordWriter.format("%s(%.5f)\t", it.m_name, m_logSpace ? Math.exp(it.m_value) : it.m_value);
                    topWordWriter.write("\n");
                }
            }
            topWordWriter.close();
        } catch(Exception ex){
            System.err.format("[Error]Failed to open file %s\n", topWordPath);
        }
    }

    @Override
    public void printParam(String folderName, String topicmodel){
        String priorSigmaPath = String.format("%s%s_priorSigma_%d.txt", folderName, topicmodel, number_of_topics);
        String priorMuPath = String.format("%s%s_priorMu_%d.txt", folderName, topicmodel, number_of_topics);
        String postSoftmaxPath = String.format("%s%s_postSoftmax_%d.txt", folderName, topicmodel, number_of_topics);

        //print out prior parameter for covariance: Sigma
        File file = new File(priorSigmaPath);
        try{
            file.getParentFile().mkdirs();
            file.createNewFile();
        } catch(IOException e){
            e.printStackTrace();
        }
        try{
            PrintWriter sigmaWriter = new PrintWriter(file);

            for(int i = 0; i < len2; i++) {
                for(int j = 0; j < len2; j++)
                    sigmaWriter.format("%.8f\t", cov[i][j]);
                sigmaWriter.write("\n");
            }
            sigmaWriter.close();
        } catch(Exception ex){
            System.err.format("[Error]Failed to open file %s\n", priorSigmaPath);
        }

        //print out prior parameter for mean: eta
        file = new File(priorMuPath);
        try{
            file.getParentFile().mkdirs();
            file.createNewFile();
        } catch(IOException e){
            e.printStackTrace();
        }
        try{
            PrintWriter muWriter = new PrintWriter(file);
            for(int i = 0; i < len2; i++)
                muWriter.format("%.8f\t", mu[i]);
            muWriter.close();
        } catch(Exception ex){
            System.err.format("[Error]Failed to open file %s\n", priorMuPath);
        }


        //print out estimated parameter of multinomial distribution: softmax(eta)
        file = new File(postSoftmaxPath);
        try{
            file.getParentFile().mkdirs();
            file.createNewFile();
        } catch(IOException e){
            e.printStackTrace();
        }
        try{
            PrintWriter softmaxWriter = new PrintWriter(file);

            for(int idx = 0; idx < m_trainSet.size(); idx++) {
                softmaxWriter.write(String.format("No. %d Doc(user: %s, item: %s) ***************\n", idx,
                        ((_Doc4ETBIR) m_trainSet.get(idx)).getUserID(),
                        ((_Doc4ETBIR) m_trainSet.get(idx)).getItemID()));
                double sum = Utils.logSum(((_Doc4ETBIR) m_trainSet.get(idx)).m_mu)+1;
                for (int i = 0; i < number_of_topics; i++) {
                    softmaxWriter.format("%.5f\t", Math.exp(((_Doc4ETBIR) m_trainSet.get(idx)).m_mu[i] - sum));
                }
                softmaxWriter.println();
            }
            softmaxWriter.close();
        } catch(Exception ex){
            System.err.format("[Error]Failed to open file %s\n", postSoftmaxPath);
        }
    }
}