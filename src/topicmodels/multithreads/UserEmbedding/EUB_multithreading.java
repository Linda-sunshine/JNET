package topicmodels.multithreads.UserEmbedding;

import structures.*;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.EmbedModelWorker;
import topicmodels.multithreads.EmbedModel_worker;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;
import utils.Utils;

import java.util.Collection;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EUB_multithreading extends EUB {

    protected EmbedModelWorker[] m_userWorkers = null;
    protected EmbedModelWorker[] m_topicWorkers = null;

    public class Doc_worker extends TopicModel_worker {
        public Doc_worker(int number_of_topics, int vocabulary_size){
            super(number_of_topics, vocabulary_size);
        }

        @Override
        public void run(){
            m_likelihood = 0;
            m_perplexity = 0;
            m_totalWords = 0;
            for(_Doc d: m_corpus) {
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(d);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    m_perplexity += inference(d);
                    m_totalWords += d.getTotalDocLength();
                }
            }
        }

        @Override
        public double calculate_E_step(_Doc d) {
            _Doc4EUB doc = (_Doc4EUB) d;
            double cur = varInference4Doc(doc);
            updateStats4Doc(doc);
            return cur;
        }

//        public double calc_term_log_likelihood(_Doc d) {
//            int wid;
//            double v, logLikelihood = 0;
//
//            //collect the sufficient statistics
//            _SparseFeature[] fv = d.getSparse();
//            for(int n=0; n<fv.length; n++) {
//                wid = fv[n].getIndex();
//                v = fv[n].getValue();
//                double sum = 0;
//                for(int i=0; i<number_of_topics; i++) {
//                    sum += Math.exp(d.m_topics[i]) * Math.exp(topic_term_probabilty[i][wid]);
//                }
//                logLikelihood += v * Math.log(sum);
//            }
//
//            return logLikelihood;
//        }

        @Override
        public double inference(_Doc d){
            initTestDoc(d);
            double likelihood = calculate_E_step(d);
            estThetaInDoc(d);
            return likelihood;
        }

        protected void updateStats4Doc(_Doc d){
            _Doc4EUB doc = (_Doc4EUB) d;
            // update m_word_topic_stats for updating beta
            _SparseFeature[] fv = doc.getSparse();
            int wid;
            double v;
            for(int n=0; n<fv.length; n++) {
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                for(int i=0; i<number_of_topics; i++)
                    sstat[i][wid] += v*doc.m_phi[n][i];
            }
        }

        @Override
        public double accumluateStats(double[][] word_topic_sstat) {

            return super.accumluateStats(word_topic_sstat);
        }

    }

    public class User_worker extends EmbedModel_worker {

        public User_worker(int dim){
            super(dim);
        }

        @Override
        public void run(){
            m_likelihood = 0;
            m_perplexity = 0;
            for(Object o: m_objects){
                _User4EUB user = (_User4EUB) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(user);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    m_perplexity += varInference4User(user);
                }
            }
        }

        @Override
        public double calculate_E_step(Object o) {
            _User4EUB user = (_User4EUB) o;
            return varInference4User(user);
        }

        @Override
        public double accumluateStats(){
            return m_likelihood;
        }

        @Override
        public void resetStats(){}
    }

    public class Topic_worker extends EmbedModel_worker{

        public Topic_worker(int dim){
            super(dim);
        }

        @Override
        public void run(){
            m_likelihood = 0;
            m_perplexity = 0;

            for(Object o: m_objects){
                _Topic4EUB topic = (_Topic4EUB) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(topic);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    m_perplexity += varInference4Topic(topic);
                }
            }
        }

        @Override
        public double calculate_E_step(Object o) {
            _Topic4EUB topic = (_Topic4EUB) o;
            return varInference4Topic(topic);
        }

        @Override
        public double accumluateStats(){
            return m_likelihood;
        }

        @Override
        public void resetStats(){}

    }

    public EUB_multithreading(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                              int number_of_topics, double alpha, int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge, m);
        m_multithread = true;
    }

    protected void initialize_probability(Collection<_Doc> collection) {


        int cores = Runtime.getRuntime().availableProcessors();
        m_threadpool = new Thread[cores];
        m_workers = new EUB_multithreading.Doc_worker[cores];
        m_topicWorkers = new EUB_multithreading.Topic_worker[cores];
        m_userWorkers = new EUB_multithreading.User_worker[cores];

        for(int i=0; i<cores; i++) {
            m_workers[i] = new EUB_multithreading.Doc_worker(number_of_topics, vocabulary_size);
            m_topicWorkers[i] = new EUB_multithreading.Topic_worker(m_embeddingDim);
            m_userWorkers[i] = new EUB_multithreading.User_worker(m_embeddingDim);
        }

        int workerID = 0;
        for(_Doc d: collection) {
            m_workers[workerID%cores].addDoc(d);
            workerID++;
        }
        workerID = 0;
        for(_Topic4EUB t: m_topics){
            m_topicWorkers[workerID%cores].addObject(t);
            workerID++;
        }
        workerID = 0;
        for(_User4EUB u: m_users) {
            m_userWorkers[workerID%cores].addObject(u);
            workerID++;
        }
        super.initialize_probability(collection);
    }

    @Override
    protected void init() { // clear up for next iteration
        super.init();
        for(TopicModelWorker worker:m_workers)
            worker.resetStats();
    }


    @Override
    public double multithread_E_step() {
        int iter = 0;
        double likelihood = 0, docLikelihood, topicLikelihood, userLikelihood, last = -1.0, converge;

        do {
            init();

            // doc
            docLikelihood = super.multithread_E_step();
            if(Double.isNaN(docLikelihood) || Double.isInfinite(docLikelihood)){
                System.err.println("[Error]E_step for documents results in NaN likelihood...");
                break;
            }

            // topic
            topicLikelihood = multithread_general(m_topicWorkers);
            if(Double.isNaN(topicLikelihood) || Double.isInfinite(topicLikelihood)){
                System.err.println("[Error]E_step for topics results in NaN likelihood...");
            }

            // user
            userLikelihood = multithread_general(m_userWorkers);
            if(Double.isNaN(userLikelihood) || Double.isInfinite(userLikelihood)){
                System.err.println("[Error]E_step for users results in NaN likelihood...");
                break;
            }

//            likelihood = userLikelihood;
            likelihood = docLikelihood + topicLikelihood + userLikelihood;

            if(iter > 0)
                converge = Math.abs((likelihood - last) / last);
            else
                converge = 1.0;

            last = likelihood;

            if(converge < m_varConverge)
                break;
            System.out.format("[Multi-E-step] %d iteration, likelihood(d:t:u)=(%.2f, %.2f, %.2f), converge to %.8f\n",
                    iter, docLikelihood, topicLikelihood, userLikelihood, converge);
//            System.out.format("[Multi-E-step] %d iteration, likelihood-u=%.2f, converge to %.8f\n", iter, userLikelihood, converge);
        }while(iter++ < m_varMaxIter);

        return likelihood;
    }


    protected double multithread_general(EmbedModelWorker[] workers){
        double likelihood = 0.0;
        for (int i = 0; i < workers.length; i++) {
            workers[i].setType(TopicModel_worker.RunType.RT_EM);
            m_threadpool[i] = new Thread(workers[i]);
            m_threadpool[i].start();
        }
        for (Thread thread : m_threadpool) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        for (EmbedModelWorker worker : workers)
            likelihood += worker.accumluateStats();

        return likelihood;
    }


    @Override
    protected double multithread_inference() {
        int iter = 0;
        double perplexity = 0, totalWords = 0, last = -1.0, converge;

        //clear up for adding new testing documents
        for (int i = 0; i < m_workers.length; i++) {
            m_workers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_workers[i].clearCorpus();
        }

        //evenly allocate the testing work load
        int workerID = 0;
        for (_Doc d : m_testSet) {
            m_workers[workerID % m_workers.length].addDoc(d);
            workerID++;
        }

        do {
            init();
            perplexity = 0.0;
            totalWords = 0;

            // doc
            for (int i = 0; i < m_workers.length; i++) {
                m_threadpool[i] = new Thread(m_workers[i]);
                m_threadpool[i].start();
            }

            //wait till all finished
            for (Thread thread : m_threadpool) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            for (TopicModelWorker worker : m_workers) {
                perplexity += ((Doc_worker) worker).getPerplexity();
                totalWords += ((Doc_worker) worker).getTotalWords();
            }

            if(Double.isNaN(perplexity) || Double.isInfinite(perplexity)){
                System.err.format("[Error]Inference generate NaN\n");
                break;
            }

            if(iter > 0)
                converge = Math.abs((perplexity - last) / last);
            else
                converge = 1.0;

            last = perplexity;
            System.out.format("[Inference]Likelihood: %.2f\n", last);
            if(converge < m_varConverge)
                break;
        }while(iter++ < m_varMaxIter);

        return perplexity;
    }

    @Override
    // evaluation in multi-thread
    public double evaluation() {

        double allLoglikelihood = 0;
        int totalWords = 0;
        multithread_inference();

        for(TopicModelWorker worker:m_workers) {
            allLoglikelihood += worker.getPerplexity();
            totalWords += worker.getTotalWords();
        }

        double perplexity = Math.exp(-allLoglikelihood/totalWords);
        double avgLoglikelihood = allLoglikelihood / m_testSet.size();

        System.out.format("[Stat]TestInferIter=%d, perplexity=%.4f, totalWords=%d, allLogLikelihood=%.4f" +
                        ", avgLogLikelihood=%.4f\n\n",
                m_testInferMaxIter, perplexity, totalWords, allLoglikelihood, avgLoglikelihood);
        return perplexity;
    }

}
