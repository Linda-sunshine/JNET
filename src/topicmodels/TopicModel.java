package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Review;
import structures._User;
import topicmodels.markovmodel.HTSM;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;
import topicmodels.multithreads.TopicModel_worker.RunType;
import utils.Utils;

public abstract class TopicModel {
	protected int number_of_topics;
	protected int vocabulary_size;
	protected double m_converge;//relative change in log-likelihood to terminate EM
	protected int number_of_iteration;//number of iterations in inferencing testing document
	protected _Corpus m_corpus;	
	
	protected boolean m_logSpace; // whether the computations are all in log space
	
	protected boolean m_LoadnewEggInTrain = true; // check whether newEgg will be loaded in trainSet or Not
	protected boolean m_randomFold = true; // true mean randomly take K fold and test and false means use only 1 fold and use the fixed trainset
	
	//for training/testing split
	protected ArrayList<_Doc> m_trainSet, m_testSet;
	protected double[][] word_topic_sstat; /* fractional count for p(z|d,w) */
	
	//smoothing parameter for p(w|z, \beta)
	protected double d_beta; 	
	
	protected int m_displayLap; // output EM iterations for every lap iterations (negative and zero means no verbose display)
	protected boolean m_collectCorpusStats; // if we will collect corpus-level statistics (for efficiency purpose)
	
	protected boolean m_multithread = false; // by default we do not use multi-thread mode
	protected Thread[] m_threadpool = null;
	protected TopicModelWorker[] m_workers = null;
	
	protected double m_testWord4PerplexityProportion;
	
	public PrintWriter infoWriter;
	public PrintWriter summaryWriter;
	public PrintWriter debugWriter;
	
	public TopicModel(int number_of_iteration, double converge, double beta, _Corpus c) {
		this.vocabulary_size = c.getFeatureSize();
		this.number_of_iteration = number_of_iteration;
		this.m_converge = converge;
		this.d_beta = beta;
		this.m_corpus = c;
		
		m_displayLap = 0; // by default we will not verbosely track EM iterations
	}
	
	@Override
	public String toString() {
		return "Topic Model";
	}
	
	public void setDisplayLap(int lap) {
		m_displayLap = lap;
	}
	
	public void setNewEggLoadInTrain(boolean flag){
		if(flag)
			System.out.println("NewEgg is added in Training Set");
		else
			System.out.println("NewEgg is NOT added in Training Set");
		m_LoadnewEggInTrain = flag;
	}
	
	public void setRandomFold(boolean flag){
		m_randomFold = flag;
	}
	
	public void setSummaryWriter(String path){
		System.out.println("Summary File Path: "+ path);
		try{
			summaryWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	public void setInforWriter(String path){
		System.out.println("Info File Path: "+ path);
		try{
			infoWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	public void setDebugWriter(String path){
		System.out.println("Debug File Path: "+ path);
		try{
			debugWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	public void closeWriter(){		
		if(summaryWriter!=null){
			summaryWriter.flush();
			summaryWriter.close();
		}
		
		if(debugWriter!=null){
			debugWriter.flush();
			debugWriter.close();
		}
		
		if(infoWriter!=null){
			infoWriter.flush();
			infoWriter.close();
		}
	}

	public void setCorpus(_Corpus c) { this.m_corpus = c; }

	public void setTrainSet(ArrayList<_Doc> trainset){ this.m_trainSet = trainset; }

	public void setTestSet(ArrayList<_Doc> testset) { this.m_testSet = testset; }

	//initialize necessary model parameters
	protected abstract void initialize_probability(Collection<_Doc> collection);	
	
	// to be called per EM-iteration
	protected abstract void init();
	
	protected abstract void initial();
	
	// to be called by the end of EM algorithm 
	protected abstract void finalEst();
	
	// to be call per test document
	protected abstract void initTestDoc(_Doc d);
	
	//estimate posterior distribution of p(\theta|d)
	protected abstract void estThetaInDoc(_Doc d);
	
	// perform inference of topic distribution in the document
	public double inference(_Doc d) {
		initTestDoc(d);//this is not a corpus level estimation
		
		double delta, last = 1, current;
		int  i = 0;
		do {
			current = calculate_E_step(d);
			estThetaInDoc(d);			
			
			delta = (last - current)/last;
			last = current;
		} while (Math.abs(delta)>m_converge && ++i<this.number_of_iteration);
		return current;
	}
		
	//E-step should be per-document computation
	public abstract double calculate_E_step(_Doc d); // return log-likelihood
	
	//M-step should be per-corpus computation
	public abstract void calculate_M_step(int i); // input current iteration to control sampling based algorithm
	
	//compute per-document log-likelihood
	protected abstract double calculate_log_likelihood(_Doc d);
	
	//print top k words under each topic
	public abstract void printTopWords(int k, String topWordPath);
	public abstract void printTopWords(int k);
	
	// compute corpus level log-likelihood
	protected double calculate_log_likelihood() {
		return 0;
	}
	
	public void EMonCorpus() {
		m_trainSet = m_corpus.getCollection();
		EM();
	}
	

	protected double multithread_E_step() {
		for(int i=0; i<m_workers.length; i++) {
			m_workers[i].setType(RunType.RT_EM);
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		//wait till all finished
		for(Thread thread:m_threadpool){
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		double likelihood = 0;
		for(TopicModelWorker worker:m_workers)
			likelihood += worker.accumluateStats(word_topic_sstat);
		return likelihood;
	}
	
	protected double multithread_inference() {
		//clear up for adding new testing documents
		for(int i=0; i<m_workers.length; i++) {
			m_workers[i].setType(RunType.RT_inference);
			m_workers[i].clearCorpus();
		}
		
		//evenly allocate the testing work load
		int workerID = 0;
		
		if(debugWriter==null){
			for(_Doc d:m_testSet) {
				m_workers[workerID%m_workers.length].addDoc(d);
				workerID++;
			}
		}else{
			for(_Doc d:m_corpus.getCollection()) {
				m_workers[workerID%m_workers.length].addDoc(d);
				workerID++;
			}
		}
			
		for(int i=0; i<m_workers.length; i++) {
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		//wait till all finished
		for(Thread thread: m_threadpool){
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		return 0;
	}

	public void EM() {
		System.out.format("[Info]EM Starting %s...\n", toString());
		
		long starttime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
//		double delta, last = calculate_log_likelihood(), current;
		double delta=0, last=0, current=0;
		int i = 0, displayCount = 0;
		do {
			init();
			
			if (m_multithread)
				current = multithread_E_step();
			else {
				current = 0;
				for(_Doc d:m_trainSet)
					current += calculate_E_step(d);
			}
			
			calculate_M_step(i);
			
//			if (m_converge>0 || (m_displayLap>0 && i%m_displayLap==0 && displayCount > 6)){//required to display log-likelihood
//				current += calculate_log_likelihood();//together with corpus-level log-likelihood
//
//				if (i>0)
//					delta = (last-current)/last;
//				else
//					delta = 1.0;
//				last = current;
//			}

			if (i>0)
				delta = (last-current)/last;
			else
				delta = 1.0;
			last = current;
			
			if (m_displayLap>0 && i%m_displayLap==0) {
				if (m_converge>0) {
				    System.out.println("==============");
					System.out.format("[Info]Likelihood %.5f at step %s converge to %.10f...\n", current, i, delta);
//					infoWriter.format("[Info]Likelihood %.5f at step %s converge to %.10f...\n", current, i, delta);

				} else {
					System.out.print(".");
					if (displayCount > 6){
						System.out.format("\t%d:%.3f\n", i, current);
//						infoWriter.format("\t%d:%.3f\n", i, current);
					}
					displayCount ++;
				}
			}

			if (m_converge>0 && Math.abs(delta)<m_converge)
				break;//to speed-up, we don't need to compute likelihood in many cases
		} while (++i<this.number_of_iteration);
		
		finalEst();
		
		long endtime = System.currentTimeMillis() - starttime;
		System.out.format("[Info]Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);
//		infoWriter.format("[Info]Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);
	}

	public double Evaluation() {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, log2 = Math.log(2.0), sumLikelihood = 0;
		double totalWords = 0.0;
		if (m_multithread) {
			multithread_inference();
			System.out.println("[Info]Start evaluation in thread...");
			for(TopicModelWorker worker:m_workers) {
				sumLikelihood += worker.getLogLikelihood();
				perplexity += worker.getPerplexity();
				totalWords += worker.getTotalWords();
			}
		} else {
			System.out.println("[Info]Start evaluation in Normal...");
			for(_Doc d:m_testSet) {				
				loglikelihood = inference(d);
				sumLikelihood += loglikelihood;
				perplexity += loglikelihood;
				totalWords += d.getTotalDocLength();
//				perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
			}
			
		}
        perplexity = Math.exp(-perplexity/totalWords);
		sumLikelihood /= m_testSet.size();
		
		if(this instanceof HTSM)
			calculatePrecisionRecall();

		System.out.format("[Stat]Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		
		return perplexity;
	}
	
	public void debugOutputWrite(){
		debugWriter.println("Doc ID, Source, SentenceIndex, ActualSentiment, PredictedSentiment");
		for(_Doc d:m_corpus.getCollection()){
			for(int i=0; i<d.getSenetenceSize(); i++){
				debugWriter.format("%d,%d,%d,%s,%d,%d\n", d.getID(),d.getSourceType(),i,d.getSentence(i).getRawSentence(),d.getSentence(i).getStnSentiLabel(),d.getSentence(i).getStnPredSentiLabel());
			}
		}
		debugWriter.flush();
		debugWriter.close();
	}
	
	public void calculatePrecisionRecall(){
		int[][] precision_recall = new int [2][2];
		precision_recall [0][0] = 0; // 0 is for pos
		precision_recall[0][1] = 0; // 1 is neg 
		precision_recall[1][0] = 0;
		precision_recall[1][1] = 0;
		
		int actualLabel, predictedLabel;
		
		for(_Doc d:m_testSet) {
			// if document is from newEgg which is 2 then calculate precision-recall
			if(d.getSourceType()==2){
				
				for(int i=0; i<d.getSenetenceSize(); i++){
					actualLabel = d.getSentence(i).getStnSentiLabel();
					predictedLabel = d.getSentence(i).getStnPredSentiLabel();
					precision_recall[actualLabel][predictedLabel]++;
				}
			}
		}
		
		System.out.println("Confusion Matrix");
		for(int i=0; i<2; i++)
		{
			for(int j=0; j<2; j++)
			{
				System.out.print(precision_recall[i][j]+",");
			}
			System.out.println();
		}
		
		double pros_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
		double cons_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);
		
		
		double pros_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
		double cons_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);
		
		System.out.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
		System.out.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);
		
		
		double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
		double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
		
		System.out.println("F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
	}
	
	public void setPerplexityProportion(double proportion){
		m_testWord4PerplexityProportion = proportion;
	}

	//k-fold Cross Validation.
	public void crossValidation(int k) {
        m_trainSet = new ArrayList<_Doc>();
        m_testSet = new ArrayList<_Doc>();
		
		double[] perf;
		int amazonTrainsetRatingCount[] = {0,0,0,0,0};
		int amazonRatingCount[] = {0,0,0,0,0};
		
		int newEggRatingCount[] = {0,0,0,0,0};
		int newEggTrainsetRatingCount[] = {0,0,0,0,0};		
		
		if(m_randomFold==true){
			perf = new double[k];
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
				EM();
				perf[i] = Evaluation();
                System.out.format("[Info]%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				m_trainSet.clear();
				m_testSet.clear();
			}
		} else {
			k = 1;
			perf = new double[k];
		    int totalNewqEggDoc = 0;
		    int totalAmazonDoc = 0;
			for(_Doc d:m_corpus.getCollection()){
				if(d.getSourceType()==2){
					newEggRatingCount[d.getYLabel()]++;
					totalNewqEggDoc++;
					}
				else if(d.getSourceType()==1){
					amazonRatingCount[d.getYLabel()]++;
					totalAmazonDoc++;
				}
			}
			System.out.println("Total New Egg Doc:"+totalNewqEggDoc);
			System.out.println("Total Amazon Doc:"+ totalAmazonDoc);
			
			int amazonTrainSize = 0;
			int amazonTestSize = 0;
			int newEggTrainSize = 0;
			int newEggTestSize = 0;
			
			for(_Doc d:m_corpus.getCollection()){
				
				if(d.getSourceType()==1){ // from Amazon
					int rating = d.getYLabel();
					
					if(amazonTrainsetRatingCount[rating]<=0.8*amazonRatingCount[rating]){
						m_trainSet.add(d);
						amazonTrainsetRatingCount[rating]++;
						amazonTrainSize++;
					}else{
						m_testSet.add(d);
						amazonTestSize++;
					}
				}
				
				if(m_LoadnewEggInTrain==true && d.getSourceType()==2) {
					
					int rating = d.getYLabel();
					if(newEggTrainsetRatingCount[rating]<=0.8*newEggRatingCount[rating]){
						m_trainSet.add(d);
						newEggTrainsetRatingCount[rating]++;
						newEggTrainSize++;
					}else{
						m_testSet.add(d);
						newEggTestSize++;
					}
					
				}
				if(m_LoadnewEggInTrain==false && d.getSourceType()==2) {
					int rating = d.getYLabel();
					if(newEggTrainsetRatingCount[rating]<=0.8*newEggRatingCount[rating]){
						// Do nothing simply ignore it make for similar for two different configurations (i.e. newEgg loaded and not loaded in trainset) set
						newEggTrainsetRatingCount[rating]++;
					}else{
						m_testSet.add(d);
						newEggTestSize++;
					}
				}
			}
			
			System.out.println("Neweeg Train Size: "+newEggTrainSize+" test Size: "+newEggTestSize);			
			System.out.println("Amazon Train Size: "+amazonTrainSize+" test Size: "+amazonTestSize);
			
			for(int i=0; i<amazonTrainsetRatingCount.length; i++){
				System.out.println("Rating ["+i+"] and Amazon TrainSize:"+amazonTrainsetRatingCount[i]+" and newEgg TrainSize:"+newEggTrainsetRatingCount[i]);
			}
	
			System.out.println("Combined Train Set Size "+m_trainSet.size());
			System.out.println("Combined Test Set Size "+m_testSet.size());
			
			long start = System.currentTimeMillis();
			EM();
			perf[0] = Evaluation();
			System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
			
		}
		//output the performance statistics
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("[Stat]Perplexity %.3f+/-%.3f\n", mean, var);
	}

    public double[] oneFoldValidation(){
        m_trainSet = new ArrayList<_Doc>();
        m_testSet = new ArrayList<_Doc>();
        for(_Doc d:m_corpus.getCollection()){
            if(d.getType() == _Doc.rType.TRAIN){
                m_trainSet.add(d);
            }else if(d.getType() == _Doc.rType.TEST){
                m_testSet.add(d);
            }
        }

        System.out.format("-- train size = %d, test size = %d....\n", m_trainSet.size(), m_testSet.size());

        long start = System.currentTimeMillis();
        EM();
        double[] results = EvaluationMultipleMetrics();
        System.out.format("[Info]%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);

        return results;
    }


	// fixed cross validation with specified fold number
	// added by Lin for debugging purpose
	public void fixedCrossValidation(ArrayList<_User> users, int k){
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();

		System.out.format("\n==========Start %d-fold cross validation=========\n", k);
		m_trainSet.clear();
		m_testSet.clear();
		for(int i=0; i<users.size(); i++){
			for(_Review r: users.get(i).getReviews()){
				if(r.getMask4CV() == k){
					r.setType(_Doc.rType.TEST);
					m_testSet.add(r);
				} else {
					r.setType(_Doc.rType.TRAIN);
					m_trainSet.add(r);
				}
			}
		}
		System.out.format("-- train size = %d, test size = %d....\n", m_trainSet.size(), m_testSet.size());
		EM();
		double[] results = EvaluationMultipleMetrics();
	}

    public double[] EvaluationMultipleMetrics() {
        m_collectCorpusStats = false;
        double perplexity = 0, loglikelihood, sumLikelihood = 0;
        double totalWords = 0.0;
        if (m_multithread) {
            multithread_inference();
            System.out.println("[Info]Start evaluation in thread...");
            for(TopicModelWorker worker:m_workers) {
                sumLikelihood += worker.getLogLikelihood();
//                perplexity += worker.getPerplexity();
                totalWords += worker.getTotalWords();
            }
        } else {
            System.out.println("[Info]Start evaluation in Normal...");
            for(_Doc d:m_testSet) {
                loglikelihood = inference(d);
//                System.out.println(loglikelihood);
                sumLikelihood += loglikelihood;
                perplexity += loglikelihood;
                totalWords += d.getTotalDocLength();
            }

        }
        double[] results = new double[2];
        results[0] = Math.exp(-sumLikelihood/totalWords);
        results[1] = sumLikelihood / m_testSet.size();

        if(this instanceof HTSM)
            calculatePrecisionRecall();
		System.out.format("[Stat]perplexity=%.4f, all-log-likelihood=%.4f, total_words=%.1f, avg_log-likelihood=%.4f\n\n",
				results[0], sumLikelihood, totalWords, results[1]);
        return results;
    }
}
