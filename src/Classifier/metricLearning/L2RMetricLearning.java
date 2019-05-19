package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;

import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.SolverType;
import Ranker.LambdaRank;
import Ranker.LambdaRank.OptimizationType;
import Ranker.LambdaRankParallel;
import Ranker.RankNet;
import Ranker.evaluator.Evaluator;
import Ranker.evaluator.MAP_Evaluator;
import Ranker.evaluator.NDCG_Evaluator;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._QUPair;
import structures._Query;
import structures._RankItem;
import utils.Utils;

public class L2RMetricLearning extends GaussianFieldsByRandomWalk {
	
	int m_topK;//top K initial ranking results 
	double m_noiseRatio; // to what extend random neighbors can be added 
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples

	double[] m_weights;
	double m_tradeoff;
	boolean m_multithread = false; // by default we will use single thread
	
	int m_ranker = 1; // 0: pairwise rankSVM; 1: LambdaRank; 2: RankNet
	ArrayList<_Query> m_queries = new ArrayList<_Query>();
	final int RankFVSize = 10;// features to be defined in genRankingFV()
	double[] m_mean, m_std; // to normalize the ranking features
	
	public L2RMetricLearning(_Corpus c, String classifier, double C, int topK) {
		super(c, classifier, C);
		m_topK = topK;
		m_noiseRatio = 0.0; // no random neighbor is needed 
		m_tradeoff = 1.0;
	}

	
	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean weightedAvg,
			int topK, double noiseRatio, boolean multithread) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				weightedAvg);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
		m_tradeoff = 1.0; // should be specified by the user
		m_multithread = multithread;
	}
	
	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean weightedAvg,
			int topK, double noiseRatio, int ranker, boolean multithread) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				weightedAvg);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
		m_tradeoff = 1.0; // should be specified by the user
		m_multithread = multithread;
		
		m_ranker = ranker;
		m_queryRatio = 1.0;
		m_documentRatio = 2;
	}
	@Override
	public String toString() {
		String ranker;
		if (m_ranker==0)
			ranker = "RankSVM";
		else
			ranker = "LambdaRank@MAP";
		return String.format("%s-[%s]", super.toString(), ranker);
	}
	
	//NOTE: this similarity is no longer symmetric!!
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		double similarity = Utils.dotProduct(m_weights, normalize(genRankingFV(di, dj)));
		if (Double.isNaN(similarity)){
			System.out.println("similarity calculation hits NaN!");
			System.exit(-1);
		} else if (Double.isInfinite(similarity)){
			System.out.println("similarity calculation hits infinite!");
			System.exit(-1);
		} 
		
		return Math.exp(similarity);//to make sure it is positive			
	}
	
	@Override
	protected void init() {
		super.init();
		
		if (m_queries==null)
			m_queries = new ArrayList<_Query>();
		else
			m_queries.clear();
	}

	@Override
	public double train(Collection<_Doc> trainSet) {
		super.train(trainSet);
		
		L2RModelTraining();
		
		return 0;
	}
	
	protected void L2RModelTraining() {
		//select the training pairs
		createTrainingCorpus();
		
		if (m_ranker==0) {
			ArrayList<Feature[]> fvs = new ArrayList<Feature[]>();
			ArrayList<Integer> labels = new ArrayList<Integer>();
			
			for(_Query q:m_queries)
				q.extractPairs4RankSVM(fvs, labels);
			Model rankSVM = SVM.libSVMTrain(fvs, labels, RankFVSize, SolverType.L2R_L1LOSS_SVC_DUAL, m_tradeoff, -1);
			
			m_weights = rankSVM.getFeatureWeights();	
			System.out.format("RankSVM training performance:\nMAP: %.4f\n", evaluate(OptimizationType.OT_MAP));

		} else if (m_ranker==1) {//all the rest use LambdaRank with different evaluator
			LambdaRank lambdaRank;
			if (m_multithread) {
				/**** multi-thread version ****/
				lambdaRank = new LambdaRankParallel(RankFVSize, m_tradeoff, m_queries, OptimizationType.OT_MAP, 10);
				lambdaRank.setSigns(getRankingFVSigns());
				lambdaRank.train(100, 100, 1.0, 0.95);//lambdaRank specific parameters
			} else {
				/**** single-thread version ****/
				lambdaRank = new LambdaRank(RankFVSize, m_tradeoff, m_queries, OptimizationType.OT_MAP);
				lambdaRank.setSigns(getRankingFVSigns());
				lambdaRank.train(300, 20, 1.0, 0.98);//lambdaRank specific parameters
			}			
			m_weights = lambdaRank.getWeights();
		} else if (m_ranker==2) {
			RankNet ranknet = new RankNet(RankFVSize, 5.0);
			ArrayList<double[]> rfvs = new ArrayList<double[]>();
			for(_Query q:m_queries)
				q.extractPairs4RankNet(rfvs);
			ranknet.setSigns(getRankingFVSigns());
			double likelihood = ranknet.train(rfvs);
			m_weights = ranknet.getWeights();
			
			System.out.format("RankNet training performance:\nlog-likelihood: %.4f\t MAP: %.4f\n", likelihood, evaluate(OptimizationType.OT_MAP));
		}		
		
		for(int i=0; i<RankFVSize; i++)
			System.out.format("%.5f ", m_weights[i]);
		System.out.println();
	}
	
	double evaluate (OptimizationType otype) {
		Evaluator eval;
		
		if (otype.equals(OptimizationType.OT_MAP))
			eval = new MAP_Evaluator();
		else if (otype.equals(OptimizationType.OT_NDCG))
			eval = new NDCG_Evaluator(LambdaRank.NDCG_K);
		else
			eval = new Evaluator();
		
		double perf = 0;
		for(_Query q:m_queries) {
			for(_QUPair qu:q.m_docList)
				qu.score(m_weights);
			
			perf += eval.eval(q);
		}
		return perf/m_queries.size();
	}
	
	//this is an important feature and will be used repeated
	private void calcLabeledSimilarities() {
		System.out.println("Creating cache for labeled documents...");
		
		int L = m_trainSet.size(), size = L*(L-1)/2;//no need to compute diagonal
		if (m_LabeledCache==null || m_LabeledCache.length<size)
			m_LabeledCache = new double[size];
		
		//using Collection<_Doc> trainSet to pass corpus parameter is really awkward
		_Doc di, dj;
		for(int i=1; i<m_trainSet.size(); i++) {
			di = m_trainSet.get(i);
			for(int j=0; j<i; j++) {
				dj = m_trainSet.get(j);
				m_LabeledCache[getIndex(i,j)] = super.getSimilarity(di, dj);
			}
		}
	}
	
	int getIndex(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return i*(i-1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
 	
	//In this training process, we want to get the weight of all pairs of samples.
	protected int createTrainingCorpus(){
		//pre-compute the similarity between labeled documents
		calcLabeledSimilarities();
		
		MyPriorityQueue<_RankItem> simRanker = new MyPriorityQueue<_RankItem>(m_topK);
		ArrayList<_Doc> neighbors = new ArrayList<_Doc>();
		
		_Query q;		
		_Doc di, dj;
		int posQ = 0, negQ = 0, pairSize = 0;
		int relevant = 0, irrelevant = 0;
		
		for(int i=0; i<m_trainSet.size(); i++) {
			//candidate query document
			di = m_trainSet.get(i);
			relevant = 0;
			irrelevant = 0;
			
			//using content similarity to construct initial ranking
			for(int j=0; j<m_trainSet.size(); j++) {
				if (i==j)
					continue;	
				dj = m_trainSet.get(j);
				simRanker.add(new _RankItem(j, m_LabeledCache[getIndex(i,j)]));
			}
			
			//find the top K similar documents by default similarity measure
			for(_RankItem it:simRanker) {
				dj = m_trainSet.get(it.m_index);
				neighbors.add(dj);
				if (di.getYLabel() == dj.getYLabel())
					relevant ++;
				else
					irrelevant ++;
			}
			
			//inject some random neighbors 
			int j = 0;
			while(neighbors.size()<(1.0+m_noiseRatio)*m_topK) {
				if (i!=j) {
					dj = m_trainSet.get(j);
					if (Math.random()<0.02 && !neighbors.contains(dj)) {
						neighbors.add(dj);
						if (di.getYLabel() == dj.getYLabel())
							relevant ++;
						else
							irrelevant ++;
					}
				}
				
				j = (j+1) % m_trainSet.size();//until we use up all the random budget 
			}
			
			if (relevant==0 || irrelevant==0 
				|| (di.getYLabel() == 1 && negQ < 1.1*posQ)){
				//clear the cache for next query
				simRanker.clear();
				neighbors.clear();
				continue;
			} else if (di.getYLabel()==1)
				posQ ++;
			else
				negQ ++;
				
			//accept the query
			q = new _Query();
			m_queries.add(q);
			
			//construct features for the most similar documents with respect to the query di
			for(_Doc d:neighbors)
				q.addQUPair(new _QUPair(d.getYLabel()==di.getYLabel()?1:0, genRankingFV(di, d)));
			pairSize += q.createRankingPairs();
			
			//clear the cache for next query
			simRanker.clear();
			neighbors.clear();
		}
		
		normalize();//normalize the features by z-score
		System.out.format("Generate %d(%d:%d) ranking pairs for L2R model training...\n", pairSize, posQ, negQ);
		return pairSize;
	}
	
	void normalize() {
		m_mean = new double[RankFVSize];
		m_std = new double[RankFVSize];
		
		double size = 0;
		for(_Query q:m_queries) {
			for(_QUPair qu:q.m_docList) {
				for(int i=0; i<RankFVSize; i++) {
					m_mean[i] += qu.m_rankFv[i];
					m_std[i] += qu.m_rankFv[i] * qu.m_rankFv[i];
					size ++;
				}
			}
		}
		
		for(int i=0; i<RankFVSize; i++) {
			m_mean[i] /= size;
			m_std[i] = Math.sqrt(m_std[i]/size - m_mean[i]*m_mean[i]);
		}
		
		for(_Query q:m_queries) {
			for(_QUPair qu:q.m_docList) {
				normalize(qu.m_rankFv);
			}
		}
	}
	
	double[] normalize(double[] fv) {
		for(int i=0; i<RankFVSize; i++)
			fv[i] = (fv[i] - m_mean[i]) / m_std[i];
		return fv;
	}
	
	int[] getRankingFVSigns() {
		int[] signs = new int[RankFVSize];
		signs[0] = 1;
		signs[1] = -1;
		signs[2] = 1;
		signs[3] = -1;
		signs[4] = 1;
		signs[5] = -1;
		signs[6] = 1;
		signs[7] = 1;
		signs[8] = -1;
		signs[9] = 1;
		return signs;
	}
	
	//generate ranking features for a query document pair
	double[] genRankingFV(_Doc q, _Doc d) {
		double[] fv = new double[RankFVSize];
		
		//Part I: pairwise features for query document pair
		//feature 1: cosine similarity
		fv[0] = getBoWSim(q, d);//0.04104
		
		//feature 2: topical similarity
		fv[1] = getTopicalSim(q, d);//-0.28595
		
		//feature 3: belong to the same product
		fv[2] = q.sameProduct(d)?1:0;//-0.01331

		//feature 4: sparse feature length difference
		fv[3] = Math.abs((double)(q.getDocLength() - d.getDocLength())/(double)q.getDocLength());//0.00045
		
		//feature 5: jaccard coefficient
		fv[4] = Utils.jaccard(q.getSparse(), d.getSparse());//0.05490
 		
		//feature 6: the sentiwordnet score for a review.
		fv[5] = Math.abs(q.getSentiScore() - d.getSentiScore());//-0.09206

		// feature 7: the pos tagging score for a pair of reviews.
		fv[6] = getPOSScore(q, d);//0.02567

		// feature 8: the aspect score for a pair of reviews.
		fv[7] = getAspectScore(q, d);//-0.03405
		
		//Part II: pointwise features for document
		//feature 9: stop words proportion
		fv[8] = d.getStopwordProportion();//-0.05709
		
		//feature 10: average IDF
		fv[9] = d.getAvgIDF();//0.05842

		return fv;
	}
	/***The following variable definitions and functions are added by Lin.***/
	double m_shrinkage=0.98;
	double m_stepSize=1;
	int m_maxIter = 300;
	int m_windowSize = 20;
	double m_queryRatio; // added by Lin, control the ratio of the class sentiment.
	double m_documentRatio; // added by Lin, controlt the ratio of the document selection for each query.
	ArrayList<ArrayList<_Doc>> m_clusters;
	
	// Get the trained weights from learning to rank.
	public double[] getWeights(){
		return m_weights;
	}
	// In lambdaRank, the tradeoff = lambda.added by Lin.
	public void setLambda(double lambda){
		m_tradeoff = lambda;
	}

	public void setShrinkage(double sk){
		m_shrinkage = sk;
	}

	public void setStepSize(double ss){
		m_stepSize = ss;
	}
		
	public void setWindowSize(int ws){
		m_windowSize = ws;
	}
		
	public void setMaxIter(int maxIter){
		m_maxIter = maxIter;
	}		
		
	public void setQueryRatio(double r){
		m_queryRatio = r;
	}
		
	public void setDocumentRatio(double r){
		m_documentRatio = r;
	}
}
