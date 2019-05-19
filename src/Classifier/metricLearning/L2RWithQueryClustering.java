package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import structures._Corpus;
import structures._Doc;
import utils.Utils;
/***
 * Learning to rank for different clusters of queries.
 * @author lin
 */
public class L2RWithQueryClustering extends L2RMetricLearning {
	
	int m_kmeans; //The number of clusters.
	double[][] m_allWeights; //Different rankSVM models for different clusters.
	HashMap<Integer, ArrayList<_Doc>> m_clusterNoDocs;
	
	public L2RWithQueryClustering(_Corpus c, String classifier, double C, 
								  double ratio, int k, int kPrime, double alhpa, double beta, 
								  double delta, double eta, boolean weightedAvg, int topK, 
								  double noiseRatio, int ranker, boolean multithread) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, 
		      eta, weightedAvg, topK, noiseRatio, ranker, multithread);
		m_clusterNoDocs = new HashMap<Integer, ArrayList<_Doc>>();
	}
	
	//Set the number of clusters.
	public void setClusterNo(int k){
		m_kmeans = k;
	}
	
	// Pass the clustering results back to L2R.
	public void setClusters(ArrayList<ArrayList<_Doc>> clusters){
		m_clusters = clusters;
	}
	
	public double train(Collection<_Doc> trainSet){
		
		super.init();
		m_classifier.train(m_trainSet);
		
		m_L = m_trainSet.size();
		m_U = m_testSet.size();
		m_labeled = m_trainSet;
		
		int clusterNo;
		
		//Init hashmap.
		for(int i=0; i<m_kmeans; i++)
			m_clusterNoDocs.put(i, new ArrayList<_Doc>());
		
		//Split the train set based on different clusters.
		for(_Doc d: trainSet){
			clusterNo = d.getClusterNo();
			m_clusterNoDocs.get(clusterNo).add(d);
		}
		
		//The model array stores all the rankSVM for all clusters.
		m_allWeights = new double[m_clusterNoDocs.size()][];
		
		//Train different cluster of documents respectively.
		for(int cNo: m_clusterNoDocs.keySet()){
			m_trainSet = m_clusterNoDocs.get(cNo);
			//Get some stat of training reviews.
			int[] count = new int[2];
			for(_Doc d: m_trainSet)
				count[d.getYLabel()]++;
			System.out.format("There are %d (pos:%d, neg:%d) training documents in the corpus.\n", m_trainSet.size(), count[1], count[0]);
			super.L2RModelTraining();
			m_allWeights[cNo] = getWeights();
		}
		return 0;
	}
	
	//NOTE: this similarity is no longer symmetric!!
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		
		int clusterNo = di.getClusterNo();
		double similarity = Utils.dotProduct(m_allWeights[clusterNo], normalize(genRankingFV(di, dj)));
		
		if (Double.isNaN(similarity)){
			System.out.println("similarity calculation hits NaN!");
			System.exit(-1);
		} else if (Double.isInfinite(similarity)){
			System.out.println("similarity calculation hits infinite!");
			System.exit(-1);
		} 
		return Math.exp(similarity);//to make sure it is positive			
	}
}
