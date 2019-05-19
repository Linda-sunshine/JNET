package Classifier.supervised.modelAdaptation.DirichletProcess;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import cc.mallet.types.SparseVector;
import clustering.KMeansAlg;
import structures._Doc;
import structures._Doc.rType;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._SparseFeature;
import utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class MTCLinAdaptWithDPKMeans extends MTCLinAdaptWithDPLR {
	double[] m_trainPerf;
	ArrayList<double[]> m_perfs = new ArrayList<double[]>();
	ArrayList<double[]> m_trainPerfs = new ArrayList<double[]>();
	
	public MTCLinAdaptWithDPKMeans(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithDPKMeans[dim:%d,supDim:%d,base:%d,threshold:%.1f,M:%d,alpha:%.4f,#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]", m_dim, m_dimSup, m_base, m_threshold, m_M, m_alpha, m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
	protected void calculateClusterProbPerUser(){
		_DPAdaptStruct user;

		constructCentroids();
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if(r.getType() == rType.ADAPTATION) 
					continue;
				r.setClusterNo(findMinClusterIndex(r));
			}
		}
	}
	
	// We split each large cluster into several small clusters.
	// Then find the sub-cluster which has the min-distance with the review.
	public int findMinClusterIndex(_Review r){
		int maxIndex = 0;
		_SparseFeature[][] subCentroids;
		double cos = 0, globalMax = Double.MIN_VALUE, localMax = Double.MIN_VALUE;
		
		// Iterate over all the large clusters.
		for(int in: m_centMap.keySet()){	
			subCentroids = m_centMap.get(in);
			localMax = Double.MIN_VALUE;
			for(int i=0; i<subCentroids.length; i++){
				cos = Utils.cosine(subCentroids[i], r.getLMSparse());
				if(cos > localMax)
					localMax = cos;
			}
			if(localMax > globalMax)
				maxIndex = in;
		}
		return maxIndex;
	}

	// Parameters used in the kmeans.
	KMeansAlg m_kmeans;
	int m_base = 100; double m_threshold = 0.2;
	HashMap<Integer, _SparseFeature[][]> m_centMap = new HashMap<Integer, _SparseFeature[][]>();

	public void setBaseThreshold(int base, double th){
		m_base = base;
		m_threshold = th;
	}

	public void constructCentroids(){
		HashMap<Integer, ArrayList<_Doc>> map = new HashMap<Integer, ArrayList<_Doc>>();
		// Construct reviews for each cluster.
		for(_AdaptStruct u: m_userList){
			_DPAdaptStruct user = (_DPAdaptStruct)u;
			int index = user.getThetaStar().getIndex();
			if(!map.containsKey(index))
				map.put(index, new ArrayList<_Doc>());
			for(_Review r: u.getReviews()){ 
				if(r.getType() == rType.ADAPTATION)
					map.get(index).add(r);
			}
		}
		// Construct subcluster inside each cluster.
		int k = 0; double[] ws;
		HashSet<Integer> indices = new HashSet<Integer>();
		for(int in: map.keySet()){
			indices.clear();
			k = map.get(in).size()/m_base + 1;
			ws = calcClusterWeights(m_thetaStars[in].getModel());
			for(int i=1; i<ws.length; i++){
				if(Math.abs(ws[i]) > m_threshold)
					indices.add(i-1);
			}
			// after collecting the indices, we represent docs with these indices.
			for(_Doc d: map.get(in)){
				d.filterIndicesValues(indices);
			}
			m_centMap.put(in, kmeans(map.get(in), k, indices));
		}
	}
	
	// We only have the vector a and b after mle, calculate the full weight vector.
	public double[] calcClusterWeights(double[] As){
		int ki, ks;
		double[] pWeights = new double[m_gWeights.length];
		for(int n=0; n<=m_featureSize; n++){
			ki = m_featureGroupMap[n];
			ks = m_featureGroupMap4SupUsr[n];
			pWeights[n] = As[ki]*(m_supModel[ks]*m_gWeights[n] + m_supModel[ks+m_dimSup])+As[ki+m_dim];
		}	
		return pWeights;
	}
	
	// kmeans applied on one cluster's reviews.
	public _SparseFeature[][] kmeans(ArrayList<_Doc> rs, int k, Set<Integer> indices){
		m_kmeans = new KMeansAlg(m_classNo, m_featureSize, k);
		m_kmeans.train(rs);
		
		_SparseFeature[][] subCentroids = new _SparseFeature[k][];
		ArrayList<SparseVector> spVcts = m_kmeans.getCentroids();
		for(int i=0; i<k; i++){
			subCentroids[i] = transfer(spVcts.get(i), indices);
		}		
		return subCentroids;
	}
	
	public _SparseFeature[] transfer(SparseVector sv, Set<Integer> indices){
		ArrayList<_SparseFeature> ins = new ArrayList<_SparseFeature>();
		int[] indexes = sv.getIndices();
		double[] values = sv.getValues();
		for(int i=0; i<sv.getIndices().length; i++){
			if(indices.contains(indexes[i]))
				ins.add(new _SparseFeature(indexes[i], values[i]));
		}
		return ins.toArray(new _SparseFeature[0]);
 	}
	
	//apply current model in the assigned clusters to users
	@Override
	protected void evaluateModel() {
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
		
		System.out.println("[Info]Accumulating evaluation results during sampling...");

		//calculate cluster posterior p(c|u)
		calculateClusterProbPerUser();
		
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();		
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_DPAdaptStruct user;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_DPAdaptStruct)m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
								
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									evaluate(r); // evoke user's own model
								}
							}							
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core, int numOfCores) {
					this.core = core;
					this.numOfCores = numOfCores;
					return this;
				}
			}).initialize(k, numberOfCores));
			
			threads.get(k).start();
		}
		
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}	
	}
	
	@Override
	// Each review corresponds with one cluster, so use that cluster model to predict.
	void evaluate(_Review r){
		int n, m;				
		double[] As = CLRWithDP.m_thetaStars[r.getClusterNo()].getModel();
		double prob = As[0]*CLinAdaptWithDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: r.getSparse()){
			n = fv.getIndex() + 1;
			m = m_featureGroupMap[n];
			prob += (As[m]*CLinAdaptWithDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
		}
	
		//accumulate the prediction results during sampling procedure
		r.m_pCount ++;
		r.m_prob += prob; //>0.5?1:0;
	}
}
