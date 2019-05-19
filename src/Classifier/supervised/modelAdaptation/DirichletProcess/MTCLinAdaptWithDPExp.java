package Classifier.supervised.modelAdaptation.DirichletProcess;

import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import cc.mallet.types.SparseVector;
import clustering.KMeansAlg;
import structures.*;
import structures._Doc.rType;
import utils.Utils;

import java.util.*;

public class MTCLinAdaptWithDPExp extends MTCLinAdaptWithDP {
	double[] m_trainPerf;
	ArrayList<double[]> m_perfs = new ArrayList<double[]>();
	ArrayList<double[]> m_trainPerfs = new ArrayList<double[]>();
	
	public MTCLinAdaptWithDPExp(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
	}
	
	public MTCLinAdaptWithDPExp(int classNo, int featureSize, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, globalModel, featureGroupMap,
				featureGroup4Sup);
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithDPExp[dim:%d,supDim:%d,base:%d,threshold:%.1f,M:%d,alpha:%.4f,#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]", m_dim, m_dimSup, m_base, m_threshold, m_M, m_alpha, m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	
	/*****Cross validation ****/
	public ArrayList<ArrayList<_Review>> collectClusterRvws(){
		HashMap<Integer, ArrayList<_Review>> clusters = new HashMap<Integer, ArrayList<_Review>>();
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct)m_userList.get(i);
			int index = user.getThetaStar().getIndex();
			if(!clusters.containsKey(index))
				clusters.put(index, new ArrayList<_Review>());
			
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				clusters.get(index).add(r);
			}
		}
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(clusters.size());
		for(int index: clusters.keySet()){
			// sort the clusters of reviews in descending order.
			queue.add(new _RankItem(index, clusters.get(index).size()));
		}
		ArrayList<ArrayList<_Review>> sortedClusters = new ArrayList<ArrayList<_Review>>();
		for(_RankItem it: queue){
			sortedClusters.add(clusters.get(it.m_index));
		}
		
		System.out.println(String.format("Collect %d clusters of reviews for sanity check.\n", clusters.size()));
		return sortedClusters;
	}
	
	SVM m_svm;// added by Lin, svm for cross validation.
	public void CrossValidation(int kfold, int threshold){
		ArrayList<ArrayList<_Review>> sortedClusters = collectClusterRvws();
		ArrayList<double[]> prfs = new ArrayList<double[]>();
		ArrayList<Integer> sizes = new ArrayList<Integer>();
		
		// Initialize the svm for training purpose.
		m_svm = new SVM(m_classNo, m_featureSize, 1);
		for(ArrayList<_Review> cluster: sortedClusters){
			if(cluster.size() > threshold){
				sizes.add(cluster.size());
				prfs.add(CV4OneCluster(cluster, kfold));
			}
		}
		System.out.print("Size\tNeg:Precision\tRecall\t\tF1\t\tPos:Precision\tRecall\t\tF1\n");
		for(int i=0; i<prfs.size(); i++){
			double[] prf = prfs.get(i);
			System.out.print(String.format("%d\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\n", 
					sizes.get(i), prf[0], prf[6], prf[1], prf[7], prf[2], prf[8],
					prf[3], prf[9], prf[4], prf[10], prf[5], prf[11]));
		}
		System.out.println(sortedClusters.size() + " clusters in total!");

	} 
	
	public double[] CV4OneCluster(ArrayList<_Review> reviews, int kfold){
		Random r = new Random();
		int[] masks = new int[reviews.size()];
		// Assign the review fold index first.
		for(int i=0; i<reviews.size(); i++){
			masks[i] = r.nextInt(kfold);
		}
		ArrayList<_Doc> trainSet = new ArrayList<_Doc>();
		ArrayList<_Doc> testSet = new ArrayList<_Doc>();
		double[][] prfs = new double[kfold][6];
		double[] AvgVar = new double[12];
		for(int k=0; k<kfold; k++){
			trainSet.clear();
			testSet.clear();
			for(int j=0; j<reviews.size(); j++){
				if(masks[j] == k)
					testSet.add(reviews.get(j));
				else
					trainSet.add(reviews.get(j));
			}
			m_svm.train(trainSet);
			prfs[k] = test(testSet);
			// sum over all the folds 
			for(int j=0; j<6; j++)
				AvgVar[j] += prfs[k][j];
		}
		// prfs[k]: avg. calculate the average performance among different folds.
		for(int j=0; j<6; j++)
			AvgVar[j] /= kfold;
		// prfs[k+1]: var. calculate the variance among different folds.
		for(int j=0; j<6; j++){
			for(int k=0; k<kfold; k++){
				AvgVar[j+6] += (prfs[k][j] - AvgVar[j]) * (prfs[k][j] - AvgVar[j]);
			}
			AvgVar[j+6] = Math.sqrt(AvgVar[j+6]/kfold);
		}
		return AvgVar;
	}
	
	public double[] test(ArrayList<_Doc> testSet){
		double[][] TPTable = new double[m_classNo][m_classNo];
		for(_Doc doc: testSet){
			int pred = m_svm.predict(doc), ans = doc.getYLabel();
			TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
		}
		
		double[] prf = new double[6];
		for (int i = 0; i < m_classNo; i++) {
			prf[3*i] = (double) TPTable[i][i] / (Utils.sumOfRow(TPTable, i) + 0.00001);// Precision of the class.
			prf[3*i + 1] = (double) TPTable[i][i] / (Utils.sumOfColumn(TPTable, i) + 0.00001);// Recall of the class.
			prf[3*i + 2] = 2 * prf[3 * i] * prf[3 * i + 1] / (prf[3 * i] + prf[3 * i + 1] + 0.00001);
		}
		return prf;
	}
	
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
	protected void calculateClusterProbPerUser(){
		double prob;
		_DPAdaptStruct user;
		double[] probs = new double[m_kBar];
		_thetaStar oldTheta;

		// calculate the centroids of all the clusters.
		calculateCentroids();

//		constructCentroids();
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
				
			oldTheta = user.getThetaStar();
			for(int k=0; k<m_kBar; k++){
				user.setThetaStar(m_thetaStars[k]);

				prob = calcDistance(k, user) + calcLogLikelihood(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
//				prob = calcLogLikelihood4Posterior(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment

				probs[k] = Math.exp(prob);//this will be in real space!
			}
			Utils.L1Normalization(probs);
			user.setClusterPosterior(probs);
			user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
		}
	}
	
//	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
//	protected void calculateClusterProbPerUser(){
//		double prob;
//		_DPAdaptStruct user;
//		double[] probs = new double[m_kBar];
//		_thetaStar oldTheta;
//
//		for(int i=0; i<m_userList.size(); i++){
//			user = (_DPAdaptStruct) m_userList.get(i);
//			
//			oldTheta = user.getThetaStar();
//			for(int k=0; k<m_kBar; k++){
//				user.setThetaStar(m_thetaStars[k]);
//
//				prob = calcLogLikelihood4Posterior(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
//				probs[k] = Math.exp(prob);//this will be in real space!
//			}
//			Utils.L1Normalization(probs);
//			user.setClusterPosterior(probs);
//			user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
//		}
//	}
	
//	// Assign cluster assignment to each user.
//	protected void initThetaStars(){
//		initPriorG0();
//		
//		m_pNewCluster = Math.log(m_alpha) - Math.log(m_M);//to avoid repeated computation
//		_DPAdaptStruct user;
//		for(int i=0; i<m_userList.size(); i++){
//			user = (_DPAdaptStruct) m_userList.get(i);
//			if(user.getAdaptationSize() >= 1)
//				sampleOneInstance(user);
//		}		
//	}
	KMeansAlg m_kmeans;
	HashMap<Integer, _SparseFeature[][]> m_centMap = new HashMap<Integer, _SparseFeature[][]>();

	// calculate the distance between the test review and the subcentroids.
//	public double calcDistance(int k, _AdaptStruct user){
//		_SparseFeature[][] subCentroids = m_centMap.get(k);
//		_SparseFeature[] subCentroid;
//		double[] ds = new double[subCentroids.length];
//		for(int i=0; i<ds.length; i++){
//			subCentroid = subCentroids[i];
//			for(_Review r: user.getReviews()){
//				if(r.getType() == rType.TEST)
//					ds[i] += Utils.cosine(subCentroid, r.getSparse());
//			}
//		}
//		return Math.log(ds[Utils.minOfArrayIndex(ds)]);
//	}

//	// The distance between the user and the kth thetastar, sum_{cosine(r, thetestar_k)}
//	public double calcDistance(int k, _AdaptStruct user){
//		double distance = 0;
//		for(_Review r: user.getReviews()){
//			if(r.getType() == rType.TEST){
//				distance += cosDistance(r, k);
//			}
//		}
//		return distance;
//	}
	
	public double cosDistance(_Review r, int k){
		double sum = 0;
		for(_SparseFeature sf: r.getLMSparse()){
			sum += m_centroids[k][sf.getIndex()]*sf.getValue();
		}
		return sum / (Utils.sumOfFeaturesL2(r.getLMSparse()) * Utils.sumOfFeaturesL2(m_centroids[k]));
	}
	
	// The distance between the user and the kth thetastar, weighted by the model weight.
	public double calcDistance(int k, _AdaptStruct user){
		double distance = 0;
		double[] ws = calcClusterWeights(m_thetaStars[k].getModel());
		for(_Review r: user.getReviews()){
			if(r.getType() == rType.TEST){
				distance += weightedCosine(r, k, ws);
			}
		}
		return distance;
	}
	
	public double weightedCosine(_Review r, int k, double[] ws){
		double sum = 0, wsum = 0;
		for(_SparseFeature sf: r.getSparse()){
			sum += m_centroids[k][sf.getIndex()]*sf.getValue()*ws[sf.getIndex()+1];
			wsum += ws[sf.getIndex()+1]*ws[sf.getIndex()+1]*sf.getValue()*sf.getValue();
		}
		return sum / Math.sqrt(wsum) * Utils.sumOfFeaturesL2(m_centroids[k]);		
	}

	int m_base = 100; double m_threshold = 0.2;
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
	
	/****We treat each cluster as one cluster***/
	double[][] m_centroids;
	// Calculate the centroids of the clusters.
	public void calculateCentroids(){
		m_centroids = new double [m_kBar][m_featureSize];
		double[] m_counts = new double[m_kBar];
		for(_AdaptStruct u: m_userList){
			_DPAdaptStruct user = (_DPAdaptStruct)u;
			int index = user.getThetaStar().getIndex();
			for(_Review r: u.getReviews()){ 
				if(r.getType() == rType.ADAPTATION){
					for(_SparseFeature sf: r.getLMSparse()){
						m_centroids[index][sf.getIndex()] += sf.getValue();
					}
					m_counts[index]++;
				}
			}
		}
		for(int k=0; k<m_kBar; k++){
			for(int j=0; j<m_featureSize; j++){
				m_centroids[k][j] /= m_counts[k]; 
			}
		}
	}
	
	//codes for incorporating the content of reviews.
	HashMap<_thetaStar, ArrayList<_DPAdaptStruct>> m_thetaUserMap = new HashMap<_thetaStar, ArrayList<_DPAdaptStruct>>();
	public void constructClusterMap(){
		for(int i=0; i<m_kBar; i++){
			m_thetaUserMap.put(m_thetaStars[i], new ArrayList<_DPAdaptStruct>());
		}
		for(_AdaptStruct u: m_userList){
			_DPAdaptStruct user = (_DPAdaptStruct) u;
			m_thetaUserMap.get(user.getThetaStar()).add(user);
		}
	}
	
	/****The code is used for generating peformance matrix.***/
	// we want to test if one cluster's model works better than others.
	public void sanityCheck(int k){
		setSupModel();
		// we first collect the test review size for each hdpthetastar.
		_DPAdaptStruct user;
		HashMap<Integer, ArrayList<_Review>> indexRvwMap = new HashMap<Integer, ArrayList<_Review>>();
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			int index = user.getThetaStar().getIndex();
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.TEST)
					continue;
				if(!indexRvwMap.containsKey(index))
					indexRvwMap.put(index, new ArrayList<_Review>());
				indexRvwMap.get(index).add(r);
			}
		}
		MyPriorityQueue<_RankItem> q = new MyPriorityQueue<_RankItem>(k);
		for(int in: indexRvwMap.keySet()){
			q.add(new _RankItem(in, indexRvwMap.get(in).size()));
		}
		ArrayList<_RankItem> rq = new ArrayList<_RankItem>();
		for(_RankItem it: q)
			rq.add(it);
		Collections.sort(rq, new Comparator<_RankItem>(){
			@Override
			public int compare(_RankItem r1, _RankItem r2){
				return (int) (r2.m_value - r1.m_value);
			}
		});
		int[] indexes = new int[rq.size()];
		for(int i=0; i<rq.size(); i++)
			indexes[i] = rq.get(i).m_index;
		double[][][] perf = new double[k][k][2];
		int i = 0;// thetastar[k]
		for(int in: indexes){
			int j = 0;
			_thetaStar theta = m_thetaStars[in];
			System.out.print(indexRvwMap.get(in).size() + "\t");
			for(int subin: indexes){
				perf[i][j] = calcPerf(theta, indexRvwMap.get(subin));
				System.out.print(String.format("%.4f/%.4f\t", perf[i][j][0], perf[i][j][1]));
				j++;
			}
			System.out.println();
			i++;
		}
	}
	
	public double[] calcPerf(_thetaStar theta, ArrayList<_Review> rs){
		int[][] TPTable = new int[m_classNo][m_classNo];
		for(_Review r: rs){
			int predL = predict(theta, r);
			int trueL = r.getYLabel();
			TPTable[predL][trueL]++;
		}
		double[] prf = new double[6];
		for (int i = 0; i < m_classNo; i++) {
			prf[3*i] = (double) TPTable[i][i] / (Utils.sumOfRow(TPTable, i) + 0.00001);// Precision of the class.
			prf[3*i + 1] = (double) TPTable[i][i] / (Utils.sumOfColumn(TPTable, i) + 0.00001);// Recall of the class.
			prf[3*i + 2] = 2 * prf[3 * i] * prf[3 * i + 1] / (prf[3 * i] + prf[3 * i + 1] + 0.00001);
		}
		return new double[]{prf[2], prf[5]};
	}
	
	public void setSupModel(){
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
	}
	public int predict(_thetaStar theta, _Review r){
		
		double[] As = theta.getModel();
		double prob, sum = As[0]*m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
		int m, n;
		for(_SparseFeature fv: r.getSparse()){
			n = fv.getIndex() + 1;
			m = m_featureGroupMap[n];
			sum += (As[m]*m_supWeights[n] + As[m_dim+m]) * fv.getValue();
		}
		prob = Utils.logistic(sum);
		return prob > 0.5 ? 1 : 0;
	}
	
	
//	@Override
//	public double test(){
//		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		ArrayList<Thread> threads = new ArrayList<Thread>();
//		
//		for(int k=0; k<numberOfCores; ++k){
//			threads.add((new Thread() {
//				int core, numOfCores;
//				public void run() {
//					_AdaptStruct user;
//					_PerformanceStat userPerfStat;
//					try {
//						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
//							user = m_userList.get(i+core);
//							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//								continue;
//								
//							userPerfStat = user.getPerfStat();								
//							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
//								//record prediction results
//								for(_Review r:user.getReviews()) {
//									if (r.getType() != rType.TEST)
//										continue;
//									int trueL = r.getYLabel();
//									int predL = user.predict(r); // evoke user's own model
//									userPerfStat.addOnePredResult(predL, trueL);
//								}
//							}							
//							userPerfStat.calculatePRF();	
//						}
//					} catch(Exception ex) {
//						ex.printStackTrace(); 
//					}
//				}
//				
//				private Thread initialize(int core, int numOfCores) {
//					this.core = core;
//					this.numOfCores = numOfCores;
//					return this;
//				}
//			}).initialize(k, numberOfCores));
//			
//			threads.get(k).start();
//		}
//		
//		for(int k=0;k<numberOfCores;++k){
//			try {
//				threads.get(k).join();
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			} 
//		}
//		double[] macroF1L = new double[m_classNo];
//		double[] macroF1M = new double[m_classNo];
//		double[] macroF1H = new double[m_classNo];
//		int cl = 0, cm = 0, ch = 0;
//		_PerformanceStat userPerfStat;
//		m_microStat.clear();
//		for(_AdaptStruct user:m_userList) {
//			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//				continue;
//			
//			userPerfStat = user.getPerfStat();
//			if(user.getUser().getReviewSize() <= 10){
//				cl++;
//				for(int i=0; i<m_classNo; i++)
//					macroF1L[i] += userPerfStat.getF1(i);
//			} else if(user.getUser().getReviewSize() <= 50){
//				cm++;
//				for(int i=0; i<m_classNo; i++)
//					macroF1M[i] += userPerfStat.getF1(i);
//			} else{
//				ch++;
//				for(int i=0; i<m_classNo; i++)
//					macroF1H[i] += userPerfStat.getF1(i);
//			}
//		}
//		
//		System.out.print(String.format("[Light]%.4f,%.4f,[medium]%.4f,%.4f,[heavy]%.4f,%.4f", 
//				macroF1L[0]/cl,macroF1L[1]/cl, macroF1M[0]/cm,macroF1M[1]/cm, macroF1H[0]/ch, macroF1H[1]/ch));
//		return 0;
//	}

}
