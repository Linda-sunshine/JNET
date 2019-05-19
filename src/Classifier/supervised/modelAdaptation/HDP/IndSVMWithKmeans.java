/**
 * 
 */
package Classifier.supervised.modelAdaptation.HDP;

import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.SparseVector;
import clustering.KMeansAlg;
import structures._Doc;
import structures._Doc.rType;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._User;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

/**
 * @author lin
 * In this class, we utilize users' adaptation reviews for training kmeans clusters.
 * Then test each test review based on weighted avearge prediction from all clusters.
 */
public class IndSVMWithKmeans extends ModelAdaptation{
	
	int m_k;// k clusters.
	double[][] m_weights; // svm weights for each individual cluster.
	boolean m_label = true; // whether we use the predicted label as weighted average or the
	KMeansAlg m_kmeans;
	Collection<_Doc> m_trainSet; // training reviews.
	HashMap<Integer, ArrayList<_Doc>> m_cIndexRvws;
	
	public IndSVMWithKmeans(int classNo, int featureSize, int k) {
		super(classNo, featureSize);
		m_k = k;
		m_testmode = TestMode.TM_batch;

	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList)
			m_userList.add(new _AdaptStruct(user));
		
	}

	// train the individual svms based on kmeans.
	public double train(){
		// Cluster adaptation reviews.
		trainKMeans();
		// Train individual svms for each cluster.
		trainSVMs();
		// Calculate the weighted average over all clusters' predictions.
		calcWeightedAverage();
		
		return 0;
	}
	
	// train kmeans based on the adaptation reviews.
	public void trainKMeans(){
		// Step 1: collect all reviews.
		m_trainSet = new ArrayList<_Doc>();
		for(_AdaptStruct u: m_userList){
			for(_Review r: u.getReviews()){
				if(r.getType() == rType.ADAPTATION){
					m_trainSet.add(r);
				}
			}
		}
		// Step 2: train kmeans.
		m_kmeans = new KMeansAlg(m_classNo, m_featureSize, m_k);
		m_kmeans.train(m_trainSet);
		constructCIndexRvwsMap(m_kmeans.getClusters());
	}
	
	// Construct the cluster index and reviews map.
	// Key: cluster index; value: review array.
	public void constructCIndexRvwsMap(InstanceList[] clusters){
		m_cIndexRvws = new HashMap<Integer, ArrayList<_Doc>>();
		for(int i=0; i<m_k; i++)
			m_cIndexRvws.put(i, new ArrayList<_Doc>());
		
		for(int i=0; i<clusters.length; i++){
			for(Instance ins: clusters[i]){
				m_cIndexRvws.get(i).add((_Doc)ins.getSource());
			}
		}
	}
	
	// Train individual svms for each cluster.
	SVM[] m_svms;
	public void trainSVMs(){
		m_svms = new SVM[m_k];
		for(int i: m_cIndexRvws.keySet()){
			m_svms[i] = new SVM(m_classNo, m_featureSize, 1);
			m_svms[i].train(m_cIndexRvws.get(i));
		}
	}
	@Override
	protected void setPersonalizedModel() {
		// TODO Auto-generated method stub
		
	}
	
	public void calcWeightedAverage(){
		double pred = 0;
		for(_AdaptStruct u: m_userList){
			for(_Review r: u.getReviews()){
				//If it is a test review.
				if(r.getType() == rType.TEST){
					pred = 0;
					double[] prob = calcProb(r);
					for(int i=0; i<prob.length; i++){
						
						pred += m_label?m_svms[i].predict(r)*prob[i]:m_svms[i].predictDouble(r)*prob[i];
					}
					r.setPredValue(pred);
				}
			}
		}
	}
	
	public void setLabel(boolean b){
		m_label = b;
	}
	// Calculate the distance between the review and the cluster center.
	public double[] calcProb(_Review r){
		double[] prob = new double[m_k];
		SparseVector sv;
		double sum = 0;
		int cSize = m_kmeans.getCentroids().size();
		for(int i=0; i<cSize; i++){
			sv = m_kmeans.getCentroids().get(i);
			prob[i] = m_kmeans.getDistance().distance(sv, m_kmeans.createInstance(r));
			sum += prob[i];
		}
		for(int i=0; i<prob.length; i++)
			prob[i] /= sum;
		return prob;
	}
	@Override
	public String toString() {
		return String.format("IndividualSVMWithKmeans[k:%d]", m_k);
	}
}
