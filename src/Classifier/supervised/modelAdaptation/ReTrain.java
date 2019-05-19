package Classifier.supervised.modelAdaptation;

import Classifier.supervised.IndividualSVM;
import structures.MyPriorityQueue;
import structures._Doc.rType;
import structures._RankItem;
import structures._Review;
import structures._User;
import utils.Utils;

import java.util.ArrayList;

/***
 * @author lin
 * In this class, we select k-nearest document for each user adn 
 */
public class ReTrain extends IndividualSVM {
	int m_topK = 0;
	ArrayList<_Review> m_trainRvws;
	public ReTrain(int classNo, int featureSize, int topK) {
		super(classNo, featureSize);
		m_trainRvws = new ArrayList<_Review>();
		m_topK = topK;
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList){
			m_userList.add(new _AdaptStruct(user));
			m_pWeights = new double[m_featureSize+1];	
			mergeTrainRvws(user);
		}
		extendTrainSet();
	}
	
	// Collect the training reviews.
	public void mergeTrainRvws(_User u){
		for(_Review r: u.getReviews()){
			if(r.getType() == rType.ADAPTATION || r.getType() == rType.TRAIN){
				m_trainRvws.add(r);
			}
		}
	}
	
	// For each review of each user, extend it with k-nearest reviews.
	public void extendTrainSet(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run(){
					_AdaptStruct user;
					ArrayList<_Review> rvws = new ArrayList<_Review>();
					try{
						for(int i=0; i+core<m_userList.size(); i+=numOfCores) {
							user = m_userList.get(i+core);	
							for(_Review r: user.getReviews()){
								if(r.getType() == rType.ADAPTATION || r.getType() == rType.TRAIN)
									rvws.addAll(findTopKNeighbors(r));
							}	
							user.getUser().appendRvws(rvws);
						}
					} catch(Exception e){
						e.printStackTrace();
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
		System.out.format("[Info]Extend reviews for %d users finished!\n", m_userList.size());
	}
	
	// For the review, find the top K reviews in the training set.
	public ArrayList<_Review> findTopKNeighbors(_Review r){
		ArrayList<_Review> topKRvws = new ArrayList<_Review>();
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(m_topK); 
		for(int i=0; i<m_trainRvws.size(); i++){
			if(!m_trainRvws.get(i).equals(r))
				queue.add(new _RankItem(i, Utils.cosine(r.getSparse(), m_trainRvws.get(i).getSparse())));
		}
		for(_RankItem nit: queue)
			topKRvws.add(m_trainRvws.get(nit.m_index));
		return topKRvws;
	}
}
