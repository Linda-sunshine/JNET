package Classifier.supervised.modelAdaptation.DirichletProcess;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import cern.jet.random.tfloat.FloatUniform;
import structures._Doc.rType;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._thetaStar;
import utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;

public class IsoCLRWithDP extends CLRWithDP {

	public IsoCLRWithDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		// TODO Auto-generated constructor stub
	}
	
	boolean m_allFlag = false;
	int m_threshold = 1; // how many reviews will be used for assignment.

	public void setClusterAssignThreshold(int t){
		m_threshold = t;
	}
	public void setAllFlag(boolean b){
		m_allFlag = b;
	}
	
	// The main MCMC algorithm, assign each user to clusters.
	protected void calculate_E_step() {
		_thetaStar curThetaStar;
		_DPAdaptStruct user;

		for (int i = 0; i < m_userList.size(); i++) {
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0)
				continue;
			curThetaStar = user.getThetaStar();
			curThetaStar.updateMemCount(-1);

			if (curThetaStar.getMemSize() == 0) {// No data associated with the
													// cluster.
				swapTheta(m_kBar - 1, findThetaStar(curThetaStar)); 
				m_kBar--;
			}
			sampleOneInstance(user);
		}
	}
	
	protected void assignCluster(){
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0)
				assignOneCluster(user);
		}
	}
	protected void assignOneCluster(_DPAdaptStruct u){
		_DPAdaptStruct user = u;
		double likelihood, logSum = 0;
		int k;
		for (k = 0; k < m_kBar; k++) {
			user.setThetaStar(m_thetaStars[k]);
			likelihood = calcLogLikelihood(user, m_threshold);
			
			likelihood += Math.log(m_thetaStars[k].getMemSize());
			m_thetaStars[k].setProportion(likelihood);// this is in log space!
		
			if (k == 0)
				logSum = likelihood;
			else
				logSum = Utils.logSum(logSum, likelihood);
		}

		logSum += Math.log(FloatUniform.staticNextFloat());

		k = 0;
		double newLogSum = m_thetaStars[0].getProportion();
		do {
			if (newLogSum >= logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_thetaStars[k].getProportion());
		} while (k < m_kBar);

		user.setThetaStar(m_thetaStars[k]);
	}
	
	protected double calcLogLikelihood(_AdaptStruct user, int k){
		double L = 0; //log likelihood.
		double Pi = 0;
		int threshold = k;
		if(m_allFlag)
			threshold = user.getReviews().size();
		_Review review;
		for(int i=0; i<threshold; i++){
			review = user.getReviews().get(i);
		
			Pi = logit(review.getSparse(), user);
			if(review.getYLabel() == 1) {
				if (Pi>0.0)
					L += Math.log(Pi);					
				else
					L -= Utils.MAX_VALUE;
			} else {
				if (Pi<1.0)
					L += Math.log(1 - Pi);					
				else
					L -= Utils.MAX_VALUE;
			}
		}
		return L;
	}
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
	@Override
	protected void calculateClusterProbPerUser(){
		double prob;
		_DPAdaptStruct user;
		double[] probs = new double[m_kBar];
		_thetaStar oldTheta;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getTestSize() == 0)
				continue;
			oldTheta = user.getThetaStar();
			for(int k=0; k<m_kBar; k++){
				user.setThetaStar(m_thetaStars[k]);
				prob = calcLogLikelihood(user, m_threshold);
				prob += Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
				probs[k] = Math.exp(prob);//this will be in real space!
			}
			Utils.L1Normalization(probs);
			user.setClusterPosterior(probs);
			user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
		}
	}
	// Sample one instance's cluster assignment.
	protected void sampleOneInstance(_DPAdaptStruct user){
		if(user.getAdaptationSize() != 0){
			super.sampleOneInstance(user);
		}	
	}
	
	@Override
	public double test(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		m_perf = new double[2];
		m_microStat = new _PerformanceStat(m_classNo);

		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_AdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							int count = 0;
							user = m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
							userPerfStat = user.getPerfStat();								
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									if(count < m_threshold){
										count++;// For the review used for assignment
										continue;
									}
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									userPerfStat.addOnePredResult(predL, trueL);
								}
							}							
							userPerfStat.calculatePRF();	
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
		
		int count = 0;
		double[] macroF1 = new double[m_classNo];
		_PerformanceStat userPerfStat;
		
		for(_AdaptStruct user:m_userList) {
			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
				continue;
			
			userPerfStat = user.getPerfStat();
			for(int i=0; i<m_classNo; i++)
				macroF1[i] += userPerfStat.getF1(i);
			m_microStat.accumulateConfusionMat(userPerfStat);
			count ++;
		}
		
		System.out.println(toString());
		calcMicroPerfStat();
		
		// macro average
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++){
			System.out.format("Class %d\t%.4f\t", i, macroF1[i]/count);
			m_perf[i] = macroF1[i]/count;
		}
		System.out.println();
		return Utils.sumOfArray(macroF1);
	}
	
	
	//apply current model in the assigned clusters to users
		protected void evaluateModel() {
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
								int count  = 0;
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
										if(count< m_threshold){
											count++;
											continue;
										}
										user.evaluate(r); // evoke user's own model
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

}
