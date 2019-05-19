package Classifier.supervised.modelAdaptation.HDP;

import Classifier.supervised.LogisticRegression4DP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
import structures._Doc;
import structures._Doc.rType;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._SparseFeature;
import utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;

public class MTCLinAdaptWithHDPLR extends MTCLinAdaptWithHDP{

	public MTCLinAdaptWithHDPLR(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}

	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPLR[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	
	double m_lambda = 1; // parameter used in lr.
	LogisticRegression4DP m_lr;
	ArrayList<_Doc> m_lrTrainSet = new ArrayList<_Doc>();
	
	// collect the training reviews and train the lr model.
	public void buildLogisticRegression(){
		int cNo = 0;
		_DPAdaptStruct user;
		m_lrTrainSet.clear();
		m_lr = new LogisticRegression4DP(m_kBar, m_lmDim, m_lambda);
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if(r.getType() == rType.ADAPTATION){
					cNo = r.getHDPThetaStar().getIndex();
					r.setClusterNo(cNo);
					m_lrTrainSet.add(r);
				}
			}
		}
		m_lr.train(m_lrTrainSet);
	}
	//apply current model in the assigned clusters to users
	@Override
	protected void evaluateModel() {
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
		
		System.out.println("[Info]Accumulating evaluation results during sampling...");

		//calculate cluster posterior p(c|u)
		calculateClusterProbPerUser();
		buildLogisticRegression();
		
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();		
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					_HDPAdaptStruct user;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_HDPAdaptStruct)m_userList.get(i+core);
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
		System.out.print(String.format("[Info]Same: %d, Different: %d\n", m_same, m_diff));
	}
	
	int m_same = 0, m_diff = 0;
	void evaluate(_Review r){
		double[] cProbs = m_lr.calcCProbs(r);
		if(Utils.argmax(cProbs) == r.getHDPThetaStar().getIndex())
			m_same++;
		else
			m_diff++;
		int n, m;
		double As[], prob = 0;
		
		double sum = 0;
		for(int k=0; k<cProbs.length; k++){
			As = CLRWithHDP.m_hdpThetaStars[k].getModel();
			sum = As[0]*CLinAdaptWithHDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
			for(_SparseFeature fv: r.getSparse()){
				n = fv.getIndex() + 1;
				m = m_featureGroupMap[n];
				sum += (As[m]*CLinAdaptWithHDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
			}
			prob += cProbs[k] * Utils.logistic(sum);
		}
	
		//accumulate the prediction results during sampling procedure
		r.m_pCount ++;
		r.m_prob += prob; //>0.5?1:0;
	}
}
