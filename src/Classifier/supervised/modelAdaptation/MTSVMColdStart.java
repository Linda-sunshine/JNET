package Classifier.supervised.modelAdaptation;

import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.*;
import structures._Doc.rType;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import utils.Utils;

import java.util.ArrayList;

public class MTSVMColdStart extends MultiTaskSVM {

	int m_threshold = 1;
	
	public MTSVMColdStart(int classNo, int featureSize, int t) {
		super(classNo, featureSize);
		m_threshold = t;
	}
	public void setThreshold(int t){
		m_threshold = t;
	}
	@Override
	public double train() {
		init();
		int trainSize = 0, validUserIndex = 0;
		ArrayList<Feature []> fvs = new ArrayList<Feature []>();
		ArrayList<Double> ys = new ArrayList<Double>();		
		_Review r;
		//Two for loop to access the reviews, indexed by users.
		ArrayList<_Review> reviews;
		for(_AdaptStruct user:m_userList){
			int count = 0;
			if(user.getTestSize() == 0)
				continue;// ignore the adaptation users.
			reviews = user.getReviews();		
			boolean validUser = false;
			while(count < m_threshold){
				r = reviews.get(count);
				fvs.add(createLibLinearFV(r, validUserIndex));
				ys.add(new Double(r.getYLabel()));
				trainSize ++;
				validUser = true;
				count++;
			}	
			if (validUser)
				validUserIndex ++;
		}		
		// Train a liblinear model based on all reviews.
		Problem libProblem = new Problem();
		libProblem.l = trainSize;		
		libProblem.x = new Feature[trainSize][];
		libProblem.y = new double[trainSize];
		for(int i=0; i<trainSize; i++) {
			libProblem.x[i] = fvs.get(i);
			libProblem.y[i] = ys.get(i);
		}
		
		setLibProblemDimension(libProblem);
//		if (m_bias) {
//			libProblem.n = (m_featureSize + 1) * (m_userSize + 1); // including bias term; global model + user models
//			libProblem.bias = 1;// bias term in liblinear.
//		} else {
//			libProblem.n = m_featureSize * (m_userSize + 1);
//			libProblem.bias = -1;// no bias term in liblinear.
//		}
		
		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;//solver type: SVM
		m_libModel = Linear.train(libProblem, new Parameter(type, m_C, SVM.EPS));
		setPersonalizedModel();
		return 0;
	}
	@Override
	protected void setPersonalizedModel() {
		double[] weight = m_libModel.getWeights();//our model always assume the bias term
		int class0 = m_libModel.getLabels()[0];
		double sign = class0 > 0 ? 1 : -1;
		int userOffset = 0, globalOffset = m_bias?(m_featureSize+1)*m_userSize:m_featureSize*m_userSize;
		for(_AdaptStruct user:m_userList) {
			if(user.getTestSize() == 0)
				continue;
			if(m_personalized){
				for(int i=0; i<m_featureSize; i++) 
					m_pWeights[i+1] = sign*(weight[globalOffset+i]/m_u + weight[userOffset+i]);
				
				if (m_bias) {
					m_pWeights[0] = sign*(weight[globalOffset+m_featureSize]/m_u + weight[userOffset+m_featureSize]);
					userOffset += m_featureSize+1;
				} 
				else
					userOffset += m_featureSize;
			} else {
				for(int i=0; i<m_featureSize; i++) // no personal model since no adaptation data
					m_pWeights[i+1] = sign*weight[globalOffset+i]/m_u;
				
				if (m_bias)
					m_pWeights[0] = sign*weight[globalOffset+m_featureSize]/m_u;
			}
			
			user.setPersonalizedModel(m_pWeights);//our model always assume the bias term
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
									if(count++ < m_threshold)// For the review used for assignment
										continue;
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
}
