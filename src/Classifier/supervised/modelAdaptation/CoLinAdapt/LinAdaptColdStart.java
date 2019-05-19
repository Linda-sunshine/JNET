package Classifier.supervised.modelAdaptation.CoLinAdapt;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc.rType;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class LinAdaptColdStart extends LinAdapt {
	int m_threshold;
	public LinAdaptColdStart(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel, String featureGroupFile, int t) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupFile);
		m_threshold = t;
	}
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0, w[], oldFValue = Double.MAX_VALUE, totalFvalue = 0;
		
		init();
		for(_AdaptStruct user:m_userList) {	
			if(user.getTestSize() == 0)
				continue;
				
			initLBFGS();
			iflag[0] = 0;
			try{
				w = user.getUserModel();
				oldFValue = Double.MAX_VALUE; 
				do{
					Arrays.fill(m_g, 0); // initialize gradient					
					fValue = calculateFuncValue(user);
					calculateGradients(user);
					
					if (m_displayLv==2) {
						System.out.println("Fvalue is " + fValue);
						gradientTest();
					} else if (m_displayLv==1) {
						if (fValue<oldFValue)
							System.out.print("o");
						else
							System.out.print("x");
					} 
					oldFValue = fValue;
					
					LBFGS.lbfgs(w.length, 6, w, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-32, iflag);//In the training process, A is updated.
				} while(iflag[0] != 0);
			} catch(ExceptionWithIflag e) {
				if (m_displayLv>0)
					System.out.print("X");
				else
					System.out.println("X");
			}
			
			if (m_displayLv>0)
				System.out.println();			
			
			totalFvalue += fValue;
		}
		
		setPersonalizedModel();
		return totalFvalue;
	}
	
	//Calculate the function value of the new added instance.
	protected double calcLogLikelihood(_AdaptStruct user) {
		double L = 0; // log likelihood.
		double Pi = 0;
		int count = 0;

		for (_Review review : user.getReviews()) {
			while (count < m_threshold) {

				Pi = logit(review.getSparse(), user);
				if (review.getYLabel() == 1) {
					if (Pi > 0.0)
						L += Math.log(Pi);
					else
						L -= Utils.MAX_VALUE;
				} else {
					if (Pi < 1.0)
						L += Math.log(1 - Pi);
					else
						L -= Utils.MAX_VALUE;
				}
				count++;
			}
		}
		if (m_LNormFlag)
			return L / getAdaptationSize(user);
		else
			return L;
	}
		
	protected void gradientByFunc(_AdaptStruct user) {
		// Update gradients one review by one review.
		int count = 0;
		while(count < m_threshold){
			_Review review = user.getReviews().get(count);
			gradientByFunc(user, review, 1.0);// weight all the instances equally
			count++;
		}
	}	
	
	@Override
	protected void setPersonalizedModel() {
		int gid;
		_LinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_LinAdaptStruct)m_userList.get(i);
			if(user.getTestSize() == 0)
				continue;
			//set bias term
			m_pWeights[0] = user.getScaling(0) * m_gWeights[0] + user.getShifting(0);
			
			//set the other features
			for(int n=0; n<m_featureSize; n++) {
				gid = m_featureGroupMap[1+n];
				m_pWeights[1+n] = user.getScaling(gid) * m_gWeights[1+n] + user.getShifting(gid);
			}
			user.setPersonalizedModel(m_pWeights);
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
