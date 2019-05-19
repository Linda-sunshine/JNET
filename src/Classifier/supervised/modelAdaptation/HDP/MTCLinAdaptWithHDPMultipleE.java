package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._HDPThetaStar;
import structures._Review;
import structures._Doc.rType;

/***
 * In the class, we consider the cluster sampling multiple times to get an expectation of the sampling results. 
 * We can assign each review to different clusters in multiple E steps and use them to do MLE.
 * @author lin
 *
 */
public class MTCLinAdaptWithHDPMultipleE extends MTCLinAdaptWithHDP{

	public MTCLinAdaptWithHDPMultipleE(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}

	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPMultipleE[dim:%d,supDim:%d,lmDim:%d,thinning:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_thinning,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	protected void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		super.sampleOneInstance(user, r);
		// We also put the sampled cluster to the review for later MLE.
		r.updateThetaCountMap(1);
	}
	
	public void clearReviewStats(){
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;//do not touch testing reviews!
				r.clearThetaCountMap();	
			}
		}
	}
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0, ecount = 0;
		
		init(); // clear user performance and init cluster assignment	

		// Burn in period, still one E-step -> one M-step -> one E-step -> one M-step
		while(count++ < m_burnIn){
			calculate_E_step();
			calculate_M_step();
		}
		System.out.println("[Info]Burn in period ends, starts iteration...");
		clearReviewStats();

		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			while(ecount++ < m_thinning){
				calculate_E_step();
				// split steps in M-step, multiple E -> MLE -> multiple E -> MLE
				assignClusterIndex();		
				sampleGamma();
			}
			
			curLikelihood += estPhi();
			delta = (lastLikelihood - curLikelihood)/curLikelihood;
				
			// After M step, we need to clear the review stats and start collecting again.
			ecount = 0;
			clearReviewStats();
			
			evaluateModel();
				
			printInfo(i%5==0);//no need to print out the details very often
			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}

	// Sample the weights given the cluster assignment.
	@Override
	protected double calculate_M_step(){
		assignClusterIndex();		
		
		//Step 1: sample gamma based on the current assignment.
		sampleGamma(); // why for loop BETA_K times?
		
		//Step 2: Optimize language model parameters with MLE.
		return estPhi();
	}
	
	// In function logLikelihood, we update the loglikelihood and corresponding gradients.
	// Thus, we only need to update the two functions correspondingly with.
	protected double calcLogLikelihoodY(_Review r){
		int index = -1;
		_HDPThetaStar oldTheta = r.getHDPThetaStar();
		HashMap<_HDPThetaStar, Integer> thetaCountMap = r.getThetaCountMap();
		double likelihood = 0;
		for(_HDPThetaStar theta: thetaCountMap.keySet()){
			index = findHDPThetaStar(theta);
			// some of the cluster may disappear, ignore them.
			if(index >= m_kBar || index < 0)
				continue;
			r.setHDPThetaStar(theta);
			// log(likelihood^k) = k * log likelihood.
			likelihood += thetaCountMap.get(theta) * super.calcLogLikelihoodY(r);
		}
		r.setHDPThetaStar(oldTheta);
		return likelihood;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		int index = -1;
		double confidence = 1;
		_Review r = (_Review) review;
		_HDPThetaStar oldTheta = r.getHDPThetaStar();
		HashMap<_HDPThetaStar, Integer> thetaCountMap = r.getThetaCountMap();
		
		for(_HDPThetaStar theta: thetaCountMap.keySet()){
			index = findHDPThetaStar(theta);
			// some of the cluster may disappear, ignore them.
			if(index >= m_kBar || index < 0)
				continue;
			r.setHDPThetaStar(theta);
			confidence = thetaCountMap.get(theta);
			// confidence plays the role of weight here, how many times the review shows in the cluster.
			super.gradientByFunc(u, review, confidence, g);
		}
		r.setHDPThetaStar(oldTheta);
	}
	
//	@Override
//	// After we finish estimating the clusters, we calculate the probability of each testing review belongs to each cluster.
//	// Indeed, it is for per review, for inheritance we don't change the function name.
//	protected void calculateClusterProbPerUser(){
//		double prob, logSum;
//		double[] probs;
//		if(m_newCluster) 
//			probs = new double[m_kBar+1];
//		else 
//			probs = new double[m_kBar];
//		
//		_HDPAdaptStruct user;
//		_HDPThetaStar oldTheta, curTheta;
//		
//		//sample a new cluster parameter first.
//		if(m_newCluster) {
//			m_hdpThetaStars[m_kBar].setGamma(m_gamma_e);//to make it consistent since we will only use one auxiliary variable
//			m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
//		}
//		for(int i=0; i<m_userList.size(); i++){
//			user = (_HDPAdaptStruct) m_userList.get(i);
//			for(_Review r: user.getReviews()){
//				if (r.getType() != rType.ADAPTATION)
//					continue;				
//				oldTheta = r.getHDPThetaStar();
//				for(int k=0; k<probs.length; k++){
//					curTheta = m_hdpThetaStars[k];
//					r.setHDPThetaStar(curTheta);
//					prob = calcLogLikelihoodX(r) + Math.log(user.getHDPThetaMemSize(curTheta) + m_eta*curTheta.getGamma());//this proportion includes the user's current cluster assignment
//					probs[k] = prob;
//				}
//			
//				logSum = Utils.logSumOfExponentials(probs);
//				for(int k=0; k<probs.length; k++)
//					probs[k] -= logSum;
//				r.setClusterPosterior(probs);//posterior in log space
//				r.setHDPThetaStar(oldTheta);
//			}
//		}
//	}
//	
//	//apply current model in the assigned clusters to users
//	protected void evaluateModel() {//this should be only used in batch testing!
//		System.out.println("[Info]Accumulating evaluation results during sampling...");
//
//		//calculate cluster posterior p(c|u)
//		calculateClusterProbPerUser();
//			
//		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		ArrayList<Thread> threads = new ArrayList<Thread>();		
//			
//		for(int k=0; k<numberOfCores; ++k){
//			threads.add((new Thread() {
//				int core, numOfCores;
//				public void run() {
//					_HDPAdaptStruct user;
//					try {
//						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
//							user = (_HDPAdaptStruct)m_userList.get(i+core);
//							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//								continue;
//									
//							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
//								//record prediction results
//								for(_Review r:user.getReviews()) {
//									if (r.getType() != rType.ADAPTATION)
//										continue;
//									user.evaluate(r); // evoke user's own model
//									user.evaluateG(r);
//								}
//							}							
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
//	}	
//
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
//							user = (_DPAdaptStruct) user;
//							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//								continue;
//								
//							userPerfStat = user.getPerfStat();								
//							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
//								//record prediction results
//								for(_Review r:user.getReviews()) {
//									if (r.getType() != rType.ADAPTATION)
//										continue;
//									int trueL = r.getYLabel();
//									int predL = user.predict(r); // evoke user's own model
//									int predL_G = ((_DPAdaptStruct) user).predictG(r);
//									r.setPredictLabel(predL);
//									r.setPredictLabelG(predL_G);
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
//		
//		int count = 0;
//		ArrayList<ArrayList<Double>> macroF1 = new ArrayList<ArrayList<Double>>();
//		
//		//init macroF1
//		for(int i=0; i<m_classNo; i++)
//			macroF1.add(new ArrayList<Double>());
//		
//		_PerformanceStat userPerfStat;
//		m_microStat.clear();
//		for(_AdaptStruct user:m_userList) {
//			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//				continue;
//			
//			userPerfStat = user.getPerfStat();
//			for(int i=0; i<m_classNo; i++){
//				if(userPerfStat.getTrueClassNo(i)!=0)
//					macroF1.get(i).add(userPerfStat.getF1(i));
//			}
//			m_microStat.accumulateConfusionMat(userPerfStat);
//			count ++;
//		}
//		System.out.print("neg users: " + macroF1.get(0).size());
//		System.out.print("\tpos users: " + macroF1.get(1).size()+"\n");
//
//		System.out.println(toString());
//		calcMicroPerfStat();
//		// macro average and standard deviation.
//		System.out.println("\nMacro F1:");
//		for(int i=0; i<m_classNo; i++){
//			double[] avgStd = calcAvgStd(macroF1.get(i));
//			System.out.format("Class %d: %.4f+%.4f\t", i, avgStd[0], avgStd[1]);
//		}
//		return 0;
//	}
//	
//	public void printUserPerformance(String filename){
//		PrintWriter writer;
//		try{
//			writer = new PrintWriter(new File(filename));
//			Collections.sort(m_userList, new Comparator<_AdaptStruct>(){
//				@Override
//				public int compare(_AdaptStruct u1, _AdaptStruct u2){
//					return String.CASE_INSENSITIVE_ORDER.compare(u1.getUserID(), u2.getUserID());
//				}
//			});
//			for(_AdaptStruct u: m_userList){
//				writer.write("-----\n");
//				writer.write(String.format("%s\t%d\n", u.getUserID(), u.getReviews().size()));
//				for(_Review r: u.getReviews()){
//					if(r.getType() == rType.TEST)
//						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
//					if(r.getType() == rType.ADAPTATION){
//						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabel(), r.getSource()));
//					}
//				}
//			}
//			writer.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
//	
//	// print out each user's test review's performance.
//	public void printGlobalUserPerformance(String filename){
//		PrintWriter writer;
//		try{
//			writer = new PrintWriter(new File(filename));
//			Collections.sort(m_userList, new Comparator<_AdaptStruct>(){
//				@Override
//				public int compare(_AdaptStruct u1, _AdaptStruct u2){
//					return String.CASE_INSENSITIVE_ORDER.compare(u1.getUserID(), u2.getUserID());
//				}
//			});
//			for(_AdaptStruct u: m_userList){
//				writer.write("-----\n");
//				writer.write(String.format("%s\t%d\n", u.getUserID(), u.getReviews().size()));
//				for(_Review r: u.getReviews()){
//					if(r.getType() == rType.TEST)
//						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
//					if(r.getType() == rType.ADAPTATION){
//						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabelG(), r.getSource()));
//					}
//				}
//			}
//			writer.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
}
