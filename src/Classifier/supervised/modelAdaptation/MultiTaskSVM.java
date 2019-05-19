package Classifier.supervised.modelAdaptation;

import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.*;
import structures._Doc.rType;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._SparseFeature;
import structures._User;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class MultiTaskSVM extends ModelAdaptation {
	double m_u = 1; // trade-off parameter between global model and individual model.
	double m_C = 1; // trade-off parameter for SVM training 
	
	Model m_libModel; // Libmodel trained by liblinear.
	boolean m_bias = true; // whether use bias term in SVM; by default, we will use it
	
	public MultiTaskSVM(int classNo, int featureSize){
		super(classNo, featureSize, null, null);
		
		// the only test mode for MultiTaskSVM is batch
		m_testmode = TestMode.TM_batch;
	}
	
	@Override
	public String toString() {
		return String.format("MT-SVM[mu:%.3f,C:%.3f,bias:%b]", m_u, m_C, m_bias);
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList) 
			m_userList.add(new _AdaptStruct(user));
		m_pWeights = new double[m_featureSize+1];
	}
	
	public void setTradeOffParam(double u, double C){
		m_u = Math.sqrt(u);
		m_C = C;
	}
	
	public void setBias(boolean bias) {
		m_bias = bias;
	}
	
	@Override
	public double train() {
		init();
		
		//Transfer all user reviews to instances recognized by SVM, indexed by users.
		int trainSize = 0, validUserIndex = 0;
		ArrayList<Feature []> fvs = new ArrayList<Feature []>();
		ArrayList<Double> ys = new ArrayList<Double>();		
		
		//Two for loop to access the reviews, indexed by users.
		ArrayList<_Review> reviews;
		for(_AdaptStruct user:m_userList){
			if(user.getAdaptationSize() == 0)
				continue;
			reviews = user.getReviews();		
			boolean validUser = false;
			for(_Review r:reviews) {				
				if (r.getType() == rType.ADAPTATION) {//we will only use the adaptation data for this purpose
					fvs.add(createLibLinearFV(r, validUserIndex));
					ys.add(new Double(r.getYLabel()));
					trainSize ++;
					validUser = true;
				}
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
		
		if (m_bias) {
			libProblem.n = (m_featureSize + 1) * (m_userSize + 1); // including bias term; global model + user models
			libProblem.bias = 1;// bias term in liblinear.
		} else {
			libProblem.n = m_featureSize * (m_userSize + 1);
			libProblem.bias = -1;// no bias term in liblinear.
		}
		
		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;//solver type: SVM
		m_libModel = Linear.train(libProblem, new Parameter(type, m_C, SVM.EPS));
		
		setPersonalizedModel();
		
		return 0;
	}
	
	public void setLibProblemDimension(Problem libProblem){
		if (m_bias) {
			libProblem.n = (m_featureSize + 1) * (m_userSize + 1); // including bias term; global model + user models
			libProblem.bias = 1;// bias term in liblinear.
		} else {
			libProblem.n = m_featureSize * (m_userSize + 1);
			libProblem.bias = -1;// no bias term in liblinear.
		}
	}
	@Override
	protected void setPersonalizedModel() {
		double[] weight = m_libModel.getWeights();//our model always assume the bias term
		int class0 = m_libModel.getLabels()[0];
		double sign = class0 > 0 ? 1 : -1, block=m_personalized?1:0;//disable personalized model when required
		int userOffset = 0, globalOffset = m_bias?(m_featureSize+1)*m_userSize:m_featureSize*m_userSize;
		setSupWeights(sign, block, weight, globalOffset);
		
		for(_AdaptStruct user:m_userList) {
			if (user.getAdaptationSize()>0) {
				for(int i=0; i<m_featureSize; i++) 
					m_pWeights[i+1] = sign*(weight[globalOffset+i]/m_u + block*weight[userOffset+i]);
				
				if (m_bias) {
					m_pWeights[0] = sign*(weight[globalOffset+m_featureSize]/m_u + block*weight[userOffset+m_featureSize]);
					userOffset += m_featureSize+1;
				} else
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
	
	// Set the global part in mt-svm
	public void setSupWeights(double sign, double block, double[] weight, int globalOffset){
		m_gWeights = new double[m_featureSize+1];
		for(int i=0; i<m_featureSize; i++) 
			m_gWeights[i+1] = sign*weight[globalOffset+i]/m_u;
		
		if (m_bias)
			m_gWeights[0] = sign*weight[globalOffset+m_featureSize]/m_u;
	}
	//create a training instance of svm.
	//for MT-SVM feature vector construction: we put user models in front of global model
	public Feature[] createLibLinearFV(_Review r, int userIndex){
		int fIndex; double fValue;
		_SparseFeature fv;
		_SparseFeature[] fvs = r.getSparse();
		
		int userOffset, globalOffset;		
		Feature[] node;//0-th: x//sqrt(u); t-th: x.
		
		if (m_bias) {
			userOffset = (m_featureSize + 1) * userIndex;
			globalOffset = (m_featureSize + 1) * m_userSize;
			node = new Feature[(1+fvs.length) * 2];
		} else {
			userOffset = m_featureSize * userIndex;
			globalOffset = m_featureSize * m_userSize;
			node = new Feature[fvs.length * 2];
		}
		
		for(int i = 0; i < fvs.length; i++){
			fv = fvs[i];
			fIndex = fv.getIndex() + 1;//liblinear's feature index starts from one
			fValue = fv.getValue();
			
			//Construct the user part of the training instance.			
			node[i] = new FeatureNode(userOffset + fIndex, fValue);
			
			//Construct the global part of the training instance.
			if (m_bias)
				node[i + fvs.length + 1] = new FeatureNode(globalOffset + fIndex, fValue/m_u); // global model's bias term has to be moved to the last
			else
				node[i + fvs.length] = new FeatureNode(globalOffset + fIndex, fValue/m_u); // global model's bias term has to be moved to the last
		}
		
		if (m_bias) {//add the bias term		
			node[fvs.length] = new FeatureNode((m_featureSize + 1) * (userIndex + 1), 1.0);//user model's bias
			node[2*fvs.length+1] = new FeatureNode((m_featureSize + 1) * (m_userSize + 1), 1.0 / m_u);//global model's bias
		}
		return node;
	}
	
	public void printEachUserPerf(){
		PrintWriter writer;
		_PerformanceStat stat;
		double pos = 0, neg = 0, count = 0;
		try{
			writer = new PrintWriter(new File("perf.txt"));
			for(_AdaptStruct user: m_userList){
				stat = user.getPerfStat();
				pos = 0; neg = 0;
				for(_Review r: user.getReviews()){
					if(r.getYLabel() == 1) pos++;
					else neg++;
				}
				pos /= user.getReviews().size();
				neg /= user.getReviews().size();
				if(pos == 1 || neg == 1) count++;
				writer.write(String.format("%d\t(%.4f, %.4f)\t(%.4f, %.4f)\n", user.getReviews().size(), pos, neg, stat.getF1(1), stat.getF1(0)));
			}
			System.out.println(count);
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
		
	// for debug purpose
	public _AdaptStruct findUser(String userID){
		for(_AdaptStruct u: m_userList){
			if(u.getUserID().equals(userID))
				return u;
		}
		return null;
	}
	
	// print out the sup part of the mt-svm
	public void saveSupModel(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			if(m_bias)
				writer.write(m_gWeights[m_featureSize]+"\n");
			else
				writer.write(0);
				
			for(int i=0; i<m_featureSize; i++)
					writer.write(m_gWeights[i]+"\n");
			
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
//	/*****Predict each review's label based on mtsvm and mtsvm_global******/
//	@Override
//	public double test(){
//		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		ArrayList<Thread> threads = new ArrayList<Thread>();
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
////								// Only predict the first review of each user.
////								_Review r = user.getReviews().get(0);
////								if (r.getType() != rType.ADAPTATION)
////									System.out.println("Not adaptation data.");
////								int trueL = r.getYLabel();
////								int predL = user.predict(r); // evoke user's own model
////								int predL_G = predictG(r);
////								r.setPredictLabel(predL);
////								r.setPredictLabelG(predL_G);
////								userPerfStat.addOnePredResult(predL, trueL);
//								
//								//record prediction results
//								for(_Review r:user.getReviews()) {
//									if (r.getType() != rType.ADAPTATION)
//										continue;
//									int trueL = r.getYLabel();
//									int predL = user.predict(r); // evoke user's own model
//									int predL_G = predictG(r);
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
//			
//			userPerfStat = user.getPerfStat();
//			for(int i=0; i<m_classNo; i++){
//				if(userPerfStat.getTrueClassNo(i)!=0)
//					macroF1.get(i).add(userPerfStat.getF1(i));
//			}
//			m_microStat.accumulateConfusionMat(userPerfStat);
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
	
	
	/*****Print out users' testing performance of mtsvm_gloabl******/
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
//					if(r.getType() == rType.ADAPTATION)
//						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
//					if(r.getType() == rType.TEST){
//						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabelG(), r.getSource()));
//					}
//				}
//			}
//			writer.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
	
	/*****Print out users' training performance******/
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
//					if(r.getType() == rType.ADAPTATION){
//						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabel(), r.getSource()));
//					}
//					else
//						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
//				}
//			}
//			writer.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
//	
//	// print out each user's training review's performance.
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
//					if(r.getType() == rType.ADAPTATION){
//						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabelG(), r.getSource()));
//					} else
//						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
//				}
//			}
//			writer.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
}
