/**
 * 
 */
package Classifier.supervised.modelAdaptation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import Classifier.BaseClassifier;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import structures._Doc;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;
import structures._Doc.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

/**
 * @author Hongning Wang
 * abstract class for model adaptation algorithms
 */
public abstract class ModelAdaptation extends BaseClassifier {
	protected ArrayList<_AdaptStruct> m_userList; // references to the users	
	protected int m_userSize; // valid user size
	
	protected double[] m_gWeights; //global model weight
	protected double[] m_pWeights; // cache for personalized weight

	protected TestMode m_testmode; // test mode of different algorithms 
	protected int m_displayLv = 1;//0: display nothing during training; 1: display the change of objective function; 2: display everything

	//if we will set the personalized model to the target user (otherwise use the global model)
	protected boolean m_personalized;

	// Decide if we will normalize the likelihood.
	protected boolean m_LNormFlag = true;
//	protected String m_dataset = "Amazon"; // Default dataset.
	protected double[] m_perf = new double[2]; // added by Lin for retrieving performance after each test.
	
	// added by Lin.
	public ModelAdaptation(int classNo, int featureSize) {
		super(classNo, featureSize);
		m_pWeights = null;
		m_personalized = true;
	}
	
	public ModelAdaptation(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize);
		
		loadGlobalModel(featureMap, globalModel);
		m_pWeights = null;
		m_personalized = true;
	}
	
	public ModelAdaptation(int classNo, int featureSize, String globalModel) {
		super(classNo, featureSize);
		
		loadGlobalModel(globalModel);
		m_pWeights = null;
		m_personalized = true;
	}
	
	public void setDisplayLv(int level) {
		m_displayLv = level;
	}
	
	public void setTestMode(TestMode mode) {
		m_testmode = mode;
	}
	
	public void setPersonalization(boolean p) {
		m_personalized = p;
	}	
	
	public void setLNormFlag(boolean b){
		m_LNormFlag = b;
	}
	
	//Load global model from file.
	public void loadGlobalModel(HashMap<String, Integer> featureMap, String filename){
		if (featureMap==null || filename==null)
			return;
		
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, features[];
			int pos;
			
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			m_features = new String[m_featureSize+1];//list of detailed features
			
			while((line=reader.readLine()) != null) {
				features = line.split(":");
				if (features[0].equals("BIAS")) {
					m_gWeights[0] = Double.valueOf(features[1]);
					m_features[0] = "BIAS";
				}
				else if (featureMap.containsKey(features[0])){
					pos = featureMap.get(features[0]);
					if (pos>=0 && pos<m_featureSize) {
						m_gWeights[pos+1] = Double.valueOf(features[1]);
						m_features[pos+1] = features[0];
					} else
						System.err.println("[Warning]Unknown feature " + features[0]);
				} else 
					System.err.println("[Warning]Unknown feature " + features[0]);
			}
			
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}

	//Load global model from file.
	public void loadGlobalModel(String filename){
		if (filename==null)
			return;
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, features[];
			int index = 0;
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			while((line=reader.readLine()) != null) {
				features = line.split("\\s+");
				if(features.length == 1 && !features[0].equals("w")){
					m_gWeights[index++] = Double.valueOf(features[0]);
				}
			}			
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	
	abstract public void loadUsers(ArrayList<_User> userList);
	
	protected void constructNeighborhood(final SimType sType) {
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				
				@Override
				public void run() {
					CoAdaptStruct ui, uj;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							ui = (CoAdaptStruct)m_userList.get(i+core);
							for(int j=0; j<m_userList.size(); j++) {
								if (j == i+core)
									continue;
								uj = (CoAdaptStruct)(m_userList.get(j));
								
								ui.addNeighbor(j, ui.getSimilarity(uj, sType));
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

		System.out.format("[Info]Neighborhood graph based on %s constructed for %d users...\n", sType, m_userList.size());
	}
	
	protected int[] constructReverseNeighborhood() {
		int adaptSize = 0;//total number of adaptation instances
		
		//construct the reverse link
		CoAdaptStruct ui, uj;
		for(int i=0; i<m_userList.size(); i++) {
			ui = (CoAdaptStruct)(m_userList.get(i));
			for(_RankItem nit:ui.getNeighbors()) {
				uj = (CoAdaptStruct)(m_userList.get(nit.m_index));//uj is a neighbor of ui
				
				uj.addReverseNeighbor(i, nit.m_value);
			}
			adaptSize += ui.getAdaptationSize();
		}
		
		//construct the order of online updating
		ArrayList<_RankItem> userorder = new ArrayList<_RankItem>();
		for(int i=0; i<m_userList.size(); i++) {
			ui = (CoAdaptStruct)(m_userList.get(i));
			
			for(_Review r:ui.getReviews()) {//reviews in each user is already ordered by time
				if (r.getType() == rType.ADAPTATION) {
					userorder.add(new _RankItem(i, r.getTimeStamp()));//to be in ascending order
				}
			}
		}
		
		Collections.sort(userorder);
		
		int[] userOrder = new int[adaptSize];
		for(int i=0; i<adaptSize; i++)
			userOrder[i] = userorder.get(i).m_index;
		return userOrder;
	}
	
	
	@Override
	protected void init(){
		m_userSize = 0;//need to get the total number of valid users to construct feature vector for MT-SVM
		for(_AdaptStruct user:m_userList){			
			if (user.getAdaptationSize()>0) 				
				m_userSize ++;	
			user.getPerfStat().clear(); // clear accumulate performance statistics
		}
	}
	
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationSize();
	}
	
	abstract protected void setPersonalizedModel();
	
	// Used for sanity check for personalization.
	public int predictG(_Doc doc) {
		_SparseFeature[] fv = doc.getSparse();

		double maxScore = Utils.dotProduct(m_gWeights, fv, 0);
		if (m_classNo==2) {
			return maxScore>0?1:0;
		} 
		System.err.print("Wrong classification task!");
		return -1;
	}
	@Override
	public double test(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					_AdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
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
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									r.setPredictLabel(predL);
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
		ArrayList<ArrayList<Double>> macroF1 = new ArrayList<ArrayList<Double>>();
		
		//init macroF1
		for(int i=0; i<m_classNo; i++)
			macroF1.add(new ArrayList<Double>());
		
		_PerformanceStat userPerfStat;
		m_microStat.clear();
		for(_AdaptStruct user:m_userList) {
			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
				continue;
			
			userPerfStat = user.getPerfStat();
			for(int i=0; i<m_classNo; i++){
				if(userPerfStat.getTrueClassNo(i) > 0)
					macroF1.get(i).add(userPerfStat.getF1(i));
			}
			m_microStat.accumulateConfusionMat(userPerfStat);
			count ++;
		}
		System.out.print("neg users: " + macroF1.get(0).size());
		System.out.print("\tpos users: " + macroF1.get(1).size()+"\n");

		System.out.println(toString());
		calcMicroPerfStat();
		// macro average and standard deviation.
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++){
			double[] avgStd = calcAvgStd(macroF1.get(i));
			m_perf[i] = avgStd[0];
			System.out.format("Class %d: %.4f+%.4f\t", i, avgStd[0], avgStd[1]);
		}
//		printPerformance();
		return 0;
	}

	public void printPerformance(){
		_PerformanceStat perf;
		for(_AdaptStruct user:m_userList){
			perf = user.getPerfStat();
			System.out.print(String.format("pos:%d\tneg:%d\tposF1:%.4f\tnegF1:%.4f\n",
					perf.getTrueClassNo(1), perf.getTrueClassNo(0), perf.getF1(1), perf.getF1(0)));
			
		}
	}
	public double[] calcAvgStd(ArrayList<Double> fs){
		double avg = 0, std = 0;
		for(double f: fs)
			avg += f;
		avg /= fs.size();
		for(double f: fs)
			std += (f - avg) * (f - avg);
		std = Math.sqrt(std/fs.size());
		return new double[]{avg, std};
	}
	
	@Override
	public void saveModel(String modelLocation) {	
		File dir = new File(modelLocation);
		if(!dir.exists())
			dir.mkdirs();
		for(_AdaptStruct user:m_userList) {
			try {
	            BufferedWriter writer = new BufferedWriter(new FileWriter(modelLocation+"/"+user.getUserID()+".txt"));
	            StringBuilder buffer = new StringBuilder(512);
	            double[] pWeights = user.getPWeights();
	            for(int i=0; i<pWeights.length; i++) {
	            	buffer.append(pWeights[i]);
	            	if (i<pWeights.length-1)
	            		buffer.append(',');
	            }
	            writer.write(buffer.toString());
	            writer.close();
	        } catch (Exception e) {
	            e.printStackTrace(); 
	        } 
		}
		System.out.format("\n[Info]Save personalized models to %s.", modelLocation);
	}
	
	@Override
	public double train(Collection<_Doc> trainSet) {
		System.err.println("[Error]train(Collection<_Doc> trainSet) is not implemented in ModelAdaptation family!");
		System.exit(-1);
		return Double.NaN;
	}

	@Override
	public int predict(_Doc doc) {//predict by global model		
		System.err.println("[Error]predict(_Doc doc) is not implemented in ModelAdaptation family!");
		System.exit(-1);
		return Integer.MAX_VALUE;
	}

	@Override
	public double score(_Doc d, int label) {//prediction score by global model
		System.err.println("[Error]score(_Doc d, int label) is not implemented in ModelAdaptation family!");
		System.exit(-1);
		return Double.NaN;
	}

	@Override
	protected void debug(_Doc d) {
		System.err.println("[Error]debug(_Doc d) is not implemented in ModelAdaptation family!");
		System.exit(-1);
	}
	
	
	public void savePerf(String filename){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			for(_AdaptStruct u: m_userList){
				writer.write(String.format("%s\t%.5f\t%.5f\n", u.getUserID(), u.getPerfStat().getF1(0), u.getPerfStat().getF1(1)));
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		
	}
	
	public double[] getPerf(){
		return m_perf;
	}
	
	// added by Lin for model performance comparison.
	// print out each user's test review's performance.
	public void printUserPerformance(String filename){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			Collections.sort(m_userList, new Comparator<_AdaptStruct>(){
				@Override
				public int compare(_AdaptStruct u1, _AdaptStruct u2){
					return String.CASE_INSENSITIVE_ORDER.compare(u1.getUserID(), u2.getUserID());
				}
			});
			for(_AdaptStruct u: m_userList){
				writer.write("-----\n");
				writer.write(String.format("%s\t%d\n", u.getUserID(), u.getReviews().size()));
				for(_Review r: u.getReviews()){
					if(r.getType() == rType.ADAPTATION)
						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
					if(r.getType() == rType.TEST){
						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabel(), r.getSource()));
					}
				}
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public ArrayList<_AdaptStruct> getUsers(){
		return m_userList;
	}
}
