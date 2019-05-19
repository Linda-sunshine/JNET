package Classifier.supervised.modelAdaptation.MMB;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures.*;
import structures._Doc.rType;
import structures._PerformanceStat.TestMode;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/***
 * @Auther Lin Gong
 * The joint modeling of network structure modeling and opinionated content modeling
 */

public class MTCLinAdaptWithMMB extends CLinAdaptWithMMB {
	protected int m_dimSup;
	protected int[] m_featureGroupMap4SupUsr; // bias term is at position 0
	protected double[] m_supModel; // linear transformation for super user
	
	protected double m_eta3 = 0.05, m_eta4 = 0.05; // will be used to scale regularization term
	public MTCLinAdaptWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
			String globalModel, String featureGroupMap, String featureGroup4Sup, double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap, betas);
		loadFeatureGroupMap4SupUsr(featureGroup4Sup);
		m_supModel = new double[m_dimSup*2]; // globally shared transformation matrix.
		//construct the new global model for simplicity
		m_supWeights = new double[m_featureSize+1];
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithMMB[dim:%d,dimSup:%d,lmDim:%d,M:%d,rho:%.5f,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]\n",m_dim,m_dimSup,m_lmDim,m_M,m_rho,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_numberOfIterations,m_abNuA[0],m_abNuA[1],m_abNuB[0],m_abNuB[1]);
	}

	@Override
	protected int getVSize() {
		return m_kBar*m_dim*2 + m_dimSup*2;// we have global here.
	}
	
	@Override
	protected void accumulateClusterModels(){
		super.accumulateClusterModels();
		
		// we put the global part in the end
		System.arraycopy(m_supModel, 0, m_models, m_dim*2*m_kBar, m_dimSup*2);
	}
	
	@Override
	protected void initPriorG0() {
		super.initPriorG0();
		
		//sample the global model adaptation parameters
		m_G0.sampling(m_supModel);
	}
	
	@Override
	// R1 over each cluster, R1 over super cluster.
	protected double calculateR1(){
		double R1 = super.calculateR1();
				
		R1 += m_G0.logLikelihood(m_supModel, m_eta3, m_eta4);
		
		// R1 by super model.
		int offset = m_dim*2*m_kBar;
		for(int k=0; k<m_dimSup; k++){
			m_g[offset+k] += m_eta3 * (m_supModel[k]-m_abNuB[0])/m_abNuB[1]/m_abNuB[1]; // scaling
			m_g[offset+k+m_dimSup] += m_eta4 * (m_supModel[m_dimSup+k]-m_abNuA[0])/m_abNuA[1]/m_abNuA[1];
		}
		return R1;
	}
	
	protected double getSupWeights(int n){
		int gid = m_featureGroupMap4SupUsr[n];
		return m_supModel[gid]*m_gWeights[n] + m_supModel[gid+m_dimSup];		
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_Review r = (_Review) review;
		_HDPThetaStar theta = r.getHDPThetaStar();

		int n, k, s; // feature index
		int cIndex = theta.getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		
		int offset = m_dim*2*cIndex, offsetSup = m_dim*2*m_kBar;
		double[] Au = theta.getModel();
		double delta = (review.getYLabel() - logit(review.getSparse(), r)) * weight;
		
		// Bias term for individual user.
		g[offset] -= delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		g[offset + m_dim] -= delta;//b[0]

		// Bias term for super user.
		g[offsetSup] -= delta*Au[0]*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		g[offsetSup + m_dimSup] -= delta*Au[0]; //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			g[offset + k] -= delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
			
			s = m_featureGroupMap4SupUsr[n];
			g[offsetSup + s] -= delta*Au[k]*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			g[offsetSup + m_dimSup + s] -= delta*Au[k]*fv.getValue(); // a_i*x_di
		}
	}
	
	@Override
	protected double gradientTest() {
		double magC = 0, magS = 0 ;
		int offset = m_dim*2*m_kBar;
		for(int i=0; i<offset; i++)
			magC += m_g[i]*m_g[i];
		for(int i=offset; i<m_g.length; i++)
			magS += m_g[i]*m_g[i];
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for clusters: %.5f, super model: %.5f\n", magC/m_kBar, magS);
		return magC + magS;
	}
	
	// Feature group map for the super user.
	protected void loadFeatureGroupMap4SupUsr(String filename){
		// If there is no feature group for the super user.
		if(filename == null){
			m_dimSup = m_featureSize + 1;
			m_featureGroupMap4SupUsr = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<=m_featureSize; i++)
				m_featureGroupMap4SupUsr[i] = i;
			return;
		} else{// If there is feature grouping for the super user, load it.
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
				String[] features = reader.readLine().split(",");//Group information of each feature.
				reader.close();
				
				m_featureGroupMap4SupUsr = new int[features.length + 1]; //One more term for bias, bias->0.
				m_dimSup = 0;
				//Group index starts from 0, so add 1 for it.
				for(int i=0; i<features.length; i++) {
					m_featureGroupMap4SupUsr[i+1] = Integer.valueOf(features[i]) + 1;
					if (m_dimSup < m_featureGroupMap4SupUsr[i+1])
						m_dimSup = m_featureGroupMap4SupUsr[i+1];
				}
				m_dimSup ++;
			} catch(IOException e){
				System.err.format("[Error]Fail to open super user group file %s.\n", filename);
			}
		}
		
		System.out.format("[Info]Feature group size for super user %d\n", m_dimSup);
	}
	
	// Logit function is different from the father class.
	@Override
	protected double logit(_SparseFeature[] fvs, _Review r){
		int k, n;
		double[] Au = r.getHDPThetaStar().getModel();
		double sum = Au[0]*getSupWeights(0) + Au[m_dim];//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			sum += (Au[k]*getSupWeights(n) + Au[m_dim+k]) * fv.getValue();
		}
		return Utils.logistic(sum);
	}

	// Assign the optimized models to the clusters.
	@Override
	protected void setThetaStars(){
		super.setThetaStars();
		
		// Assign model to super user.
		System.arraycopy(m_models, m_dim*2*m_kBar, m_supModel, 0, m_dimSup*2);
	}
	
	//apply current model in the assigned clusters to users
	@Override
	protected void evaluateModel() {
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
		
		super.evaluateModel();	
	}
	
	public void setR2TradeOffs(double eta3, double eta4) {
		m_eta3 = eta3;
		m_eta4 = eta4;
	}	
	public void printClusterInfo(){
		int[] sizes = new int[m_kBar];
		for(int i=0; i<m_kBar; i++){
			sizes[i] = m_hdpThetaStars[i].getMemSize();
		}
		Arrays.sort(sizes);
		for(int i=sizes.length-1; i>=0; i--)
			System.out.print(sizes[i]+"\t");
		System.out.println();
	}
	
	// Save the sentiment models of thetaStars
	@Override
	public void saveClusterModels(String clusterdir){
	
		PrintWriter writer;
		String filename;
		File dir = new File(clusterdir);
		double[] Ac;
		int ki, ks;
		try{
			if(!dir.exists())
				dir.mkdirs();
			for(int i=0; i<m_kBar; i++){
				Ac = m_hdpThetaStars[i].getModel();
				m_pWeights = new double[m_gWeights.length];
				for(int n=0; n<=m_featureSize; n++){
					ki = m_featureGroupMap[n];
					ks = m_featureGroupMap4SupUsr[n];
					m_pWeights[n] = Ac[ki]*(m_supModel[ks]*m_gWeights[n] + m_supModel[ks+m_dimSup])+Ac[ki+m_dim];
				}
				filename = String.format("%s/%d.classifier", clusterdir, m_hdpThetaStars[i].getIndex());
				writer = new PrintWriter(new File(filename));
				for(int v=0; v<m_pWeights.length; v++){
					if(v == m_pWeights.length-1)
						writer.write(Double.toString(m_pWeights[v]));
					else
						writer.write(m_pWeights[v]+",");
				}
				writer.close();
			}
		} catch (IOException e){
				e.printStackTrace();
		}
	}
	// save the user mixture membership into a file
	public void saveUserMembership(String clusterdir, String filename){
		PrintWriter writer;
		File dir = new File(clusterdir);
		if(!dir.exists())
			dir.mkdirs();
		
		try {
			writer = new PrintWriter(new File(clusterdir+"/UserMembership.txt"));
			for(_AdaptStruct u: m_userList){
				_MMBAdaptStruct user = (_MMBAdaptStruct) u;
				writer.write(String.format("%s\n", u.getUserID()));
				// write the clusters with edges first
				for(_HDPThetaStar theta: user.getHDPTheta4Edge()){
					writer.write(String.format("(%d, %d, %d, %d)\t", theta.getIndex(), user.getHDPThetaMemSize(theta), user.getHDPThetaOneEdgeSize(theta, 0), user.getHDPThetaOneEdgeSize(theta, 1)));
				}
				// write the clusters with members then
				for(_HDPThetaStar theta: user.getHDPTheta4Rvw()){
					if(!user.getHDPTheta4Edge().contains(theta))
						writer.write(String.format("(%d, %d, %d, %d)\t", theta.getIndex(), user.getHDPThetaMemSize(theta), user.getHDPThetaOneEdgeSize(theta, 0), user.getHDPThetaOneEdgeSize(theta, 1)));
				}
				writer.write("\n");
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	// print out related information for analysis
	public void saveEverything(String dir){
		String sentimentDir = String.format("%s/sentiment_models/", dir);
		String lmDir = String.format("%s/lm_models/", dir);
		String userMemFile = String.format("%s/UserMembership.txt", dir);
		
		// save cluster information: sentiment model, language model, user membership
		saveClusterModels(sentimentDir);
		saveUserMembership(dir, userMemFile);
		saveClusterLanguageModels(lmDir);
		
		String statFile = String.format("%s/stat.txt", dir);
		String edgeFile = String.format("%s/edge_assignment.txt", dir);
		String BFile = String.format("%s/B.txt", dir);
		String perfFile = String.format("%s/mmb_perf.txt", dir);
		
		printStat(statFile);
		printEdgeAssignment(edgeFile);
		printBMatrix(BFile);
		printUserPerformance(perfFile);
	}
	
	
	HashMap<String, ArrayList<Double[]>> m_perfMap = new HashMap<>();
	String[] m_keys;
	@Override
	public double trainTrace(String data, long time){
		
		m_perfMap.clear();
		m_keys = new String[]{"doc", "edge", "m", "exp", "doc_all", "edge_all", "m_all", "exp_all"};
		m_perfMap.put("doc", new ArrayList<Double[]>());
		m_perfMap.put("edge", new ArrayList<Double[]>());
		m_perfMap.put("m", new ArrayList<Double[]>());
		m_perfMap.put("exp", new ArrayList<Double[]>());
		
		m_perfMap.put("doc_all", new ArrayList<Double[]>());
		m_perfMap.put("edge_all", new ArrayList<Double[]>());
		m_perfMap.put("m_all", new ArrayList<Double[]>());
		m_perfMap.put("exp_all", new ArrayList<Double[]>());
		
		System.out.print(toString());
		
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		double likelihoodX = 0, likelihoodY = 0;
		int count = 0;
		
		double likelihoodE = 0;
		// clear user performance, init cluster assignment, assign each review to one cluster
		init();	
		initThetaStars_Edges_Joint();
		sanityCheck();
		
		// Burn in period for doc.
		while(count++ < m_burnIn){
			calculate_E_step();
			calculate_E_step_Edge();
			calculate_M_step();
		}
		
		try{
			String traceFile = String.format("%s_iter_%d_burnin_%d_thin_%d_%b_%d.txt", data, m_numberOfIterations, m_burnIn, m_thinning, m_jointAll, time); 
			PrintWriter writer = new PrintWriter(new File(traceFile));
			// EM iteration.
			for(int i=0; i<m_numberOfIterations; i++){
				
				// E step
				long start = System.currentTimeMillis();
				
				// record the performance after sampling documents
				calculate_E_step();
				recordPerformance("doc");
			
				// record the performance after sampling edges
				calculate_E_step_Edge();
				recordPerformance("edge");

				long end = System.currentTimeMillis();
				printClusterInfo(start, end);
				
				// M step
				likelihoodY = calculate_M_step();
				recordPerformance("m");
				
				// accumulate the likelihood
				likelihoodX = accumulateLikelihoodX();				
				likelihoodE = accumulateLikelihoodEMMB();
				likelihoodE += (m_MNL[2]/2)*Math.log(1-m_rho);
				
				curLikelihood = likelihoodY + likelihoodX + likelihoodE;
				delta = (lastLikelihood - curLikelihood)/curLikelihood;
				
				// evaluate the model
				if (i%m_thinning==0){
					evaluateModel();
					test();
					for(_AdaptStruct u: m_userList)
						u.getPerfStat().clear();
				}
				
				// record the expectation of all the predictions too for comparison
				m_perfMap.get("exp").add(new Double[]{m_perf[0], m_perf[1]});
				m_perfMap.get("exp_all").add(new Double[]{m_microStat.getF1(0), m_microStat.getF1(1)});
				writer.write(String.format("%.5f\t%.5f\t%.5f\t%.5f\t%d\t%.5f\t%.5f\n", likelihoodY, likelihoodX, likelihoodE, delta, m_kBar, m_perf[0], m_perf[1]));
				System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
				if(Math.abs(delta) < m_converge)
					break;
				lastLikelihood = curLikelihood;
			}
			saveDetailPerformance(data, time);
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}
	
	// record the performance of one step (sample docs, sample edges, optimize models)
	public void recordPerformance(String key){
		
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
		// calculate the posterior probabilit for each user
		calculateClusterProbPerUser();

		// predict the label for each review in real time
		testUserClusterPerf();
		
		m_perfMap.get(key).add(new Double[]{m_perf[0], m_perf[1]});
		m_perfMap.get(key+"_all").add(new Double[]{m_microStat.getF1(0), m_microStat.getF1(1)});

		// clear the performance data for each cluster
		for(int k=0; k<m_kBar; k++){
			m_hdpThetaStars[k].getPerfStat().clear();
		}
		// clear the performance data for each user
		for(_AdaptStruct u: m_userList){
			u.getPerfStat().clear();
		}
		System.out.format("-----------Finish recording performance after %s !-----------\n\n", key);
	}

	// test the model performance with independent prediction for each review.
	public void testUserClusterPerf(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					_MMBAdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_MMBAdaptStruct) m_userList.get(i+core);
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
									int predL = user.predictIndependently(r); // evoke user's own model
									r.setPredictLabel(predL);
									r.getHDPThetaStar().getPerfStat().addOnePredResult(predL, trueL);
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
		
		// calculate the F1 for each cluster
		for(int k=0; k<m_kBar; k++){
			m_hdpThetaStars[k].getPerfStat().calculatePRF();	
		}
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
		}
		calcMicroPerfStat();
		System.out.println("\nMacro F1:");
		// macro average and standard deviation.
		for(int i=0; i<m_classNo; i++){
			double[] avgStd = calcAvgStd(macroF1.get(i));
			m_perf[i] = avgStd[0];
			System.out.format("Class %d: %.4f+%.4f\t", i, avgStd[0], avgStd[1]);
		}
		System.out.println();
	}

	public void saveDetailPerformance(String data, long time){
		try{
			String perfFile = String.format("%s_iter_%d_burnin_%d_thin_%d_detail_%d.txt", data, m_numberOfIterations, m_burnIn, m_thinning, time); 
			PrintWriter writer = new PrintWriter(new File(perfFile));
			// EM iteration.
			writer.write("doc_neg doc_pos edge_neg edge_pos m_neg m_pos exp_neg exp_pos ");
			writer.write("doc_all_neg doc_all_pos edge_all_neg edge_all_pos m_all_neg m_all_pos exp_all_neg exp_all_pos\n");
			int step = m_perfMap.get("doc").size();
			for(int i=0; i<step; i++){
				for(String key: m_keys){
					Double[] perf = m_perfMap.get(key).get(i);
					writer.write(String.format("%.5f\t%.5f\t",perf[0], perf[1]));
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}

//	@Override
//	public void evaluateModel(){
//		for(int i=0; i<m_featureSize+1; i++)
//			m_supWeights[i] = getSupWeights(i);
//		calculateClusterProbPerUser();
//
//	}
}

