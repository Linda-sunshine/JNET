package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import cern.jet.random.tfloat.FloatUniform;
import structures.MyPriorityQueue;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;
import structures._Doc.rType;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import utils.Utils;

public class CLRWithDP extends LinAdapt {
	protected int m_M = 6, m_kBar = 0; // The number of auxiliary components.
	protected int m_numberOfIterations = 50;
	protected int m_burnIn = 10, m_thinning = 3;// burn in time, thinning time.
	protected double m_converge = 1e-6;
	protected double m_alpha = 1; // Scaling parameter of DP.
	protected double m_pNewCluster; // proportion of sampling a new cluster, to be assigned before EM starts
	protected NormalPrior m_G0; // prior distribution
	protected boolean m_vctMean = true; // flag to determine whether we should use w_0 as prior for w_u
	
	//structure for multi-threading
	protected boolean m_multiThread = true; // if we will use multi-threading in M-step
	protected double[] m_fValues;
	protected double[][] m_gradients;

	// Parameters of the prior for the intercept and coefficients.
	protected double[] m_abNuA = new double[]{0, 1}; // N(0,1) for shifting in adaptation based models

	// parameter for global weights.
	public static double m_q = .10;// the wc + m_q*wg;

	protected double[] m_models; // model parameters for clusters to be used in l-bfgs optimization
	public static _thetaStar[] m_thetaStars = new _thetaStar[1000];//to facilitate prediction in each user 

	public CLRWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel){
		super(classNo, featureSize, featureMap, globalModel, null);
		m_dim = m_featureSize + 1; // to add the bias term
	}	
	
	public CLRWithDP(int classNo, int featureSize, String globalModel){
		super(classNo, featureSize, globalModel, null);
		m_dim = m_featureSize + 1; // to add the bias term
	}	
	
	protected void assignClusterIndex(){
		for(int i=0; i<m_kBar; i++)
			m_thetaStars[i].setIndex(i);
	}
	
	protected void accumulateClusterModels(){
		if (m_models==null || m_models.length!=getVSize())
			m_models = new double[getVSize()];
		
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_thetaStars[i].getModel(), 0, m_models, m_dim*i, m_dim);
	}
	
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
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

				prob = calcLogLikelihood(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
				probs[k] = Math.exp(prob);//this will be in real space!
			}
			Utils.L1Normalization(probs);
			user.setClusterPosterior(probs);

			user.setThetaStar(oldTheta);//restore the cluster assignment during EM iterations
		}
	}
	
	//added by Lin. Calculate the function value of the new added instance.
	protected double calcLogLikelihood4Posterior(_AdaptStruct user){
		double L = 0; //log likelihood.
		double Pi = 0;
			
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION  && review.getType() != rType.TEST)
					continue; // only touch the adaptation data
				
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
		if(m_LNormFlag)
			return L/getAdaptationSize(user);
		else
			return L;
	}

	// The main MCMC algorithm, assign each user to clusters.
	protected void calculate_E_step(){
		_thetaStar curThetaStar;
		_DPAdaptStruct user;
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0) 
				continue;
			curThetaStar = user.getThetaStar();
			curThetaStar.updateMemCount(-1);
			
			if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
				swapTheta(m_kBar-1, findThetaStar(curThetaStar)); // move it back to \theta*
				m_kBar --;
			}
			sampleOneInstance(user);
		}
	}
	
	protected double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_kBar; i++)
			R1 += m_G0.logLikelihood(m_thetaStars[i].getModel(), m_eta1, 0);//the last is dummy input
		
		// Gradient by the regularization.
		if (m_G0.hasVctMean()) {//we have specified the whole mean vector
			for(int i=0; i<m_kBar*m_dim; i++) 
				m_g[i] += m_eta1 * (m_models[i]-m_gWeights[i%m_dim]) / (m_abNuA[1]*m_abNuA[1]);
		} else {//we only have a simple prior
			for(int i=0; i<m_kBar*m_dim; i++)
				m_g[i] += m_eta1 * (m_models[i]-m_abNuA[0]) / (m_abNuA[1]*m_abNuA[1]);
		}
		return R1;
	}
	
	// Sample the weights given the cluster assignment.
	protected double calculate_M_step(){
		assignClusterIndex();
		return estPhi();		
	}
	
	protected double estPhi(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;
		int displayCount = 0;		

		initLBFGS();// init for lbfgs.
		
		try{
			do{
				Arrays.fill(m_g, 0); // initialize gradient
				
				//regularization part
				fValue = calculateR1();
				if (m_multiThread)
					fValue += logLikelihood_MultiThread();
				else
					fValue += logLikelihood();
				
				if (m_displayLv==2) {
					System.out.print("Fvalue is " + fValue + "\t");
					gradientTest();
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					
					if (++displayCount%100==0)
						System.out.println();
				} 
				
				LBFGS.lbfgs(m_g.length, 6, m_models, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				setThetaStars();
				oldFValue = fValue;
				
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.err.println("LBFGS fails!!!!");
			e.printStackTrace();
		}	
		return oldFValue;
	}
	
	//apply current model in the assigned clusters to users
	protected void evaluateModel() {//this should be only used in batch testing!
		System.out.println("[Info]Accumulating evaluation results during sampling...");

		//calculate cluster posterior p(c|u)
		calculateClusterProbPerUser();
		
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();		
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					_DPAdaptStruct user;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
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

	int findThetaStar(_thetaStar theta) {
		for(int i=0; i<m_kBar; i++)
			if (theta == m_thetaStars[i])
				return i;
		
		System.err.println("[Error]Hit unknown theta star when searching!");
		return -1;// impossible to hit here!
	}
	
	@Override
	protected int getVSize() {
		return m_kBar*m_dim;
	}

	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		gradientByFunc(u, review, weight, m_g);
	}

	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
	
		int n; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
	
		int offset = m_dim*cIndex;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
	
		//Bias term.
		g[offset] -= delta; //x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			g[offset + n] -= delta * fv.getValue();
		}
	}

	@Override
	protected double gradientTest() {
		double mag = 0 ;
		for(int i=0; i<m_g.length; i++)
			mag += m_g[i]*m_g[i];

		if (m_displayLv==2)
			System.out.format("Gradient magnitude: %.5f\n", mag/m_kBar);
		return mag;
	}
	
	
	protected void initPriorG0() {
		if (m_vctMean)
			m_G0 = new NormalPrior(m_gWeights, m_abNuA[1]);//using the global model as prior
		else
			m_G0 = new NormalPrior(m_abNuA[0], m_abNuA[1]);//only for shifting
	}
	
	// Assign cluster assignment to each user.
	protected void initThetaStars(){
		Arrays.fill(m_thetaStars, null);
		initPriorG0();
		
		m_pNewCluster = Math.log(m_alpha) - Math.log(m_M);//to avoid repeated computation
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0)
				continue;
			sampleOneInstance(user);
		}		
	}
	
	@Override
	protected void init(){
		super.init();
		initThetaStars();
		
		//init the structures for multi-threading
		if (m_multiThread) {
			int numberOfCores = Runtime.getRuntime().availableProcessors();
			m_fValues = new double[numberOfCores];
			m_gradients = new double[numberOfCores][]; 
		}
	}

	//very inefficient, a per cluster optimization procedure will not have this problem
	@Override
	protected void initLBFGS(){
		if (m_g==null || m_g.length!=getVSize()) {
			m_g = new double[getVSize()];
			m_diag = new double[getVSize()];
			
			if (m_multiThread) {
				int numberOfCores = Runtime.getRuntime().availableProcessors();
				for(int k=0; k<numberOfCores; k++)
					m_gradients[k] = new double[getVSize()];
			}
		}
		
		accumulateClusterModels();
	}
	
	protected double logLikelihood() {
		_DPAdaptStruct user;
		double fValue = 0;
		
		// Use instances inside one cluster to update the thetastar.
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			fValue -= calcLogLikelihood(user);
			gradientByFunc(user); // calculate the gradient by the user.
		}
		return fValue;
	}
	
	protected double logLikelihood_MultiThread() {
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();		
		
		//init the shared structure		
		Arrays.fill(m_fValues, 0);
		for(int k=0; k<numberOfCores; ++k){
			Arrays.fill(m_gradients[k], 0);
			
			threads.add((new Thread() {
				int core, numOfCores;
				double[] m_gradient, m_fValue;

				@Override
				public void run() {
					_DPAdaptStruct user;
					try {						
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_DPAdaptStruct)m_userList.get(i+core);
							if(user.getAdaptationSize() == 0)
								continue;
							m_fValue[core] -= calcLogLikelihood(user);
							
							for(_Review review:user.getReviews()){
								if (review.getType() != rType.ADAPTATION )//&& review.getType() != rType.TEST)
									continue;								
								
								gradientByFunc(user, review, 1.0, this.m_gradient);//weight all the instances equally
							}			
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core, int numOfCores, double[] gradient, double[] f) {
					this.core = core;
					this.numOfCores = numOfCores;
					this.m_gradient = gradient;
					this.m_fValue = f;
					
					return this;
				}
			}).initialize(k, numberOfCores, m_gradients[k], m_fValues));
			
			threads.get(k).start();
		}
		
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for(int k=0;k<numberOfCores;++k)
			Utils.scaleArray(m_g, m_gradients[k], 1);
		return Utils.sumOfArray(m_fValues);
	}	

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
//			m_userList.add(new _DPAdaptStruct(user, user.getUserID()));
			m_userList.add(new _DPAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];		
	}
	
	@Override	
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		double sum = Utils.dotProduct(((_DPAdaptStruct)u).getThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}
	
	public int predict(_AdaptStruct user, _thetaStar theta){
		double[] As;
		double sum;
		int m, n, predL = 0, count = 0;
		for(_Review r: user.getReviews()){
			if(r.getType() == rType.TEST){
				As = theta.getModel();
				sum = As[0]*MTCLinAdaptWithDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: r.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*MTCLinAdaptWithDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
				}
				if(sum > 0.5) 
					predL = 1;
				if(predL == r.getYLabel())
					count++;
			}
		}
		return count;
	}

	public void printInfo(){
		
		MyPriorityQueue<_RankItem> clusterRanker = new MyPriorityQueue<_RankItem>(5);		

		//clear the statistics
		for(int i=0; i<m_kBar; i++){ 
			m_thetaStars[i].resetCount();
			clusterRanker.add(new _RankItem(i, m_thetaStars[i].getMemSize()));
		}

		//collect statistics across users in adaptation data
		_thetaStar theta = null;
		for(int i=0; i<m_userList.size(); i++) {
			_DPAdaptStruct user = (_DPAdaptStruct)m_userList.get(i);
			theta = user.getThetaStar();
			
			for(_Review review:user.getReviews()){
				if (review.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				else if (review.getYLabel()==1)
					theta.incPosCount();
				else
					theta.incNegCount();
			}
		}
		
		System.out.print("[Info]Clusters:");
		for(int i=0; i<m_kBar; i++)
			System.out.format("%s\t", m_thetaStars[i].showStat());	
		System.out.print(String.format("\n[Info]%d Clusters are found in total!\n", m_kBar));			
	}

	void printTopWords(_thetaStar cluster) {
		MyPriorityQueue<_RankItem> wordRanker = new MyPriorityQueue<_RankItem>(10);
		double[] phi = cluster.getModel();
	
		//we will skip the bias term!
		System.out.format("Cluster %d (%d)\n[positive]: ", cluster.getIndex(), cluster.getMemSize());
		for(int i=1; i<phi.length; i++) 
			wordRanker.add(new _RankItem(i, phi[i]));//top positive words with expected polarity

		for(_RankItem it:wordRanker)
			System.out.format("%s:%.3f\t", m_features[it.m_index], phi[it.m_index]);
	
		System.out.format("\n[negative]: ");
		wordRanker.clear();
		for(int i=1; i<phi.length; i++) 
			wordRanker.add(new _RankItem(i, -phi[i]));//top negative words

		for(_RankItem it:wordRanker)
			System.out.format("%s:%.3f\t", m_features[it.m_index], phi[it.m_index]);	
	}

	// Set a bunch of parameters.
	public void setAlpha(double a){
		m_alpha = a;
	}
	
	public void setBurnIn(int n){
		m_burnIn = n;
	}
	
	public void setM(int m){
		m_M = m;
	}
	
	public void setsdA(double v){
		m_abNuA[1] = v;
	}
	public void setNumberOfIterations(int num){
		m_numberOfIterations = num;
	}
	
	@Override
	protected void setPersonalizedModel() {
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			user.setPersonalizedModel(user.getThetaStar().getModel());
		}
	}
	
	// Assign the optimized weights to the cluster.
	protected void setThetaStars(){
		double[] beta;
		for(int i=0; i<m_kBar; i++){
			beta = m_thetaStars[i].getModel();
			System.arraycopy(m_models, i*m_dim, beta, 0, m_dim);
		}
	}
	
	public void setThinning(int t){
		m_thinning = t;
	}
	
	public void setMultiTheadFlag(boolean b){
		m_multiThread = b;
	}
	
	
	// Sample thetaStars.
	protected void sampleThetaStars(){
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_thetaStars[m] == null) {
				if (this instanceof CLinAdaptWithDP)// this should include all the inherited classes for adaptation based models
					m_thetaStars[m] = new _thetaStar(2*m_dim);
				else
					m_thetaStars[m] = new _thetaStar(m_dim);
			}
			m_G0.sampling(m_thetaStars[m].getModel());
		}
	}
	
	// Sample one instance's cluster assignment.
	protected void sampleOneInstance(_DPAdaptStruct user){
		// sanity check
		if(user.getAdaptationSize() == 0)
			System.out.println("The user does not have adaptation data!");
		
		double likelihood, logSum = 0;
		int k;
		
		//reset thetaStars
		sampleThetaStars();
		for(k=0; k<m_kBar+m_M; k++){
			user.setThetaStar(m_thetaStars[k]);
			likelihood = calcLogLikelihood(user);
			if (k<m_kBar)
				likelihood += Math.log(m_thetaStars[k].getMemSize());
			else
				likelihood += m_pNewCluster;
			 
			m_thetaStars[k].setProportion(likelihood);//this is in log space!
			
			if (k==0)
				logSum = likelihood;
			else
				logSum = Utils.logSum(logSum, likelihood);
//			System.out.print(String.format("%.4f\t%.4f\n",likelihood, logSum));

		}
		
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		
		k = 0;
		double newLogSum = m_thetaStars[0].getProportion();
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_thetaStars[k].getProportion());
		} while (k<m_kBar+m_M);
//		System.out.print(String.format("------kBar:%d, k:%d-----------", m_kBar, k));

		if (k==m_kBar+m_M) {
			System.err.println("[Warning]Hit the very last element in theatStar!");
			k--; // we might hit the very last
		}
		
		m_thetaStars[k].updateMemCount(1);
		user.setThetaStar(m_thetaStars[k]);
		if(k >= m_kBar){
			swapTheta(m_kBar, k);
			m_kBar++;
		}
	}	
	
	protected void swapTheta(int a, int b) {
		_thetaStar cTheta = m_thetaStars[a];
		m_thetaStars[a] = m_thetaStars[b];
		m_thetaStars[b] = cTheta;// kBar starts from 0, the size decides how many are valid.
	}
	
	public void setGlobalModel(int fvSize){
		m_gWeights = new double[fvSize+1];
	}
	
	public void setQ(double q){
		m_q = q;
	}
	
	public void saveClusterModels(String model){
		PrintWriter writer;
		String filename;
		File dir = new File(model);
		_thetaStar theta;
		double[] weight;
		try{
			if(!dir.exists())
				dir.mkdirs();
			for(int i=0; i<m_kBar; i++){
				theta = m_thetaStars[i]; 
				filename = String.format("%s/%d.classifier", model, theta.getIndex());
				writer = new PrintWriter(new File(filename));
				weight = theta.getModel();
				for(int v=0; v<weight.length; v++){
					if(v == weight.length-1)
						writer.write(Double.toString(weight[v]));
					else
						writer.write(weight[v]+",");
				}
				writer.close();
			}
			writer = new PrintWriter(new File(model+"/ClusterMember.txt"));
			for(_AdaptStruct u: m_userList){
				_DPAdaptStruct user = (_DPAdaptStruct) u;
				writer.write(user.getUserID()+"\t"+user.getThetaStar().getIndex()+"\n");
			}
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
	
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		init(); // clear user performance and init cluster assignment		
		
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			lastLikelihood = calculate_M_step();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			calculate_E_step();
			
			// Optimize the parameters
			curLikelihood = calculate_M_step();

			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0)
				evaluateModel();
			
			printInfo();
			System.out.print(String.format("[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.6f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		evaluateModel(); // we do not want to miss the last sample?!
		setPersonalizedModel();
		return curLikelihood;
	}

	// added by Lin for tracking trace. 
	public double trainTrace(String data, long start){
		
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
			
		init(); // clear user performance and init cluster assignment		
			
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			lastLikelihood = calculate_M_step();
		}
		try{
			String traceFile = String.format("%s_iter_%d_burnin_%d_thin_%d_%d.txt", data, m_numberOfIterations, m_burnIn, m_thinning, start); 
			PrintWriter writer = new PrintWriter(new File(traceFile));
			// EM iteration.
			for(int i=0; i<m_numberOfIterations; i++){
				// Cluster assignment, thinning to reduce auto-correlation.
				calculate_E_step();
				
				// Optimize the parameters
				curLikelihood = calculate_M_step();
				
				delta = (lastLikelihood - curLikelihood)/curLikelihood;
				if (i%m_thinning==0){
					evaluateModel();
					test();
					for(_AdaptStruct u: m_userList)
						u.getPerfStat().clear();
				}
				writer.write(String.format("%.5f\t%.5f\t%d\t%.5f\t%.5f\n", curLikelihood, delta, m_kBar, m_perf[0], m_perf[1]));
				System.out.print(String.format("[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
				if(Math.abs(delta) < m_converge)
					break;
				lastLikelihood = curLikelihood;
		}
		writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		evaluateModel(); // we do not want to miss the last sample?!
		setPersonalizedModel();
		return curLikelihood;
	}

	@Override
	public String toString() {
		return String.format("CLRWithDP[dim:%d,M:%d,alpha:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim, m_M, m_alpha, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
}
