package Classifier.supervised.modelAdaptation.MMB;

import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import cern.jet.random.tdouble.Beta;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tfloat.FloatUniform;
import org.apache.commons.math3.distribution.BinomialDistribution;
import structures.*;
import structures._Doc.rType;
import structures._HDPThetaStar._Connection;
import structures._PerformanceStat.TestMode;
import utils.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class CLRWithMMB extends CLRWithHDP {
	// sparsity parameter
	protected double m_rho = 0.01; 
	// As we store all the indicators for all the edges(even edges from background model), we maintain a matrix for indexing.
	protected _HDPThetaStar[][] m_indicator;
	
	// prob for the new cluster in sampling mmb edges.
	protected double[] m_pNew = new double[2]; 
	// parameters used in the gamma function in mmb model, prior of B~beta(a, b), prior of \rho~Beta(c, d)
	protected double[] m_abcd = new double[]{0.1, 0.01, 2, 2}; 
	// Me: total number of edges eij=0;Ne: total number of edges eij=1 from mmb; Le: total number of edges eij=0 from background model.
	protected double[] m_MNL = new double[3];
	// Bernoulli distribution used in deciding whether the edge belongs to mmb or background model.
	protected BinomialDistribution m_bernoulli;
	// whether we perform joint sampling for all zero edges or just background edges
	protected boolean m_jointAll = false;
	
	// for debug purpose
	protected HashMap<String, ArrayList<Integer>> stat = new HashMap<>();
	protected static double m_log2 = Math.log(2);
	ArrayList<Integer> mmb_0 = new ArrayList<Integer>();
	ArrayList<Integer> mmb_1 = new ArrayList<Integer>();
	ArrayList<Integer> bk_0 = new ArrayList<Integer>();
	
	public CLRWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
		initStat();
	} 
	
	public CLRWithMMB(int classNo, int featureSize, String globalModel,
			double[] betas) {
		super(classNo, featureSize, globalModel, betas);
		initStat();
	} 
	
	// add the edge assignment to corresponding cluster
	public void addConnection(_MMBAdaptStruct ui, _MMBAdaptStruct uj, int e){
		_HDPThetaStar theta_g, theta_h;
		theta_g = ui.getOneNeighbor(uj).getHDPThetaStar();
		theta_h = uj.getOneNeighbor(ui).getHDPThetaStar();
		theta_g.addConnection(theta_h, e);
		theta_h.addConnection(theta_g, e);
	}
	
	// traverse all the clusters to get the likelihood given by mmb edges
	protected double accumulateLikelihoodEMMB(){
		double likelihoodE = 0;
		_Connection connection;
		int e_0, e_1;
		_HDPThetaStar theta_g, theta_h;
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=g; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				if(!theta_g.hasConnection(theta_h)) continue;
				connection = theta_g.getConnection(theta_h);
				e_1 = connection.getEdge()[1];
				e_0 = connection.getEdge()[0];
				//likelihoodE: m*log(rho*(a+e_1))/(a+b+e_0+e_1))+n*log(rho*(b+e_0))/(a+b+e_0+e_1))
				likelihoodE += (e_0+e_1)*Math.log(m_rho)+e_1*Math.log(m_abcd[0]+e_1)+
						e_0*Math.log(m_abcd[1]+e_0)-(e_0+e_1)*Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
//				likelihoodE += e_1*Math.log(m_abcd[0]+e_1)+e_0*Math.log(m_abcd[1]+e_0)
//						-(e_0+e_1)*Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
			}
		}
		return likelihoodE;
	}
	
	// traverse all the clusters to get the decomposed likelihood
	// 0: m*log(rho*zBz); 1: n*log(1-rho*zBz); 2:n*log(rho(1-zBz))
	protected double[] accumulateDecomposedLikelihoodEMMB(){
		double likelihoodE[] = new double[4];
		_Connection connection;
		int e_0, e_1;
		double logRho = Math.log(m_rho), log_zBz = 0, zBz = 0;
		_HDPThetaStar theta_g, theta_h;
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=g; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				if(!theta_g.hasConnection(theta_h)) continue;
				connection = theta_g.getConnection(theta_h);
				e_1 = connection.getEdge()[1];
				e_0 = connection.getEdge()[0];
				log_zBz = Math.log(m_abcd[0]+e_1) - Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
				zBz = (m_abcd[0] + e_1)/(m_abcd[0]+m_abcd[1]+e_0+e_1);
				likelihoodE[0] += e_1*(logRho + log_zBz);
				likelihoodE[1] += e_0*(Math.log(1-m_rho * zBz));
				likelihoodE[2] += e_0*(logRho + Math.log(1-zBz));
				//likelihoodE: m*log(rho*(a+e_1))/(a+b+e_0+e_1))+n*log(rho*(b+e_0))/(a+b+e_0+e_1))
//				likelihoodE += (e_0+e_1)*Math.log(m_rho)+e_1*Math.log(m_abcd[0]+e_1)+
//						e_0*Math.log(m_abcd[1]+e_0)-(e_0+e_1)*Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
//				likelihoodE += e_1*Math.log(m_abcd[0]+e_1)+e_0*Math.log(m_abcd[1]+e_0)
//						-(e_0+e_1)*Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
			}
		}
		return likelihoodE;
	}
	
	// calculate the probability for generating new clusters in sampling edges.
	protected void calcProbNew(){
		// if e_ij = 1, p = \rho*a/(a+b)
		m_pNew[1] = Math.log(m_rho) + Math.log(m_abcd[0]) - Math.log(m_abcd[0] + m_abcd[1]);
		// if e_ij = 0, p = (1-\rho*a/(a+b))
		m_pNew[0] = Math.log(1 - m_rho * m_abcd[0] /(m_abcd[0] + m_abcd[1]));
	}

	@Override
	// calculate the group popularity in sampling documents and edges.
	protected double calcGroupPopularity(_HDPAdaptStruct u, int k, double gamma_k){
		_MMBAdaptStruct user= (_MMBAdaptStruct) u;
		return user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k + user.getHDPThetaEdgeSize(m_hdpThetaStars[k]);
	}
	
	/*** p(e=1)=\rho*B_gh, p(e=0)=1-\rho*B_gh
	/* corresponding predictive posterior distribution is:
	/* e=1: \rho*(a+e_1)/(a+b+e_0+e_1)
	/* e=0: 1-\rho*(a+e_1)/(a+b+e_0+e_1) **/
	public double calcLogLikelihoodEMarginal(_HDPThetaStar theta_g, _HDPThetaStar theta_h, int e){
		
		double prob = 0;
		double e_0 = 0, e_1 = 0;
		// some background model may have non-existing cluster
		if(theta_g.isValid() && theta_h.isValid()){
			e_0 = theta_g.getConnectionEdgeCount(theta_h, 0);
			e_1 = theta_g.getConnectionEdgeCount(theta_h, 1);
		} else{
			System.out.println("[Error]Invalid thetas inside calcLogLikelihoodEMarginal()!");
		}
		prob = Math.log(m_rho) + Math.log(m_abcd[0] + e_1) - Math.log(m_abcd[0] + m_abcd[1] + e_0 + e_1);
		return e == 1 ? prob : Math.log(1 - Math.exp(prob));
	}
	
	/**e=0: \int_{B_gh} \rho*(1-B_{gh})*prior d_{B_{gh}} 
	/* e=1: \int_{B_gh} \rho*B_{gh}*prior d_{B_{gh}} 
	/* prior is Beta(a+e_1, b+e_0)***/
	public double calcLogLikelihoodE(_HDPThetaStar theta_g, _HDPThetaStar theta_h, int e){
		
		double prob = 0;
		double e_0 = 0, e_1 = 0;
		// some background model may have non-existing cluster
		if(theta_g.isValid() && theta_h.isValid()){
			e_0 = theta_g.getConnectionEdgeCount(theta_h, 0);
			e_1 = theta_g.getConnectionEdgeCount(theta_h, 1);
		}
		prob = e == 0 ? Math.log(m_abcd[1] + e_0) : Math.log(m_abcd[0] + e_1);
		prob += Math.log(m_rho) - Math.log(m_abcd[0] + m_abcd[1] + e_0 + e_1);
		return prob;
	}
	
	
	int m_multipleE = 1;
	public void setMultipleE(int e){
		m_multipleE = e;
	}
	
	// Decide which sampling method to take
	public void calculate_E_step_Edge(){
		if(m_jointAll)
			calculate_E_step_Edge_joint_all();
		else
			calculate_E_step_Edge_joint_bk();
	}

	// calculate the mixture for train user based on review assignment and edge assignment
	public void calcMix4UsersWithAdaptReviews(_MMBAdaptStruct user){
		double sum = 0;
		double[] probs = new double[m_kBar];
		_HDPThetaStar theta;
		// The set of clusters for review and edge could be different, just iterate over kBar
		for(int k=0; k<m_kBar; k++){
			theta = m_hdpThetaStars[k];
			probs[k] = user.getHDPThetaMemSize(theta) + user.getHDPThetaEdgeSize(theta);
			sum += probs[k];
		}
		for(int k=0; k<m_kBar; k++){
			probs[k] /= sum;
		}
		user.setMixture(probs);
	}
	
	// calculate the mixture for test user based on review content and group popularity
	// we cannot touch friendship, thus, only review assignment is utilized for mixture
	public void calcMix4UsersNoAdaptReviews(_MMBAdaptStruct user){
		int cIndex = 0;
		double prob, logSum, sum = 0;
		double[] probs = new double[m_kBar];
		_HDPThetaStar curTheta;
		
		// calculate the cluster assignment for each review first
		for(_Review r: user.getReviews()){
			// suppose all reviews are test review in this setting
			if (r.getType() != rType.TEST)
				continue;
			
			for(int k=0; k<probs.length; k++){
				curTheta = m_hdpThetaStars[k];
				r.setHDPThetaStar(curTheta);
				prob = calcLogLikelihoodX(r) + Math.log(calcGroupPopularity(user, k, curTheta.getGamma()));
				probs[k] = prob;
			}
			// normalize the prob 
			logSum = Utils.logSumOfExponentials(probs);
			for(int k=0; k<probs.length; k++)
				probs[k] -= logSum;
			
			// take the cluster that has maximum prob as the review's cluster assignment
			curTheta = m_hdpThetaStars[Utils.argmax(probs)];
			r.setHDPThetaStar(curTheta);
			// update the cluster assignment for the user
			user.incHDPThetaStarMemSize(r.getHDPThetaStar(), 1);
		}
		// calculate the mixture: get the review assignment and normalize it
		Arrays.fill(probs, 0);
		// calculate the sum first
		for(_HDPThetaStar theta: user.getHDPTheta4Rvw()){
			sum += user.getHDPThetaMemSize(theta);
		}
		// calculate the prob for each dim
		for(_HDPThetaStar theta: user.getHDPTheta4Rvw()){
			cIndex = theta.getIndex();
			probs[cIndex] = user.getHDPThetaMemSize(theta)/sum;
		}
		user.setMixture(probs);
	}

	// variable used to record the sampling time for different edges
	// [0]: mmb_0; [1]: mmb_1; 
	long[] m_time = new long[3];
	// for all zero edges, we apply joint sampling
	protected void calculate_E_step_Edge_joint_all(){
//		calcProbNew();
		// sample z_{i->j}
		_MMBAdaptStruct ui, uj;
		int sampleSize = 0, eij = 0;
		Arrays.fill(m_time, 0);
		
		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// print out the process of sampling edges
				if (++sampleSize%100000==0) {
					System.out.print('.');
					if (sampleSize%50000000==0)
						System.out.println();
				}
				// eij=1
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 1){
					long start = System.currentTimeMillis();
					eij = 1;
					// remove the connection for B_gh, i->j \in g, j->i \in h.
					rmConnection(ui, uj, eij);
					// update membership from ui->uj, remove the edge
					updateEdgeMembership(i, j, eij);	
					// sample new cluster for the edge
					sampleEdge(i, j, eij);
					// update membership from uj->ui, remove the edge
					updateEdgeMembership(j, i, eij);
					// sample new clusters for the two edges
					sampleEdge(j, i, eij);
					// add the new connection for B_g'h', i->j \in g', j->i \in h'
					addConnection(ui, uj, eij);
					long end = System.currentTimeMillis();
					m_time[1] += (end - start);
				// eij = 0
				}else if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
					long start = System.currentTimeMillis();
					eij = 0;
					// remove the connection for B_gh, i->j \in g, j->i \in h.
					rmConnection(ui, uj, eij);
					// update membership from ui->uj, uj->ui
					updateEdgeMembership(i, j, eij);
					updateEdgeMembership(j, i, eij);
					sampleZeroEdgeJoint(i, j);
					long end = System.currentTimeMillis();
					m_time[0] += (end - start);
				} else{
					// remove the two edges from background model
					long start = System.currentTimeMillis();
					updateSampleSize(2, -2);
					sampleZeroEdgeJoint(i, j);
					long end = System.currentTimeMillis();
					m_time[2] += (end - start);
					
				}
			}
		}
		mmb_0.add((int) m_MNL[0]); mmb_1.add((int) m_MNL[1]);bk_0.add((int) m_MNL[2]);
		System.out.print(String.format("\n[Time]Sampling: mmb_0: %.3f secs, mmb_1: %.3f secs, bk_0: %.3f secs\n", (double)m_time[0]/1000, (double)m_time[1]/1000, (double)m_time[2]/1000));
		System.out.print(String.format("[Info]kBar: %d, background prob: %.5f, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, 1-m_rho, m_MNL[0], m_MNL[1],m_MNL[2]));
	}
	protected void calculate_E_step_Edge_joint_bk(){
//		calcProbNew();
		// sample z_{i->j}
		_MMBAdaptStruct ui, uj;
		int sampleSize = 0, eij = 0;
		double p_mmb_0 = 0, p_bk = 1-m_rho;

		Arrays.fill(m_time, 0);
		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// print out the process of sampling edges
				if (++sampleSize%100000==0) {
					System.out.print('.');
					if (sampleSize%50000000==0)
						System.out.println();
				}
				// eij=1
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 1){
					long start = System.currentTimeMillis();
					eij = 1;
					// remove the connection for B_gh, i->j \in g, j->i \in h.
					rmConnection(ui, uj, eij);
					// update membership from ui->uj, remove the edge
					updateEdgeMembership(i, j, eij);	
					// sample new cluster for the edge
					sampleEdge(i, j, eij);
					// update membership from uj->ui, remove the edge
					updateEdgeMembership(j, i, eij);
					// sample new clusters for the two edges
					sampleEdge(j, i, eij);
					// add the new connection for B_g'h', i->j \in g', j->i \in h'
					addConnection(ui, uj, eij);
					long end = System.currentTimeMillis();
					m_time[1] += (end - start);
				// eij = 0
				}else if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
					long start = System.currentTimeMillis();
					eij = 0;
					// use bernoulli distribution to decide whether it is background or mmb
					p_bk = 1-m_rho;
					p_mmb_0 = Math.exp(calcLogLikelihoodE(m_indicator[i][j], m_indicator[j][i], 0));
					m_bernoulli = new BinomialDistribution(1, p_mmb_0/(p_bk + p_mmb_0));
					// the edge belongs to bk
					if(m_bernoulli.sample() == 0){
						// remove the connection for B_gh, i->j \in g, j->i \in h.
						rmConnection(ui, uj, eij);
						// update membership from ui->uj, uj->ui
						updateEdgeMembership(i, j, eij);
						updateEdgeMembership(j, i, eij);
						updateSampleSize(2, 2);
					} else{
						rmConnection(ui, uj, eij);
						updateEdgeMembership(i, j, eij);	
						sampleEdge(i, j, eij);
						// update membership from uj->ui, remove the edge
						updateEdgeMembership(j, i, eij);
						sampleEdge(j, i, eij);
						addConnection(ui, uj, eij);
					}
					long end = System.currentTimeMillis();
					m_time[0] += (end - start);
				} else{
					long start = System.currentTimeMillis();
					// remove the two edges from background model
					updateSampleSize(2, -2);
					sampleZeroEdgeJoint(i, j);
					long end = System.currentTimeMillis();
					m_time[2] += (end - start);
				}
			}
		}
		mmb_0.add((int) m_MNL[0]); mmb_1.add((int) m_MNL[1]);bk_0.add((int) m_MNL[2]);
		System.out.print(String.format("\n[Time]Sampling: mmb_0: %.3f secs, mmb_1: %.3f secs, bk_0: %.3f secs\n", (double)m_time[0]/1000, (double)m_time[1]/1000, (double)m_time[2]/1000));
		System.out.print(String.format("[Info]kBar: %d, background prob: %.5f, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, 1-m_rho, m_MNL[0], m_MNL[1],m_MNL[2]));
	}

	private void checkClusters(){
		int index = 0;
		int zeroDoc = 0, zeroEdge = 0;
		while(m_hdpThetaStars[index] != null){
			if(index < m_kBar && m_hdpThetaStars[index].getTotalEdgeSize() == 0)
				zeroEdge++;
			if(index < m_kBar && m_hdpThetaStars[index].getMemSize() == 0)
				zeroDoc++;
			index++;
		}
		stat.get("onlyedges").add(zeroDoc);
		stat.get("onlydocs").add(zeroEdge);
		stat.get("mixture").add(m_kBar-zeroDoc-zeroEdge);
		
		System.out.print(String.format("[Info]Clusters with only edges: %d, Clusters with only docs: %d, kBar:%d, non_null hdp: %d\n", zeroDoc, zeroEdge, m_kBar, index));
	}
	
	// check if the sum(m_MNL) == sum(edges of all clusters)
	protected void checkEdges(){
		int mmb_0 = 0, mmb_1 = 0;
		_HDPThetaStar theta;
		for(int i=0; i<m_kBar; i++){
			theta = m_hdpThetaStars[i];
			mmb_0 += theta.getEdgeSize(0);
			mmb_1 += theta.getEdgeSize(1);
		}
		if(mmb_0 != m_MNL[0])
			System.out.println("Zero edges sampled from mmb is not correct!");
		if(mmb_1 != m_MNL[1])
			System.out.println("One edges sampled from mmb is not correct!");
	}
	
	protected void checkMMBEdges(){
		int mmb = 0;
		for(_AdaptStruct u: m_userList){
			_MMBAdaptStruct user = (_MMBAdaptStruct) u;
			for(_HDPThetaStar th: user.getHDPTheta4Edge()){
				mmb += user.getHDPThetaEdgeSize(th);
			}
		}
		if(mmb != m_MNL[0] + m_MNL[1])
			System.out.println("mmb edges is not correct!");
	}
	public int getKBar(){
		return m_kBar;
	}
	
	public int getUserSize(){
		return m_userList.size();
	}
	// Estimate the sparsity parameter.
	// \rho = (M+N+c-1)/(M+N+L+c+d-2)
	public double estRho(){
		m_rho = (m_MNL[0] + m_MNL[1] + m_abcd[2] - 1) / (m_MNL[0] + m_MNL[1] + m_MNL[2] + m_abcd[2] + m_abcd[3] - 2);
		return 0;
	}
	
	private void initStat(){
		stat.put("onlyedges", new ArrayList<Integer>());
		stat.put("onlydocs", new ArrayList<Integer>());
		stat.put("mixture", new ArrayList<Integer>());
	}
	
	public void initThetaStars_Edges_Joint(){
		calcProbNew();
		_MMBAdaptStruct ui, uj;
		int sampleSize = 0;
		// add the friends one by one.
		for(int i=0; i< m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				// print out the process of sampling edges
				if (++sampleSize%100000==0) {
					System.out.print('.');
					if (sampleSize%50000000==0)
						System.out.println();
				}
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// if ui and uj are friends, random sample clusters for the two connections
				// e_ij = 1, z_{i->j}, e_ji = 1, z_{j -> i} = 1
				if(ui.getUser().hasFriend(uj.getUserID())){
					// sample two edges between i and j
					randomSampleEdges(i, j, 1);
					// add the edge assignment to corresponding cluster
					// we have to add connections after we know the two edge assignment (the clusters for i->j and j->i)
					addConnection(ui, uj, 1);
					// update the sample size with the specified index and value
					// index 0 : e_ij = 0 from mmb; index 1 : e_ij = 1 from mmb; index 2 : 0 from background model
					updateSampleSize(1, 2);
				} else{
					sampleZeroEdgeJoint(i, j);
				}
			}
		}
		mmb_0.add((int) m_MNL[0]); mmb_1.add((int) m_MNL[1]);bk_0.add((int) m_MNL[2]);
		System.out.print(String.format("\n[Info]kBar: %d, background prob: %.5f, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, 1-m_rho, m_MNL[0], m_MNL[1],m_MNL[2]));
	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList)
			m_userList.add(new _MMBAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];			
		m_indicator = new _HDPThetaStar[m_userList.size()][m_userList.size()];
	}
	
	// MLE of B matrix
	public double[][] MLEB(){
		int e_0 = 0, e_1 = 0;
		double b = 0;
		double[][] B = new double[m_kBar][m_kBar];
		_HDPThetaStar theta_g, theta_h;
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=0; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				e_0 = theta_g.getConnectionEdgeCount(theta_h, 0);
				e_1 = theta_g.getConnectionEdgeCount(theta_h, 1);
				b = (e_1 + m_abcd[0] -1)/(e_0 + e_1 + m_abcd[0] + m_abcd[1] -2);
				B[g][h] = b;
				B[h][g] = b;
			}
		}
		return B;
	}

	// if ui and uj are friends, random sample clusters for the two connections
	// e_ij = 1, z_{i->j}, e_ji = 1, z_{j -> i} = 1
	protected void randomSampleEdges(int i, int j, int e){
		randomSampleEdge(i, j, e);
		randomSampleEdge(j, i, e);
	}
	
	/*** In order to avoid creating too many thetas, we randomly assign nodes to thetas at beginning.
	 *   and this sampling function is only used for initial states.***/
	private void randomSampleEdge(int i,int j, int e){
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);
		
		// Random sample one cluster.
		int k = (int) (Math.random() * m_kBar);
		
		// Step 3: update the setting after sampling z_ij
		// update the edge count for the cluster: first param means edge (0 or 1), the second one mean increase by 1.
		m_hdpThetaStars[k].updateEdgeCount(e, 1);
		// update the neighbor information to the neighbor hashmap 
		ui.addNeighbor(uj, m_hdpThetaStars[k], e);
		// update the user info with the newly sampled hdpThetaStar
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[k], 1, e);//-->3	
			
		// Step 5: Put the cluster info in the matrix for later use
		// Since we have all the info, we don't need to put the theta info in the _MMBNeighbor structure.
		m_indicator[i][j] = m_hdpThetaStars[k];
	}

	// remove the connection between ui and uj, where i->j \in g, j->i \in h.
	public void rmConnection(_MMBAdaptStruct ui, _MMBAdaptStruct uj, int e){
		_HDPThetaStar theta_g, theta_h;
		theta_g = ui.getOneNeighbor(uj).getHDPThetaStar();
		theta_h = uj.getOneNeighbor(ui).getHDPThetaStar();
		theta_g.rmConnection(theta_h, e);
		theta_h.rmConnection(theta_g, e);
	}

	// we assume all zero edges are from mmb first
	// then utilize bernoulli to sample edges from background model
	protected void sampleC(){
		_MMBAdaptStruct ui, uj;
		double p_mmb_0 = 0, p_bk = 1-m_rho;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// eij = 0 from mmb ( should be all zero edges)
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
					// bernoulli distribution to decide whether it is background or mmb
					p_mmb_0 = Math.exp(calcLogLikelihoodE(m_indicator[i][j], m_indicator[j][i], 0));
					m_bernoulli = new BinomialDistribution(1, p_mmb_0/(p_bk + p_mmb_0));
					// the edge belongs to bk
					if(m_bernoulli.sample() == 0){
						rmConnection(ui, uj, 0);
						updateEdgeMembership(i, j, 0);
						updateEdgeMembership(j, i, 0);
						updateSampleSize(2, 2);
					}
					// if the edge belongs to mmb, we just keep it
				}
			}
		}
		mmb_0.add((int) m_MNL[0]); mmb_1.add((int) m_MNL[1]);bk_0.add((int) m_MNL[2]);
		System.out.print(String.format("\n[Info]kBar: %d, background prob: %.5f, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, 1-m_rho, m_MNL[0], m_MNL[1],m_MNL[2]));
	}
	
	protected void sampleEdge(int i, int j, int e){
		int k = 0;
		_HDPThetaStar theta_s, theta_h = m_indicator[j][i];
		double likelihood, logNew, gamma_k, logSum = 0;
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);

		for(k=0; k<m_kBar; k++){			
			//log likelihood of the edge p(e_{ij}, z, B)
			// p(eij|z_{i->j}, z_{j->i}, B)*p(z_{i->j}|\pi_i)*p(z_{j->i|\pj_j})
			theta_s = m_hdpThetaStars[k];
			// we record all the 
			if(!theta_h.isValid())
				System.out.println("[Error]Invalid theta inside sampleEdge()!!");
		
			likelihood = calcLogLikelihoodEMarginal(theta_s, theta_h, e);
			if(Double.isInfinite(likelihood))
				System.out.println("Infinite!");
		
			//p(z=k|\gamma,\eta)
			gamma_k = m_hdpThetaStars[k].getGamma();
		
			likelihood += Math.log(calcGroupPopularity(ui, k, gamma_k));
		
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
					
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
		}
		// fix1: the probability for new cluster
		logNew = Math.log(m_eta*m_gamma_e) + m_pNew[e];
		logSum = Utils.logSum(logSum, logNew);
	
		//Sample group k with likelihood.
		k = sampleEdgeInLogSpace(logSum, e);
	
		if(k == -1){
			sampleNewCluster4Edge();// shall we consider the current edge?? posterior sampling??
			k = m_kBar - 1;
			// for getting stat
			System.out.print("em*");
			m_newCluster4Edge++;
		}
		// update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateEdgeCount(e, 1);//first 1 means edge 1, the second one mean increase by 1.
	
		m_MNL[e]++;
		// update the user info with the newly sampled hdpThetaStar.
		ui.addNeighbor(uj, m_hdpThetaStars[k], e);
	
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[k], 1, e);//-->3	
	
		// Put the reference to the matrix for later usage.
		// Since we have all the info, we don't need to put the theta info in the _MMBNeighbor structure.
		m_indicator[i][j] = m_hdpThetaStars[k];
	}
	
	//Sample hdpThetaStar with likelihood.
	protected int sampleEdgeInLogSpace(double logSum, int e){
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
			
		int k = -1;
		// [fixed bug], the prob for new cluster should consider the gamma too.
		double newLogSum = Math.log(m_eta*m_gamma_e) + m_pNew[e];
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_hdpThetaStars[k].getProportion());
		} while (k<m_kBar);
			
		if (k==m_kBar)
			k--; // we might hit the very last
		return k;
	}
	

	//Sample hdpThetaStar with likelihood.
	 protected int sampleIn2DimArrayLogSpace(double logSum, double back_prob, double[][] cacheB){
	 
	 	double rnd = FloatUniform.staticNextFloat();
	 	logSum += Math.log(rnd);//we might need a better random number generator
	 		
	 	int k = -1;
	 	// we start from the background model.
	 	double newLogSum = back_prob;
	 	do {
	 		if (newLogSum>=logSum)
	 			break;
	 		k++;
	 		if (k==(m_kBar+1)*(m_kBar+1)){
	 			k--; // we might hit the very last
	 			return k;
	 		}
	 		newLogSum = Utils.logSum(newLogSum, cacheB[k/(m_kBar+1)][k%(m_kBar+1)]);
	 			
	 	} while (k<(m_kBar+1)*(m_kBar+1));
	 	return k;
	 }

	// sample eij = 0 from the joint probabilities of cij, zij and zji.
 	public void sampleZeroEdgeJoint(int i, int j){
 		/**we will consider all possible combinations of different memberships.
 		 * 1.cij=0, cji=0, prob: (1-\rho), 1 case
 		 * 2.cij=1, cji=1, known (Bgh, Bhg), prob: \rho(1-Bgh), k(k+1)/2 possible cases
 		 * posterior prob: \rho*(b+e_0)/(a+b+e_0+e_1)
 		 * 3.cij=1, cji=1, unknows (Bgh, Bhg), prob: \rho*b/(a+b), k+1 possible cases 
 		 * In total, we have (k+1)*(k+2)/2+1 possible cases. **/
 		// Step 1: calc prob for different cases of cij, cji.
 		// case 0: background model while the prob is not stored in the two-dim array.
 		double logSum = Math.log(1-m_rho);
 		/**We maintain a matrix for storing probability. As the matrix is 
 		 * symmetric, we only calculate upper-triangle. **/
 		double[][] cacheB = new double[m_kBar+1][m_kBar+1];
 		for(double[] b: cacheB)
 			Arrays.fill(b, Double.NEGATIVE_INFINITY);
 		
 		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
 		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);
 
 		_HDPThetaStar theta_g, theta_h;
 		// case 1: existing thetas.
 		for(int g=0; g<m_kBar; g++){
 			theta_g = m_hdpThetaStars[g];
 			for(int h=g; h<m_kBar; h++){
 				theta_h = m_hdpThetaStars[h];
 				cacheB[g][h] = calcLogLikelihoodE(theta_g, theta_h, 0);
 				cacheB[g][h] += Math.log(theta_g.getGamma()) + Math.log(theta_h.getGamma());
 				if(g == h){
 					logSum = Utils.logSum(logSum, cacheB[g][h]);
 				} else{
 					cacheB[h][g] = cacheB[g][h];
 					// we need to add twice of logp.
 					logSum = Utils.logSum(logSum, cacheB[h][g]+m_log2);
 				}
 			}
 		}
 		// case 2: either one is from new cluster.
 		// pre-calculate \rho*(b/(a+b))*\gamma_e
 		double pNew = Math.log(m_rho) + Math.log(m_abcd[1]) - Math.log(m_abcd[0] + m_abcd[1]);
 		double gamma_g = 0;
 		for(int k=0; k<m_kBar; k++){
 			gamma_g = m_hdpThetaStars[k].getGamma();
 			// if either one is 0, then prob is 0 -> log prob = -Infinity
 			if(m_gamma_e != 0 && gamma_g != 0){
 	 			cacheB[k][m_kBar] = pNew + Math.log(m_gamma_e) + Math.log(gamma_g);
 	 			cacheB[m_kBar][k] = cacheB[k][m_kBar];
 	 			logSum = Utils.logSum(logSum, cacheB[k][m_kBar]+m_log2);
 			}
  		}
 		// both are from new clusters.
 		if(m_gamma_e != 0){
 	 		cacheB[m_kBar][m_kBar] = pNew + Math.log(m_gamma_e) + m_log2;
 	 		logSum = Utils.logSum(logSum, cacheB[m_kBar][m_kBar]);
 		} 	
 		
  		// Step 2: sample one pair from the prob matrix.
 		int k = sampleIn2DimArrayLogSpace(logSum, Math.log(1-m_rho), cacheB);
  		
  		// Step 3: Analyze the sampled cluster results.
  		// case 1: k == -1, sample from the background model;
 		// case 2: k!= 1, sample from mmb model.
 		int g = 0, h = 0;
 		if(k != -1){
 			g = k / (m_kBar+1);
 			h = k % (m_kBar+1);
 			if(g == m_kBar || h == m_kBar){
 				// we need to sample the new cluster
 				sampleNewCluster4Edge();// shall we consider the current edge?? posterior sampling??
 				// for getting stat
 				System.out.print("ej*");
 				m_newCluster4EdgeJoint++;
 			}
 			// Update the thetaStar and user info after getting z_ij.
 			m_hdpThetaStars[g].updateEdgeCount(0, 1);//-->1
 			ui.addNeighbor(uj, m_hdpThetaStars[g], 0);
 			ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[g], 1, 0);	
 			m_indicator[i][j] = m_hdpThetaStars[g];
 			updateSampleSize(0, 1);
 			
 			// Update the thetaStar and user info after getting z_ji.
 			m_hdpThetaStars[h].updateEdgeCount(0, 1);
 			uj.addNeighbor(ui, m_hdpThetaStars[h], 0);
 			uj.incHDPThetaStarEdgeSize(m_hdpThetaStars[h], 1, 0);
  			m_indicator[j][i] = m_hdpThetaStars[h];
  			updateSampleSize(0, 1);
  			addConnection(ui, uj, 0);
  		} else{
  			updateSampleSize(2, 2);
  		}
  	}
	
	// Sample new cluster based on sampling of z_{i->j}, thus, the cluster will have edges info.
	public void sampleNewCluster4Edge(){
		// use the first available one as the new cluster.	
		if (m_hdpThetaStars[m_kBar] == null){
			if (this instanceof CLinAdaptWithMMB)// this should include all the inherited classes for adaptation based models
				m_hdpThetaStars[m_kBar] = new _HDPThetaStar(2*m_dim);
			else
				m_hdpThetaStars[m_kBar] = new _HDPThetaStar(m_dim);
		}
		
		m_hdpThetaStars[m_kBar].enable();
		m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
		m_hdpThetaStars[m_kBar].initLMStat(m_lmDim);
		m_hdpThetaStars[m_kBar].setPerfStat(m_classNo);
		
		double rnd = Beta.staticNextDouble(1, m_alpha);
		m_hdpThetaStars[m_kBar].setGamma(rnd*m_gamma_e);
		m_gamma_e = (1-rnd)*m_gamma_e;
		
		m_kBar++;
	}
	
	@Override
	public void sampleThetaStars(){
		double gamma_e = m_gamma_e/m_M;
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_hdpThetaStars[m] == null){
				if (this instanceof CLinAdaptWithMMB)// this should include all the inherited classes for adaptation based models
					m_hdpThetaStars[m] = new _HDPThetaStar(2*m_dim, gamma_e);
				else
					m_hdpThetaStars[m] = new _HDPThetaStar(m_dim, gamma_e);
			} else
				m_hdpThetaStars[m].setGamma(gamma_e);//to unify the later operations
			
			//sample \phi from Normal distribution.
			m_G0.sampling(m_hdpThetaStars[m].getModel());//getModel-> get \phi.
		}
	}
	
	//Sample the global mixture proportion, \gamma~Dir(m1, m2,..,\alpha)
	@Override
	protected void sampleGamma(){

		for(int k=0; k<m_kBar; k++)
			m_hdpThetaStars[k].m_hSize = 0;
		
		_MMBAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_MMBAdaptStruct) m_userList.get(i);	
			if(user.getAdaptationSize() == 0)
				continue;
			// collect the thetas for docs and edges
			Set<_HDPThetaStar> thetas = new HashSet<_HDPThetaStar>();
			thetas.addAll(user.getHDPTheta4Rvw());
			thetas.addAll(user.getHDPTheta4Edge());
			for(_HDPThetaStar s: thetas){
				s.m_hSize += sampleH(user, s);
			}
		}		
		
		m_cache[m_kBar] = Gamma.staticNextDouble(m_alpha, 1);//for gamma_e
		
		double sum = m_cache[m_kBar];
		for(int k=0; k<m_kBar; k++){
			m_cache[k] = Gamma.staticNextDouble(m_hdpThetaStars[k].m_hSize+m_alpha, 1);
			sum += m_cache[k];
		}
		
		for(int k=0; k<m_kBar; k++) 
			m_hdpThetaStars[k].setGamma(m_cache[k]/sum);
		
		m_gamma_e = m_cache[m_kBar]/sum;//\gamma_e.
	}
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_MMBAdaptStruct user, _HDPThetaStar s){
			int n = user.getHDPThetaMemSize(s);
			n += user.getHDPThetaEdgeSize(s);
			if(n==1)
				return 1;//s(1,1)=1		

			double etaGammak = Math.log(m_eta) + Math.log(s.getGamma());
			//the number of local groups lies in the range [1, n];
			for(int h=1; h<=n; h++){
				double logStir = logStirling(n, h);
				m_cache[h-1] = h*etaGammak + logStir;
			}
			
			//h starts from 0, we want the number of tables here.	
			return Utils.sampleInLogArray(m_cache, n) + 1;
		}
	
	// Save the language models of thetaStars
	public void saveClusterLanguageModels(String model){
		PrintWriter writer;
		String filename;
		File dir = new File(model);
		_HDPThetaStar theta;
		double[] lm;
		try{
			if(!dir.exists())
				dir.mkdirs();
			for(int i=0; i<m_kBar; i++){
				theta = m_hdpThetaStars[i];
				filename = String.format("%s/%d.lm", model, theta.getIndex());
				writer = new PrintWriter(new File(filename));
				lm = theta.getLMStat();
				for(int v=0; v<lm.length; v++){
					if(v == lm.length-1)
						writer.write(Double.toString(lm[v]));
					else
						writer.write(lm[v]+",");
				}
				writer.close();
			}
		} catch (IOException e){
			e.printStackTrace();
		}
	}

	public void setJointSampling(boolean b){
		m_jointAll = b;
	}
	// Set the sparsity parameter
	public void setRho(double v){
		m_rho = v;
	}

	@Override
	public String toString() {
		return String.format("CLRWithMMB[dim:%d,lmDim:%d,M:%d,rho:%.5f,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]\n", m_dim,m_lmDim,m_M, m_rho, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}

	protected void sanityCheck(){
		checkClusters();
		checkEdges();
		checkMMBEdges();
	}
	
	@Override
	public void printInfo(){
		MyPriorityQueue<_RankItem> clusterRanker = new MyPriorityQueue<_RankItem>(m_kBar);		
		
		//clear the statistics
		for(int i=0; i<m_kBar; i++) {
			m_hdpThetaStars[i].resetCount();
			clusterRanker.add(new _RankItem(i, m_hdpThetaStars[i].getMemSize()));//get the most popular clusters
		}

		//collect statistics across users in adaptation data
		for(int i=0; i<m_userList.size(); i++) {
			_MMBAdaptStruct user = (_MMBAdaptStruct)m_userList.get(i);
			
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				else{
					_HDPThetaStar theta = r.getHDPThetaStar();
					if(r.getYLabel() == 1) 
						theta.incPosCount(); 
					else 
						theta.incNegCount();
				}
			}
		}
		System.out.println("[Info]Clusters:");
		for(_RankItem it: clusterRanker){
			_HDPThetaStar theta = m_hdpThetaStars[it.m_index];
			double edgeSize = theta.getEdgeSize(0) + theta.getEdgeSize(1);
			double ratio = edgeSize == 0 ? 0: edgeSize/(theta.getPosCount()+theta.getNegCount());
			System.out.format("%s-(e_0:%d,e_1:%d,e/r:%.4f)-(pos_f1:%.4f,neg_f1:%.4f)\n", theta.showStat(), 
			theta.getEdgeSize(0), theta.getEdgeSize(1), ratio, theta.getPerfStat().getF1(1), theta.getPerfStat().getF1(0));	
		}
		System.out.println();
	}
	// go over all the reviews and calculate the performance of each cluster
	public void evaluateClusterPerformance(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
			
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					_AdaptStruct user;
					_HDPThetaStar theta;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data
								continue;
									
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									r.setPredictLabel(predL);
									theta = r.getHDPThetaStar();
									theta.getPerfStat().addOnePredResult(predL, trueL);
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
			
		// calculate the F1 for each cluster
		for(int k=0; k<m_kBar; k++){
			m_hdpThetaStars[k].getPerfStat().calculatePRF();	
		}
	}
	
	public void printClusterInfo(long start, long end){
		System.out.format("[Cluster]Sampling (docs/edges generaly/edges jointly) generates (%d, %d, %d) new clusters.\n", m_newCluster4Doc, m_newCluster4Edge, m_newCluster4EdgeJoint);
		System.out.println("[Time]The sampling iteration took " + (end-start)/1000 + " secs.");
	}
	
	// In the training process, we sample documents first, then sample edges.
	@Override
	public double train(){
		
		System.out.print(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		double likelihoodX = 0, likelihoodY = 0, likelihoodE = 0;
		int count = 0;
		
		/**We want to sample documents first without knowing edges,
		 * So we have to rewrite the init function to split init thetastar for docs and edges.**/
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
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			
			// E step
			int stepCount = 0;
			long start = System.currentTimeMillis();
			// multiple E steps of sampling docs
			while(stepCount++ < m_multipleE){
				calculate_E_step();
			}
			calculate_E_step_Edge();
			long end = System.currentTimeMillis();
			printClusterInfo(start, end);
			
			// M step
			likelihoodY = calculate_M_step();
			
			likelihoodX = accumulateLikelihoodX();
			likelihoodE = accumulateLikelihoodEMMB();
			likelihoodE += (m_MNL[2]/2)*Math.log(1-m_rho);
			curLikelihood = likelihoodY + likelihoodX + likelihoodE;
			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0){
				evaluateModel();
				evaluateClusterPerformance();
				printInfo();
				// clear the performance data for each cluster
				for(int k=0; k<m_kBar; k++){
					m_hdpThetaStars[k].getPerfStat().clear();
				}
			}

			System.out.print(String.format("[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}
		
		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}

	@Override
	public double trainTrace(String data, long time){
			
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
				int stepCount = 0;
				long start = System.currentTimeMillis();
				// multiple E steps of sampling docs
				while(stepCount++ < m_multipleE){
					calculate_E_step();
				}
				calculate_E_step_Edge();
				long end = System.currentTimeMillis();
				printClusterInfo(start, end);
				
				// M step
				likelihoodY = calculate_M_step();

				// accumulate the likelihood
				likelihoodX = accumulateLikelihoodX();
//				likelihoodE = accumulateDecomposedLikelihoodEMMB();
//				likelihoodE[3] = (m_MNL[2]/2)*Math.log(1-m_rho);
				
				likelihoodE = accumulateLikelihoodEMMB();
				likelihoodE += (m_MNL[2]/2)*Math.log(1-m_rho);
				
//				curLikelihood = likelihoodY + likelihoodX + likelihoodE[0] + likelihoodE[1] + likelihoodE[3];
				curLikelihood = likelihoodY + likelihoodX + likelihoodE;
				delta = (lastLikelihood - curLikelihood)/curLikelihood;
				
				// evaluate the model
				if (i%m_thinning==0){
					evaluateModel();
					test();
					for(_AdaptStruct u: m_userList)
						u.getPerfStat().clear();
				}
//				writer.write(String.format("%.5f\t%.5f\t%.5f\t%.5f\t%d\t%.5f\t%.5f\n", likelihoodE[0], likelihoodE[1], likelihoodE[2], likelihoodE[3], m_kBar, m_perf[0], m_perf[1]));
				writer.write(String.format("%.5f\t%.5f\t%.5f\t%.5f\t%d\t%.5f\t%.5f\n", likelihoodY, likelihoodX, likelihoodE, delta, m_kBar, m_perf[0], m_perf[1]));
				System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
				if(Math.abs(delta) < m_converge)
					break;
				lastLikelihood = curLikelihood;
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}
	
	protected void updateSampleSize(int index, int val){
		if(index <0 || index > m_MNL.length)
			System.err.println("[Error]Wrong index!");
		m_MNL[index] += val;
		if (Utils.sumOfArray(m_MNL) % 1000000==0) {
			System.out.print('.');
			if (Utils.sumOfArray(m_MNL) % 50000000==0)
				System.out.println();
		}
	}
	@Override
	// Override this function since we have different conditions for removing clusters.
	public void updateDocMembership(_HDPAdaptStruct user, _Review r){
		int index = -1;
		_HDPThetaStar curThetaStar = r.getHDPThetaStar();

		// remove the current review from the user side.
		user.incHDPThetaStarMemSize(r.getHDPThetaStar(), -1);
				
		// remove the current review from the theta side.
		// remove the lm stat first before decrease the document count
		curThetaStar.rmLMStat(r.getLMSparse());
		curThetaStar.updateMemCount(-1);
		
		// No data associated with the cluster
		if(curThetaStar.getMemSize() == 0 && curThetaStar.getTotalEdgeSize() == 0) {
			System.out.println("[Debug]Zero cluster detected in updating doc!");
			// check if every dim gets 0 count in language mode
			LMStatSanityCheck(curThetaStar);
			
			// recycle the gamma
			m_gamma_e += curThetaStar.getGamma();
//			curThetaStar.resetGamma();	
			
			// swap the disabled theta to the last for later use
			index = findHDPThetaStar(curThetaStar);
			swapTheta(m_kBar-1, index); // move it back to \theta*
			
			curThetaStar.reset();
			m_kBar --;
		}
	}
	
	public void updateEdgeMembership(int i, int j, int e){
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);
		
		int index = -1;
		_HDPThetaStar thetai = ui.getThetaStar(uj);
		
		// remove the neighbor from user
		ui.rmNeighbor(uj);
		
		// update the edge information inside the user
		ui.incHDPThetaStarEdgeSize(thetai, -1, e);
		
		// update the edge count for the thetastar
		thetai.updateEdgeCount(e, -1);
		
		m_MNL[e]--;
		// No data associated with the cluster
		if(thetai.getMemSize() == 0 && thetai.getTotalEdgeSize() == 0){		
			System.out.println("[Info]Zero cluster detected in updating edge!");
			// recycle the gamma
			m_gamma_e += thetai.getGamma();
			
			// swap the disabled theta to the last for later use
			index = findHDPThetaStar(thetai);
			if(index == -1)
				System.out.println("Bug");
			swapTheta(m_kBar-1, index); // move it back to \theta*
			
			thetai.reset();
			m_kBar --;
		}
	}
	
	public void printBMatrix(String filename){
		// Get the B matrix
		int idx = filename.indexOf("txt");
		String zerofile = filename.substring(0, idx-1)+"_0.txt";
		String onefile = filename.substring(0, idx-1)+"_1.txt";

		int[] eij;
		int[][][] B = new int[m_kBar][m_kBar][2];
		_HDPThetaStar theta1;
		int index1 = 0, index2 = 0;
		for(int i=0; i<m_kBar; i++){
			theta1 = m_hdpThetaStars[i];
			index1 = theta1.getIndex();
			HashMap<_HDPThetaStar, _Connection> connectionMap = theta1.getConnectionMap();
			for(_HDPThetaStar theta2: connectionMap.keySet()){
				index2 = theta2.getIndex();
				eij = connectionMap.get(theta2).getEdge();
				B[index1][index2][0] = eij[0];
				B[index1][index2][1] = eij[1];

			}
		}
		try{
			// print out the zero edges in B matrix
			PrintWriter writer = new PrintWriter(new File(zerofile), "UTF-8");
			for(int i=0; i<B.length; i++){
				int[][] row = B[i];
				for(int j=0; j<row.length; j++){
					writer.write(String.format("%d", B[i][j][0]));
					if(j != row.length - 1){
						writer.write("\t");
					}
				}
				writer.write("\n");
			}
			writer.close();
			// print out the one edges in B matrix
			writer = new PrintWriter(new File(onefile), "UTF-8");
			for(int i=0; i<B.length; i++){
				int[][] row = B[i];
				for(int j=0; j<row.length; j++){
					writer.write(String.format("%d", B[i][j][1]));
					if(j != row.length - 1){
						writer.write("\t");
					}
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printClusterInfo(String filename){
		try {
			_HDPThetaStar theta;
			PrintWriter writer = new PrintWriter(new File(filename));
			for(int k=0; k<m_kBar; k++){
				theta = m_hdpThetaStars[k];
				writer.write(String.format("%d,%d,%d\n", theta.getMemSize(), theta.getEdgeSize(0), theta.getEdgeSize(1)));
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void printEdgeAssignment(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			_MMBAdaptStruct u1;
			int eij = 0, index1 = 0, index2 = 0;
			HashMap<_HDPAdaptStruct, _MMBNeighbor> neighbors;
			for(int i=0; i<m_userList.size(); i++){
				u1 = (_MMBAdaptStruct) m_userList.get(i);
				neighbors = u1.getNeighbors();
				for(_HDPAdaptStruct nei: neighbors.keySet()){
					eij = u1.getNeighbors().get(nei).getEdge();
					if(eij == 1){
						index1 = neighbors.get(nei).getHDPThetaStarIndex();
						index2 = ((_MMBAdaptStruct) nei).getNeighbors().get(u1).getHDPThetaStarIndex();
						writer.write(String.format("%s,%s,%d,%d\n", u1.getUserID(), nei.getUserID(), index1, index2));
					}
				}
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printStat(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(String key: stat.keySet()){
				writer.write(key+"\n");
				for(int v: stat.get(key)){
					writer.write(v+",");
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printEdgeCount(String filename){
		try{
			PrintWriter writer = new PrintWriter(filename);
			for(int v: mmb_0)
				writer.write(v+"\t");
			writer.write("\n");
			for(int v: mmb_1)
				writer.write(v+"\t");
			writer.write("\n");
			for(int v: bk_0)
				writer.write(v+"\t");
			writer.write("\n");
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
}
