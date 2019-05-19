package Classifier.semisupervised;

import java.io.IOException;
import java.util.Arrays;

import structures._Corpus;
import structures._Doc;
import structures._Edge;
import structures._Node;
import utils.Utils;

public class GaussianFieldsByRandomWalk extends GaussianFields {
	double m_difference; //The difference between the previous labels and current labels.
	double m_eta; //The parameter used in random walk. 
	double[] m_pred_last; // result from last round of random walk
	
	double m_delta; // convergence criterion for random walk
	boolean m_weightedAvg; // random walk strategy: True - weighted average; False - majority vote
	boolean m_simFlag; //This flag is used to determine whether we'll consider similarity as weight or not.
	
	//Default constructor without any default parameters.
	public GaussianFieldsByRandomWalk(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		
		m_eta = 0.1;
		m_labelRatio = 0.1;
		m_delta = 1e-5;
		m_weightedAvg = true;
		m_simFlag = false;
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByRandomWalk(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean weightedAvg){
		super(c, classifier, C, ratio, k, kPrime);
		
		m_alpha = alhpa;
		m_beta = beta;
		m_delta = delta;
		m_eta = eta;
		m_weightedAvg = weightedAvg;
		m_simFlag = false;
	}
	
	@Override
	public String toString() {
		if (m_weightedAvg)
			return String.format("Gaussian Fields by random walk [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f]", 
					m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta);
		else
			return String.format("Random walk by majority vote[C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f, simWeight:%s]", 
					m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta, m_simFlag);
	}
	
	public void setSimilarity(boolean simFlag){
		m_simFlag = simFlag;
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the average of all neighbors as the new label until they converge.
	double randomWalkByWeightedSum(){//construct the sparse graph on the fly every time
		double wL = m_alpha, wU = m_beta, acc = 0;
		_Node node;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			node = m_nodeList[i];
			
			double wijSumU = 0, wijSumL = 0;
			double fSumU = 0, fSumL = 0;
			
			/****Walk through the top k' unlabeled neighbor for the current data.****/
			for (_Edge edge:node.m_unlabeledEdges) {				
				wijSumU += m_simFlag?edge.getSimilarity():1; //get the similarity between two nodes.
				fSumU += m_simFlag?edge.getSimilarity() * edge.getPred(): edge.getPred();
			}
			
			/****Walk through the top k labeled neighbor for the current data.****/
			for (_Edge edge:node.m_labeledEdges) {
				wijSumL += m_simFlag?edge.getSimilarity():1; //get the similarity between two nodes.
				fSumL += m_simFlag?edge.getSimilarity() * edge.getPred(): edge.getPred();
			}
			
			node.m_pred = m_eta * (fSumL*wL + fSumU*wU) / (wijSumL*wL + wijSumU*wU) + (1-m_eta) * node.m_classifierPred;
			if (Double.isNaN(node.m_pred)) {
				System.out.format("Encounter NaN in random walk!\nfSumL: %.3f, fSumU: %.3f, wijSumL: %.3f, wijSumU: %.3f\n", fSumL, fSumU, wijSumL, wijSumU);
				System.exit(-1);				
			} else if ((int)node.m_label == getLabel(node.m_pred))
				acc ++;
		}
		
		return acc / m_U;
	}
	
	//Take the majority of all neighbors(k+k') as the new label until they converge.
	double randomWalkByMajorityVote(){//construct the sparse graph on the fly every time
		double similarity = 0, acc = 0;
		int label;
		double wL = m_eta*m_alpha, wU = m_eta*m_beta;
		_Node node;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			node = m_nodeList[i];
			Arrays.fill(m_cProbs, 0);
			
			/****Walk through the top k' unlabeled neighbor for the current data.****/
			for (_Edge edge:node.m_unlabeledEdges) {
				label = getLabel(edge.getPred()); //Item n's label.
				similarity = edge.getSimilarity();
				
				m_cProbs[label] += m_simFlag?similarity*wU:wU; 
			}
			
			/****Walk through the top k labeled neighbor for the current data.****/
			for (_Edge edge:node.m_labeledEdges) {
				label = (int)edge.getLabel();
				similarity = edge.getSimilarity();
				
				m_cProbs[label] += m_simFlag?similarity*wL:wL; 
			}
			
			/****Multiple learner's prediction.****/
			label = (int) node.m_classifierPred;
			m_cProbs[label] += 1-m_eta; 
			
			node.m_pred = Utils.argmax(m_cProbs);
			
			if (node.m_label == node.m_pred)
				acc ++;
		}
		
		return acc / m_U;
	} 
	
	double updateFu() {
		m_difference = 0;
		for(int i = 0; i < m_U; i++){
			m_difference += Math.abs(m_nodeList[i].m_pred - m_pred_last[i]);
			m_pred_last[i] = m_nodeList[i].m_pred;//record the last result
		}
		return m_difference/m_U;
	}
	
//	//The test for random walk algorithm.
//	public double test(){
//		/***Construct the nearest neighbor graph****/
//		constructGraph(false);
//		
//		if (m_pred_last==null || m_pred_last.length<m_U)
//			m_pred_last = new double[m_U]; //otherwise we can reuse the current memory
//		
//		//record the starting point
//		for(int i=0; i<m_U; i++)
//			m_pred_last[i] = m_nodeList[i].m_pred;//random walk starts from multiple learner
//		
//		/***use random walk to solve matrix inverse***/
//		System.out.println("Random walk starts:");
//		int iter = 0;
//		double diff = 0, accuracy;
//		do {
//			if (m_weightedAvg)
//				accuracy = randomWalkByWeightedSum();	
//			else
//				accuracy = randomWalkByMajorityVote();
//			
//			diff = updateFu();
//			System.out.format("Iteration %d, converge to %.3f with accuracy %.4f...\n", ++iter, diff, accuracy);
//		} while(diff > m_delta && iter<50);//maximum 50 iterations 
//		
//		/***check the purity of newly constructed neighborhood graph after random walk with ground-truth labels***/
//		SimilarityCheck();
//		checkSimilarityVariance();
//		/***get some statistics***/
//		for(int i = 0; i < m_U; i++){
//			for(int j=0; j<m_classNo; j++)
//				m_pYSum[j] += Math.exp(-Math.abs(j-m_nodeList[i].m_pred));			
//		}
//		
//		/***evaluate the performance***/
//		double acc = 0;
//		int pred, ans;
//		for(int i = 0; i < m_U; i++) {
//			//pred = getLabel(m_fu[i]);
//			pred = getLabel(m_nodeList[i].m_pred);
//			ans = m_testSet.get(i).getYLabel();
//			m_TPTable[pred][ans] += 1;
//			
//			if (pred != ans) {
//				if (m_debugOutput!=null)
//					debug(m_testSet.get(i));
//			} else {
//				if (m_debugOutput!=null && Math.random()<0.02)
//					debug(m_testSet.get(i));
//				acc ++;
//			}
//		}
//		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
//		
//		return acc/m_U;
//	}
	
	@Override
	protected void debug(_Doc d) {
		debugSummary(d);
		if (m_corpus.hasContent())
			debugDetails(d);
	}
	
	void debugDetails(_Doc d){
		int id = d.getID();
		_Node node = m_nodeList[id];
		
		try {
			m_debugWriter.write(d.toString() + "\n");
			
			/****Get the top 5 elements from labeled neighbors******/
			for(int k=0; k<5; k++){
				_Edge item = node.m_labeledEdges.get(k);
				_Doc dj = getLabeledDoc(item.getNodeId());
				m_debugWriter.write(String.format("L(%d, %.4f)\t%s\n", (int)item.getClassifierPred(), item.getSimilarity(), dj.toString()));
			}
			
			/****Get the top 5 elements from k'UU******/
			for(int k=0; k<5; k++){
				_Edge item = node.m_unlabeledEdges.get(k);
				_Doc dj = getTestDoc(item.getNodeId());
				m_debugWriter.write(String.format("U(%d, %.3f, %.4f)\t%s\n", (int)item.getClassifierPred(), item.getPred(), item.getSimilarity(), dj.toString()));
			}
			m_debugWriter.write("\n");		
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	void debugSummary(_Doc d){
		int id = d.getID();
		_Node node = m_nodeList[id];
		double sim, wijSumU=0, wijSumL=0;
		double fSumU = 0, fSumL = 0;
		
		try {
			m_debugWriter.write(String.format("%d\t%.4f(%d*,%d)\t%d\n", 
					d.getYLabel(), //ground-truth
					node.m_pred, //random walk's raw prediction
					getLabel(node.m_pred), //map to discrete label
					getLabel3(node.m_pred), //consider the prior
					(int)node.m_classifierPred)); //multiple learner's prediction
		
			double mean = 0, sd = 0;
			//find top five labeled
			/****Walk through the top k labeled data for the current data.****/
			for (_Edge edge:node.m_labeledEdges) {
				wijSumL += edge.getSimilarity(); //get the similarity between two nodes.
				fSumL += edge.getSimilarity() * edge.getLabel();
				
				sd += edge.getSimilarity() * edge.getSimilarity();
			}
			
			mean = wijSumL / m_k;
			sd = Math.sqrt(sd/m_k - mean*mean);
			
			/****Get the top 10 elements from labeled neighbors******/
			for(int k=0; k<10; k++){
				_Edge item = node.m_labeledEdges.get(k);
				sim = item.getSimilarity()/wijSumL;
				
				if (k==0)
					m_debugWriter.write(String.format("L(%.2f)\t[%d:%.4f, ", fSumL/wijSumL, (int)item.getLabel(), sim));
				else if (k==9)
					m_debugWriter.write(String.format("%d:%.4f]\t%.3f\t%.3f\n", (int)item.getLabel(), sim, mean, sd));
				else
					m_debugWriter.write(String.format("%d:%.4f, ", (int)item.getLabel(), sim));
			}
			sd = 0;
			
			//find top five unlabeled
			/****Construct the top k' unlabeled data for the current data.****/
			for (_Edge edge:node.m_unlabeledEdges) {
				wijSumU += edge.getSimilarity(); //get the similarity between two nodes.
				fSumU += edge.getSimilarity() * edge.getPred();
				
				sd += edge.getSimilarity() * edge.getSimilarity();
			}
			
			mean = wijSumU / m_kPrime;
			sd = Math.sqrt(sd/m_kPrime - mean*mean);
			
			/****Get the top 10 elements from k'UU******/
			for(int k=0; k<10; k++){
				_Edge item = node.m_unlabeledEdges.get(k);
				sim = item.getSimilarity()/wijSumU;
				
				if (k==0)
					m_debugWriter.write(String.format("U(%.2f)\t[%.2f:%.4f, ", fSumU/wijSumU, item.getPred(), sim));
				else if (k==9)
					m_debugWriter.write(String.format("%.2f:%.4f]\t%.3f\t%.3f\n", item.getPred(), sim, mean, sd));
				else
					m_debugWriter.write(String.format("%.2f:%.4f, ", item.getPred(), sim));
			}
			m_debugWriter.write("\n");		
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 
	
	/******Added by Lin*****/
	int m_foldCount = 0;
	protected int m_kFold;
	double[][] m_simiStat;

	public void setKFold(int k){
		m_kFold = k;
		m_simiStat = new double[m_kFold][4];
	}
	
	//Check the mean and variance of the 
	public void checkSimilarityVariance(){
		double[] stat;
		double[] avgStat = new double[4];
		_Node node;
		for(int i=0; i<m_U; i++){
			node = m_nodeList[i];
			stat = calcMeanVar4OneNode(node);
			avgStat = addOneNodeStat(avgStat, stat);
		}
		
		for(int i=0; i<avgStat.length; i++){
			avgStat[i] = avgStat[i] / m_U;
		}
		m_simiStat[m_foldCount++] = avgStat;
	}
	
	public double[] addOneNodeStat(double[] avgStat, double[] one){
		if(avgStat.length == one.length){
			for(int i=0; i<avgStat.length; i++){
				avgStat[i] += one[i];
			}
		}
		return avgStat;
	}
	
	public double[] calcMeanVar4OneNode(_Node n){
		double[] stat4OneNode = new double[4];
		double sum = 0, mean = 0, var = 0;
		_Edge neighbor;
		for(int j=0; j<m_kPrime; j++){
			neighbor = n.m_unlabeledEdges.get(j);
			sum += neighbor.getSimilarity();
		}
		mean = sum / m_kPrime;
		stat4OneNode[0] = mean;//mean of unlabeled neighbors' similarities.
		for(int j=0; j<m_kPrime; j++){
			neighbor = n.m_unlabeledEdges.get(j);
			var += (neighbor.getSimilarity() - mean)*(neighbor.getSimilarity() - mean);
		}
		var = Math.sqrt(var/m_kPrime);
		stat4OneNode[1] = var;//variance of unlabeled neighbors' similarities.
	
		sum = 0; mean = 0; var = 0;//clear for the calculation of the labeled neighbors.
		for(int j=0; j<m_k; j++){
			neighbor = n.m_labeledEdges.get(j);
			sum += neighbor.getSimilarity();
		}
		mean = sum / m_k;
		stat4OneNode[2] = mean;//mean of unlabeled neighbors' similarities.
		for(int j=0; j<m_k; j++){
			neighbor = n.m_labeledEdges.get(j);
			var += (neighbor.getSimilarity() - mean)*(neighbor.getSimilarity() - mean);
		}
		var = Math.sqrt(var/m_k);
		stat4OneNode[3] = var;//variance of unlabeled neighbors' similarities.
		return stat4OneNode;
	}
	
	public void printSimMeanVarStat(){
		System.out.format("Unlabeled Avg\tUnlabeled Var\tLabeled Avg\tLabeled Var\n");
		for(int i=0; i<m_kFold; i++){
			System.out.format("%.4f\t%.4f\t%.4f\t%.4f\t\n", m_simiStat[i][0], m_simiStat[i][1], m_simiStat[i][2], m_simiStat[i][3]);
		}
	}
	
	
	// Added by Lin to test if clustering works for l2r.
	//The test for random walk algorithm.
	@Override
	public double test(){
		/***Construct the nearest neighbor graph****/
		constructGraph(false);
		
		if (m_pred_last==null || m_pred_last.length<m_U)
			m_pred_last = new double[m_U]; //otherwise we can reuse the current memory
		
		//record the starting point
		for(int i=0; i<m_U; i++)
			m_pred_last[i] = m_nodeList[i].m_pred;//random walk starts from multiple learner
		
		/***use random walk to solve matrix inverse***/
		System.out.println("Random walk starts:");
		int iter = 0;
		double diff = 0, accuracy;
		do {
			if (m_weightedAvg)
				accuracy = randomWalkByWeightedSum();	
			else
				accuracy = randomWalkByMajorityVote();
			
			diff = updateFu();
			System.out.format("Iteration %d, converge to %.3f with accuracy %.4f...\n", ++iter, diff, accuracy);
		} while(diff > m_delta && iter<50);//maximum 50 iterations 
		
		/***check the purity of newly constructed neighborhood graph after random walk with ground-truth labels***/
		SimilarityCheck();
		checkSimilarityVariance();
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_nodeList[i].m_pred));			
		}
		
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans, clusterNo;
		double[] accStat = new double[10];
		for(int i = 0; i < m_U; i++) {
			//pred = getLabel(m_fu[i]);
			pred = getLabel(m_nodeList[i].m_pred);
			ans = m_testSet.get(i).getYLabel();
			clusterNo = m_testSet.get(i).getClusterNo();
			m_TPTable[pred][ans] += 1;
			if(pred == ans)
				accStat[clusterNo*2]++;
			accStat[clusterNo*2+1]++;
			
			if (pred != ans) {
				if (m_debugOutput!=null)
					debug(m_testSet.get(i));
			} else {
				if (m_debugOutput!=null && Math.random()<0.02)
					debug(m_testSet.get(i));
				acc ++;
			}
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		for(int i=0; i<5 ;i++)
			System.out.format("%.4f\t", accStat[i*2]/accStat[i*2+1]);
		System.out.println();
		return acc/m_U;
	}
}
