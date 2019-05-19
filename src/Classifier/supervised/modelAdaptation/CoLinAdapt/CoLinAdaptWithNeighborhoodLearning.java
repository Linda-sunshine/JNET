package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._RankItem;
import structures._User;
import utils.Utils;

public class CoLinAdaptWithNeighborhoodLearning extends CoLinAdapt {
	
	int m_fDim; // The dimension of feature for a pair of users, including bias term.
	int m_neiDim; // The dimension of neighborhood learninng
	double[] m_w; // The array contains all user's weights.
	double[] m_gN, m_diagN; // The gradient matrix and diag matrix, used in lbfgs.
	double m_lambda; // Parameter for regularization.
	
	double m_diffA, m_diffSim, m_tol;// Tolerance of the comparison.
	double[] m_APre, m_ACur, m_simA; //, m_simNei;
	double[][] m_xijs; // The structure stores all training instances.
	
	public CoLinAdaptWithNeighborhoodLearning (int classNo, int featureSize, HashMap<String, Integer> featureMap,
			int topK, String globalModel, String featureGroupMap, int fDim) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_lambda = 0.1;
		m_fDim = fDim;
	}
	
	// Load users before initNeiLearn(), we need the number of users.
	void initNeiLearn(){
		
		m_neiDim = m_fDim * m_userList.size();
		m_w = new double[m_neiDim];
		m_gN = new double[m_neiDim];
		m_diagN = new double[m_neiDim];
		
		m_diffA = Double.MAX_VALUE;
		m_diffSim = Double.MAX_VALUE;
		m_simA = new double[m_topK * m_userList.size()];
		
		m_tol = 1e-10;
	}

	void initLBFGSNeiLearn(){
		Arrays.fill(m_diagN, 0);
		Arrays.fill(m_gN, 0);
	}
	
	@Override
	public double train(){
		initNeiLearn();
		constructXijs(); // Construct all training instances.
		m_ACur = _CoLinAdaptStruct.getSharedA();
		double fValue = 0;

		while(m_diffA > m_tol && m_diffSim > m_tol){
			// Step 1: Get the previous A matrix, copy into m_preA.
			m_APre = Arrays.copyOfRange(m_ACur, 0, m_ACur.length);
			
			// Step 2: Train the A matrix, with m_curA, update the m_diffA.
			fValue = super.train();
			updateDiffA();
			
			// Step 3: Calculate the similarity between As for training.
			calcSimilarityAs();
			
			// Step 4: Train the LR for w to calc similarity.
			neighborhoodTrain(); // Learn weights for calculating similarity.
			
			// Step 5: Update the neighbor similarity based on the new weights.
			// m_DiffSim is calculated inside update.
			updateNeighborhood();
		}	
		return fValue;
	}
	
	// Construct all training instances.
	public void constructXijs(){
		int j = 0; // Index of neighbors.
		_CoLinAdaptStruct ui;
		m_xijs = new double[m_topK * m_userList.size()][];
		
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct) m_userList.get(i);
			j = 0;

			// Traverse all neighbors.
			for(_RankItem nit: ui.getNeighbors()){
				// Construct the training instance.
				m_xijs[m_topK * i + j] = constructXij(ui.getUser(), m_userList.get(nit.m_index).getUser());
				j++;
			}
		}
	}
	
	public double[] constructXij(_User ui, _User uj){
		double[] xij = new double[m_fDim];
		xij[0] = 1; // bias term
		xij[1] = ui.getBoWSim(uj); // cosine similarity.
		xij[2] = ui.getSVDSim(uj); // svd similarity.
		// Analyze the special case.
		if(Double.isNaN(xij[2]))
			xij[2] = 0;
		return xij;
	}
	
	// Update the different between previous A and current A.
	public void updateDiffA(){
		m_diffA = 0;
		if(m_APre.length != m_ACur.length)
			System.err.print("The two vectors are not comparable with different dimensions!");
		
		for(int i=0; i<m_APre.length; i++)
			m_diffA += (m_APre[i] - m_ACur[i]) * (m_APre[i] - m_ACur[i]);
	}
	
	// Due to normalization, pre-compute the similarity among As before hand.
	public void calcSimilarityAs(){
		_CoLinAdaptStruct ui;
		double[] sims;
//		m_simA = new double[m_userList.size() * m_topK];
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct) m_userList.get(i);
			sims = calcSimA(ui, i);
			System.arraycopy(sims, 0, m_simA, i * m_topK, m_topK);
		}
	}

	// Calculate the normalized similarity between a user and its neighbors.
	public double[] calcSimA(_CoLinAdaptStruct ui, int i){
		int index = 0;
		double sum = 0;
		
		// Normalized similarity between every pair of users.
		double[] sims = new double[ui.getNeighbors().size()];
		double[] Ai = new double[m_dim * 2];
		double[] Aj = new double[m_dim * 2];
		System.arraycopy(_CoLinAdaptStruct.sharedA, i, Ai, 0, m_dim * 2);

		for(_RankItem uj: ui.getNeighbors()){
			System.arraycopy(_CoLinAdaptStruct.sharedA, uj.m_index, Aj, 0, m_dim * 2);
			sims[index] = Utils.cosine(Ai, Aj);
			sum += sims[index];
			index++;
		}
		// Normalize the similarities.
		for(int k=0; k<sims.length; k++)
			sims[k] /= sum;
		return sims;
	}
	
	// The neighborhood training process.
	public void neighborhoodTrain(){
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue;
		int fSize = m_neiDim;
		initLBFGSNeiLearn();
		
		try {
			do {
				fValue = calculateFValueGradients();
				LBFGS.lbfgs(fSize, 6, m_w, fValue, m_gN, false, m_diagN, iprint, 1e-4, 1e-10, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
	}
	
	public double calculateFValueGradients(){
		_CoLinAdaptStruct ui;
		double fValue = 0, exp = 0; //Xij is the training instances, currently, we only have one 
		double[] wi = new double[m_fDim]; // The training instance in neighborhood learning.
//		double mag = 0;
		
		//Add the L2 Regularization.
		double L2 = 0, b = 0;
		for(int i=0; i<m_w.length; i++){
			b = m_w[i];
			m_gN[i] = 2*m_lambda*b;
			L2 += b*b;
		}
		
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct) m_userList.get(i);
		
			// Traverse all neighbors.
			wi = Arrays.copyOfRange(m_w, i * m_fDim, (i + 1) * m_fDim);
			for(int j=0; j<ui.getNeighbors().size(); j++){
				exp = Math.exp(-Utils.dotProduct(m_xijs[i * m_topK + j], wi));
				fValue += m_simA[i * m_topK + j] * Math.log(1 + exp);
				
				// Update the gradients.
				for(int k=0; k<m_fDim; k++)
					m_gN[i * m_fDim + k] += exp * (-m_xijs[i * m_topK + j ][k]) * m_simA[i * m_topK + j] / (1 + exp);
			}
		}
//		for(double g: m_gN)
//			mag += g * g;
//		/**We have "-" for fValue and we want to maxmize the loglikelihood.
//		* LBFGS is minization, thus, we take off the negative sign in calculation.*/
//		System.out.format("Fvalue: %.4f\tGradient: %.4f\n", fValue, mag);
		return fValue + m_lambda*L2;
	}
	
	// Calculate the similarity with the learned weights.
	public void updateNeighborhood(){
		m_diffSim = 0;
		_CoLinAdaptStruct ui;
		int j = 0;// Index of neighbors.
		double sim = 0;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct) m_userList.get(i);
			j = 0;
			// Traverse all neighbors.
			for(_RankItem nit: ui.getNeighbors()){
				sim = logit(i, j); // New similarity between a pair of users.
				m_diffSim += (sim - nit.m_value) * (sim - nit.m_value);
				nit.m_value = sim; // Update similarity of neighbor.
				j++;
			}
		}
	}
	
	// Calculate logit function.
	public double logit(int i, int j){
			
		double sim = 0;
		double[] w = Arrays.copyOfRange(m_w, i * m_fDim, (i + 1) * m_fDim);
		double[] xij = m_xijs[m_topK * i + j];
		if (w.length != xij.length){
			System.err.print("Wrong dimension in computing logit function.\n");
			return sim;
		}
		else
			sim = 1 / (1 + Math.exp(-Utils.dotProduct(w, xij)));
		return sim;
	}
}
