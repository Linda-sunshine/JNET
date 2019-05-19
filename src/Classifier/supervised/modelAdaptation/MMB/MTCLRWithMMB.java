package Classifier.supervised.modelAdaptation.MMB;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import utils.Utils;

import java.util.HashMap;

public class MTCLRWithMMB extends CLRWithMMB {
	public static double[] m_supWeights; // newly learned global model

	public MTCLRWithMMB(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
		m_supWeights = new double[m_dim];
	}
	
	public MTCLRWithMMB(int classNo, int featureSize, String globalModel, double[] betas) {
		super(classNo, featureSize, globalModel, betas);
		m_supWeights = new double[m_dim];
	}
	
	@Override
	protected void setThetaStars() {
		super.setThetaStars();
		// Assign the optimized weights to the global model.
		System.arraycopy(m_models, m_kBar*m_dim, m_supWeights, 0, m_dim);
	}
	@Override
	protected void accumulateClusterModels(){
		super.accumulateClusterModels();
		System.arraycopy(m_supWeights, 0, m_models, m_dim*m_kBar, m_dim);
	}

	@Override
	protected int getVSize(){
		return m_dim*(m_kBar+1);
	}

	@Override
	public String toString() {
		return String.format("MTCLRWithMMB[dim:%d,lmDim:%d,q:%.3f,M:%d,rho:%.5f,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim, m_lmDim,m_q, m_M, m_rho, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}

	@Override
	protected void initPriorG0(){
		super.initPriorG0();
		//do we assume the global model has the same prior as user models?
		m_G0.sampling(m_supWeights);// sample super user's weights.
	}
	
	@Override
	protected double logit(_SparseFeature[] fvs, _Review r){
		double sum = m_q * Utils.dotProduct(m_supWeights, fvs, 0) + Utils.dotProduct(r.getHDPThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}
	
	@Override
	protected double calculateR1(){
		double R1 = super.calculateR1();//w_u should be close to 0

		// super model part.
		R1 += m_G0.logLikelihood(m_gWeights, m_supWeights, m_eta2);
		
		// Gradient by the regularization.
		if (m_G0.hasVctMean()) {//we have specified the whole mean vector
			for(int i=m_kBar*m_dim; i<m_g.length; i++) 
				m_g[i] += m_eta2 * (m_models[i]-m_gWeights[i%m_dim]) / (m_abNuA[1]*m_abNuA[1]);
		} else {//we only have a simple prior
			for(int i=m_kBar*m_dim; i<m_g.length; i++)
				m_g[i] += m_eta2 * (m_models[i]-m_abNuA[0]) / (m_abNuA[1]*m_abNuA[1]);
		}

		return R1;
	}

	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_Review r = (_Review) review;
		int n; // feature index
		int cIndex = r.getHDPThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");

		int offset = m_dim*cIndex;
		int offsetSup = m_dim*m_kBar;
		double delta = weight * (r.getYLabel() - logit(r.getSparse(), r));

		//Bias term.
		g[offset] -= delta; //x0=1, each cluster.
		g[offsetSup] -= m_q*delta; // super model.

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			g[offset + n] -= delta * fv.getValue();// cluster model.
			g[offsetSup + n] -= delta * fv.getValue() * m_q;// super model.
		}
	}
}
