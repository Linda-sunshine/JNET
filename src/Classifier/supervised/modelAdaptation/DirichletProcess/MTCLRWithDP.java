package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
/**
 * In the class, we extend the CLR to multi-task learning.
 * Intead of clusters, we also have a global part.
 * @author lin
 *
 */
public class MTCLRWithDP extends CLRWithDP {
	public static double[] m_supWeights; // newly learned global model
	// parameters for global part in multi-task learning. 
	
	public MTCLRWithDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_supWeights = new double[m_dim];
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
	protected void initPriorG0(){
		m_G0 = new NormalPrior(m_abNuA[0], m_abNuA[1]);//only for w_u
		m_G0.sampling(m_gWeights, m_supWeights);// sample super user's weights.
	}

	@Override	
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		double sum = m_q * Utils.dotProduct(m_supWeights, fvs, 0) + Utils.dotProduct(((_DPAdaptStruct)u).getThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}
	
	@Override
	protected double calculateR1(){
		double R1 = super.calculateR1();//w_u should be close to 0		
		
		// super model part.
		R1 += m_G0.logLikelihood(m_gWeights, m_supWeights, m_eta2);
		for(int i=m_kBar*m_dim; i<m_g.length; i++)//w_s should be close to w_0
			m_g[i] += m_eta2 * (m_models[i]-m_gWeights[i%m_dim])/(m_abNuA[1]*m_abNuA[1]);
				
		return R1;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;		
		int n; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		
		int offset = m_dim*cIndex, offsetSup = m_dim*m_kBar;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
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
	
	@Override
	protected void setThetaStars(){
		super.setThetaStars();
		System.arraycopy(m_models, m_kBar*m_dim, m_supWeights, 0, m_dim);
	}
	
	@Override
	public String toString() {
		return String.format("MTCLRWithDP[dim:%d,q:%.4f,M:%d,alpha:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim,m_q, m_M, m_alpha, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}	
	
	@Override
	protected void setPersonalizedModel() {
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			Utils.add2Array(user.getThetaStar().getModel(), m_supWeights, m_q);
			user.setPersonalizedModel(user.getThetaStar().getModel());
		}
	}
}
