package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._SparseFeature;
import structures._User;
/***
 * 
 * @author lin
 * In this class, super user is only represented by the weights, 
 * in logit function, personalized weights are represented as:
 * A_i(p*w_s+q*w_g)^T*x_d
 */
public class MTLinAdaptWithSupUserNoAdapt extends MTLinAdapt {

	protected double m_p; // The coefficient in front of w_s.
	protected double m_q; // The coefficient in front of w_g.
	protected double m_beta; // The coefficient in front of R1(w_s)
	
	public MTLinAdaptWithSupUserNoAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap, String featureGroupMap4Sup) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap, featureGroupMap4Sup);
		m_p = 1.0;
		m_q = 1.0;
		m_beta = 1.0;
	}
	
	public void setWsWgCoefficients(double p, double q){
		m_p = p;
		m_q = q;
	}
	
	public void setR14SupCoefficients(double beta){
		m_beta = beta;
	}
	
	@Override
	public String toString() {
		return String.format("MT-LinAdaptWithSupUserNoAdpt[dim:%d, eta1:%.3f,eta2:%.3f,p:%.3f,q:%.3f,beta:%.3f, personalized:%b]", 
				m_dim, m_eta1, m_eta2, m_p, m_q, m_beta, m_personalized);
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList){
		constructUserList(userList);
		System.arraycopy(m_sWeights, 0, m_A, 2*m_dim*m_userList.size(), m_sWeights.length);
	}
	
	@Override
	protected int getVSize() {
		return m_dim*2*m_userList.size() + m_gWeights.length;
	}
		
	@Override
	public double getSupWeights(int index){
		return m_p*m_A[m_dim*2*m_userList.size() + index] + m_q*m_gWeights[index];
	}
	
	// Calculate the R1 for the super user, As.
	protected double calculateRs(){
		int offset = m_userList.size() * m_dim * 2;
		double rs = 0;
		for(int i=0; i < m_sWeights.length; i++)
			rs += m_A[offset + i] * m_A[offset + i];
		return rs * m_beta;
	}
	
	// Gradients for the gs.
	protected void gradientByRs(){
		int offset = m_userList.size() * m_dim * 2;
		for(int i=0; i < m_sWeights.length; i++)
			m_g[offset + i] += 2 * m_beta * m_A[offset + i];
	}
	
	// Gradients from loglikelihood, contributes to both individual user's gradients and super user's gradients.
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		int n, k; // feature index and feature group index		
		int offset = 2*m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int offsetSup = 2*m_dim*m_userList.size();
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);

		// Bias term for individual user.
		m_g[offset] -= delta*getSupWeights(0); //a[0] = (p*w_s0+q*w_g0)*x0; x0=1
		m_g[offset + m_dim] -= delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= delta*ui.getScaling(0)*m_p; //a_s[0] = a_i0*p*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= delta*getSupWeights(n)*fv.getValue(); // (p*w_si+q*w_gi)*x_di
			m_g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
			
			m_g[offsetSup + n] -= delta*ui.getScaling(k)*m_p*fv.getValue(); // a_i*p*x_di
		}
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, offsetSup, uid;
		double magA = 0, magB = 0, magS = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dim]*m_g[offset+m_dim];
			}
		}

		offsetSup = vSize * m_userList.size();
		for(int i=0; i<m_sWeights.length; i++)
			magS += m_g[offsetSup+i] * m_g[offsetSup+i];
		
		if (m_displayLv==2)
			System.out.format("\t mag: %.4f\n", magA + magB + magS);
		return magA + magB + magS;
	}
}
