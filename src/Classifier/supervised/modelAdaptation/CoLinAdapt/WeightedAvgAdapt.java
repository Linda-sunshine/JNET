package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.HashMap;

import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;

/**
 * Full feature based average model adaptation
 * @author Lin Gong
 *
 */
public class WeightedAvgAdapt extends WeightedAvgTransAdapt {
	
	public WeightedAvgAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_dim = m_featureSize + 1; // We use all features to do average.
	}

	@Override
	public String toString() {
		return String.format("WeightedAvgAdapt[dim:%d,eta1:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_topK, m_sType);
	}
	
	@Override
	protected int getVSize() {
		return m_dim*m_userList.size();
	}
	
	@Override
	void constructUserList(ArrayList<_User> userList) {
		int vSize = m_dim;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, -1, i, m_topK));//we will not create transformation matrix for this user
			user.setModel(m_gWeights); // Initiate user weights with global weights.
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptStruct.sharedA = new double[getVSize()];
		
		//step 2: copy each user's weights to shared A(weights) in _CoLinAdaptStruct		
		for(int i=0; i<m_userList.size(); i++)
			System.arraycopy(m_gWeights, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
	}
	
	@Override
	// In this logit function, we need to sum over all the neighbors of the current user.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;
		// The user itself.
		double sum = ui.getSelfSim() * ui.linearFunc(fvs, 0);		
		// Traverse all neighbors of the current user.
		for(_RankItem nit: ui.getNeighbors()){
			_CoLinAdaptStruct uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);
			sum += nit.m_value * uj.linearFunc(fvs, 0);
		}
		return Utils.logistic(sum);
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {		
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		// Likelihood of the user.
		double L = calcLogLikelihood(user); //log likelihood.
		// regularization between the personal weighs and global weights.
		double R1 = m_eta1 * Utils.euclideanDistance(user.getPWeights(), m_gWeights);// 0.5*(a[i]-1)^2
		return R1 - L;
	}
	
	//shared gradient calculation by batch and online updating
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		int n, offset = m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);
		
		// Current user's info: Bias term + other features.
		m_g[offset] -= delta*ui.getSelfSim(); // \theta_{ii}*x_0 and x_0=1
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= delta * ui.getSelfSim() * fv.getValue();//\theta_{ii}*x_d
		}
		
		// Neighbors' info.
		for(_RankItem nit: ui.getNeighbors()) {
			offset = m_dim*nit.m_index;
			m_g[offset] -= delta * nit.m_value; // neighbors' bias term.
			for(_SparseFeature fv: review.getSparse()){
				n = fv.getIndex() + 1;
				m_g[offset + n] -= delta * nit.m_value * fv.getValue(); // neighbors' other features.
			}
		}
	}

	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR1(_AdaptStruct u){
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		double[] pWeights = user.getPWeights();
		int offset = m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		//R1 regularization part
		for(int k=0; k<m_dim; k++)
			m_g[offset + k] += 2 * m_eta1 * (pWeights[k]-m_gWeights[k]);// (w_i-w_g)
	}
	
	@Override
	protected void initPerIter() {
		super.initPerIter();
		preparePersonalizedModels();
	}

	void preparePersonalizedModels(){
		_CoLinAdaptStruct user;
		double[] pWeights;
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_CoLinAdaptStruct)m_userList.get(i);
			pWeights = user.getPWeights();
			
			System.arraycopy(_CoLinAdaptStruct.sharedA, i*m_dim, pWeights, 0, m_dim);
		}
	}
	
	@Override
	public void setPersonalizedModel(){	
		_CoLinAdaptStruct ui;
		double[] pWeights;
		int offset;
		
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			pWeights = ui.getPWeights();
			offset = i * m_dim;
			
			for(int n=0; n<m_dim; n++)
				pWeights[n] = ui.getSelfSim() * _CoLinAdaptStruct.sharedA[offset+n];
			
			//traverse all the neighbors
			for(_RankItem nit: ui.getNeighbors()) {
				offset = nit.m_index * m_dim;//get neighbor's index				
				for(int n=0; n<m_dim; n++) 
					pWeights[n] += nit.m_value * _CoLinAdaptStruct.sharedA[offset+n];
			}
		}
	}
	
	@Override
	protected double gradientTest() {
		int offset, uid;
		double mag = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*m_dim;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				mag += m_g[offset]*m_g[offset];
			}
		}
		if (m_displayLv==2)
			System.out.format("Gradient magnitude: %.5f\n", mag);
		return mag;
	}
}
