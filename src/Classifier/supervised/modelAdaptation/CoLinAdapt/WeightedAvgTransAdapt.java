package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

/**
 * Linear transformation based average model adaptation
 * @author Lin Gong
 *
 */
public class WeightedAvgTransAdapt extends CoLinAdapt {

	// By default, we use the cosine similarity between documents, we can also use 1/(topK+1).
	boolean m_cosSim = true;
	// default self similarity
	double m_selfSim = 1.0;
		
	public WeightedAvgTransAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}
	
	public void setNeighborWeight(boolean cosine) {
		m_cosSim = cosine;
	}
	
	@Override
	public String toString() {
		return String.format("WeightedAvgTransAdapt[dim:%d,eta1:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_topK, m_sType);
	}
	
	@Override
	public void constructNeighborhood(final SimType sType){		
		super.constructNeighborhood(sType);
		
		_CoLinAdaptStruct ui;		
		double sum; // the user's own similarity.
		// Normalize the similarity of neighbors.
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct) m_userList.get(i);
			sum = m_selfSim;
			
			// Collect the sum of similarity.
			for(_RankItem nit: ui.getNeighbors()) {
				if (m_cosSim)
					sum += nit.m_value;
				else
					sum ++;
			}
			
			// Update the user's similarity.
			ui.setSelfSim(m_selfSim/sum);
			for(_RankItem nit: ui.getNeighbors()){
				if(m_cosSim)
					nit.m_value /= sum;
				else 
					nit.m_value = 1/sum;
			}
		}
	}
	
	@Override
	// In this logit function, we need to sum over all the neighbors of the current user.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;		
		double sum = ui.getSelfSim() * linearFunc(fvs, ui);
		
		// Traverse all neighbors of the current user.
		for(_RankItem nit: ui.getNeighbors()){
			_CoLinAdaptStruct uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);			
			sum += nit.m_value * linearFunc(fvs, uj);
		}
		return Utils.logistic(sum);
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		
		double L = calcLogLikelihood(user); //log likelihood.
		double R1 = 0;
		
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1 * (user.getScaling(i)-1) * (user.getScaling(i)-1);//(a[i]-1)^2
			R1 += m_eta2 * user.getShifting(i) * user.getShifting(i);//b[i]^2
		}
		return R1 - L;
	}
	
	@Override
	// We want to user RegLR's calculateGradients while we cannot inherit from far than parent class.
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u); // inherit from LinAdapt
	}
	
	//shared gradient calculation by batch and online updating
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
			
		int n, k, offsetj;
		int offset = m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);
			
		// Current user's info: Bias term + other features.
		m_g[offset] -= delta*ui.getSelfSim()*m_gWeights[0]; // \theta_{ii}*w_g[0]*x_0 and x_0=1
		m_g[offset + m_dim] -= delta*ui.getSelfSim(); // \theta_{ii}*x_0
		
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= delta * ui.getSelfSim() * m_gWeights[n] * fv.getValue();//\theta_{ii}*x_d
			m_g[offset + k + m_dim] -= delta * ui.getSelfSim() * fv.getValue();
		}
			
		// Neighbors' info: Bias term + other features.
		for(_RankItem nit: ui.getNeighbors()) {
			offsetj = 2*m_dim*nit.m_index;
			// Bias term.
			m_g[offsetj] -= delta * nit.m_value * m_gWeights[0]; // neighbors' bias term.
			m_g[offsetj + m_dim] -= delta * nit.m_value;
			
			for(_SparseFeature fv: review.getSparse()){
				n = fv.getIndex() + 1;
				k = m_featureGroupMap[n];
				m_g[offsetj + k] -= delta * nit.m_value * m_gWeights[n] * fv.getValue(); // neighbors' other features.
				m_g[offsetj + m_dim + k] -= delta * nit.m_value * fv.getValue();
			}
		}
	}

	@Override
	public void setPersonalizedModel(){
		_CoLinAdaptStruct ui, uj;
		int k;
		
		for(int i=0; i<m_userList.size(); i++){
			ui = (_CoLinAdaptStruct)m_userList.get(i);			
			
			for(int n=0; n<m_featureSize+1; n++) {
				k = m_featureGroupMap[n];
				m_pWeights[n] = ui.getSelfSim() * (m_gWeights[n] * ui.getScaling(k) + ui.getShifting(k));
			}
			
			//traverse all the neighbors
			for(_RankItem nit: ui.getNeighbors()) {
				uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);	
				for(int n=0; n<m_featureSize+1; n++) {
					k = m_featureGroupMap[n];
					m_pWeights[n] += nit.m_value * (m_gWeights[n] * uj.getScaling(k) + uj.getShifting(k));
				}
			}
			ui.setPersonalizedModel(m_pWeights);
		}
	}
}
