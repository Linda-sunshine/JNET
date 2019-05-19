/**
 * 
 */
package Classifier.supervised.modelAdaptation.RegLR;

import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._RankItem;
import structures._Review;

/**
 * @author Hongning Wang
 * first-order asynchronized CoRegLR implementation
 */
public class asyncCoRegLRFirstOrder extends asyncCoRegLR {

	double m_neighborsHistoryWeight; // used to reweight the gradient of historical observations from neighbors
	public asyncCoRegLRFirstOrder(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			int topK, double neighborsHistoryWeight) {
		super(classNo, featureSize, featureMap, globalModel, topK);
		m_neighborsHistoryWeight = neighborsHistoryWeight;
	}

	@Override
	protected void calculateGradients(_AdaptStruct user){		
		super.calculateGradients(user);
		if (m_neighborsHistoryWeight>0)
			cachedGradientByNeighorsFunc(user, m_neighborsHistoryWeight);
		gradientByRelatedR1((_CoRegLRAdaptStruct)user);
	}
	 
	//Calculate the reweighted gradients from neighbors' historical observations
	protected void cachedGradientByNeighorsFunc(_AdaptStruct user, double weight){		
		_CoRegLRAdaptStruct ui = (_CoRegLRAdaptStruct)user, uj;

		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			for(_Review r:uj.getAdaptationCache())
				gradientByFunc(uj, r, weight);
		}

		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			for(_Review r:uj.getAdaptationCache())
				gradientByFunc(uj, r, weight);
		}
	}
	
	@Override
	void gradientByR2(_AdaptStruct ui, _AdaptStruct uj, double sim) {
		double coef = 2 * sim * m_eta2, diff;
		int offseti = (m_featureSize+1)*ui.getId(), offsetj = (m_featureSize+1)*uj.getId();
		
		for(int k=0; k<=m_featureSize; k++) {
			diff = coef * (ui.getPWeight(k) - uj.getPWeight(k));
			
			// update ui's gradient
			m_g[offseti + k] += diff;
			
			// update uj's gradient
			m_g[offsetj + k] -= diff;
		}
	}
	
	//compute gradient for all related user in first order connection (exclude itself)
	void gradientByRelatedR1(_CoRegLRAdaptStruct ui) {
		_CoRegLRAdaptStruct uj;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			gradientByR1(uj);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			gradientByR1(uj);
		}
	}
	
	@Override
	void gradientDescent(_CoRegLRAdaptStruct ui, double initStepSize, double inc) {
		super.gradientDescent(ui, initStepSize, inc);//update the current user
		
		_CoRegLRAdaptStruct uj;		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			super.gradientDescent(uj, initStepSize, inc/3);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			super.gradientDescent(uj, initStepSize, inc/3);
		}
	}
}
