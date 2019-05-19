package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._RankItem;
import structures._User;

public class MTCoLinAdapt extends MTLinAdapt {

	public MTCoLinAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap, String featureGroupMap4Sup) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap, featureGroupMap4Sup);
	}
	
	@Override
	public String toString() {
		return String.format("MT-CoLinAdapt[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,lambda1:%.3f,lambda2:%.3f,k:%d,NB:%s]", 
				m_dim, m_eta1, m_eta2, m_eta3, m_eta4, m_eta3, m_eta4, m_topK, m_sType);
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList){
		super.loadUsers(userList);
		constructNeighborhood(m_sType);
	}
	
	@Override
	// Since I cannot access the method in LinAdapt or in RegLR, I Have to rewrite.
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u);
		gradientByR2(u);
	}
	
	@Override
	//Calculate the function value of the new added instance.
	protected double calculateFuncValue(_AdaptStruct u){
		double fValue = super.calculateFuncValue(u), R2 = 0, diffA, diffB;
		
		//R2 regularization
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			diffA = 0;
			diffB = 0;
			for(int k=0; k<m_dim; k++) {
				diffA += (ui.getScaling(k) - uj.getScaling(k)) * (ui.getScaling(k) - uj.getScaling(k));
				diffB += (ui.getShifting(k) - uj.getShifting(k)) * (ui.getShifting(k) - uj.getShifting(k));
			}
			R2 += nit.m_value * (m_eta3*diffA + m_eta4*diffB);
		}
		return fValue + R2;
	}
}
