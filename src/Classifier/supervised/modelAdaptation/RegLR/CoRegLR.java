/**
 * 
 */
package Classifier.supervised.modelAdaptation.RegLR;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._User;

/**
 * @author Hongning Wang
 * collaboratively regularized logistic regression model
 */
public class CoRegLR extends RegLR {

	double m_eta2; // weight for collaborative regularization
	int m_topK;
	SimType m_sType = SimType.ST_BoW;// default neighborhood by BoW
	
	public CoRegLR(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, int topK) {
		super(classNo, featureSize, featureMap, globalModel);
		m_eta2 = 0.5;
		m_topK = topK; // when topK<0, we will use a fully connected graph 
		
		// the only possible test modes for CoLinAdapt is batch mode
		m_testmode = TestMode.TM_batch;		
	}

	public void setTradeOffs(double eta1, double eta2) {
		m_eta1 = eta1;
		m_eta2 = eta2;
	}
	
	public void setSimilarityType(SimType sType) {
		m_sType = sType;
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList){	
		int vSize = m_featureSize+1;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoRegLRAdaptStruct(user, i, vSize, m_topK));
		}
		
		//huge space consumption		
		_CoRegLRAdaptStruct.sharedW = new double[vSize*m_userList.size()];
		
		//step 3: construct neighborhood graph
		constructNeighborhood(m_sType);
	}
	
	//this will be only called once in CoRegLR
	@Override
	protected void initLBFGS(){ 
		int vSize = (m_featureSize+1)*m_userList.size();
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {
		double fValue = super.calculateFuncValue(u), R2 = 0;
		
		//R2 regularization
		_CoRegLRAdaptStruct ui = (_CoRegLRAdaptStruct)u, uj;
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			
			double diff, sum = 0;
			for(int i=0; i<=m_featureSize; i++) {
				diff = ui.getPWeight(i) - uj.getPWeight(i); 
				sum += diff * diff;
			}
			R2 += nit.m_value * sum;
		}
		return fValue + m_eta2*R2;
	}
	
	@Override
	protected void calculateGradients(_AdaptStruct u){		
		super.calculateGradients(u);
		gradientByR2(u);
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR2(_AdaptStruct user){		
		_CoRegLRAdaptStruct ui = (_CoRegLRAdaptStruct)user, uj;
		int vSize = m_featureSize+1, offseti = vSize*ui.getId(), offsetj;
		double coef, diff;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			offsetj = vSize*uj.getId();
			coef = 2 * m_eta2 * nit.m_value;
			
			for(int k=0; k<vSize; k++) {
				diff = coef * (ui.getPWeight(k) - uj.getPWeight(k));
				
				// update ui's gradient
				m_g[offseti + k] += diff;
				
				// update uj's gradient
				m_g[offsetj + k] -= diff;
			}			
		}
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = (m_featureSize+1)*m_userList.size(), displayCount = 0;
		double oldMag = 0;
		
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
				
				// accumulate function values and gradients from each user
				for(_AdaptStruct user:m_userList) {
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
				
				//added by Lin for stopping lbfgs.
				double curMag = gradientTest();
				if(Math.abs(oldMag -curMag)<0.1) 
					break;
				oldMag = curMag;
				
				if (m_displayLv==2) {
					System.out.println("Fvalue is " + fValue);
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					
					if (++displayCount%100==0)
						System.out.println();
				} 
				oldFValue = fValue;
				
				LBFGS.lbfgs(vSize, 5, _CoRegLRAdaptStruct.getSharedW(), fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, sharedW is updated.
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}		
		
		setPersonalizedModel();
		return oldFValue;
	}
	
	@Override
	protected void setPersonalizedModel() {
		int vSize = m_featureSize+1;
		m_pWeights = new double[m_featureSize+1];
		double[] sharedW = _CoRegLRAdaptStruct.getSharedW();
		for(_AdaptStruct user:m_userList) {
			System.arraycopy(sharedW, user.getId()*vSize, m_pWeights, 0, vSize);
			user.setPersonalizedModel(m_pWeights);
		}
	}
}
