/**
 * 
 */
package Classifier.supervised.modelAdaptation.RegLR;

import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;

/**
 * @author Hongning Wang
 * zero-order asynchronized CoRegLR implementation
 */
public class asyncCoRegLR extends CoRegLR {
	double m_initStepSize = 1.50;
	int[] m_userOrder; // visiting order of different users during online learning
	
	public asyncCoRegLR(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			int topK) {
		super(classNo, featureSize, featureMap, globalModel, topK);
		// all three test modes for asyncCoLinAdapt is possible, and default is online
		m_testmode = TestMode.TM_online;
	}

	public void setInitStepSize(double initStepSize) {
		m_initStepSize = initStepSize;
	}
	
	@Override
	protected void constructNeighborhood(SimType sType) {
		super.constructNeighborhood(sType);
		m_userOrder = constructReverseNeighborhood();
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct user) {		
		//Update gradients review by review within the latest adaptation cache.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review, 1.0); // equal weights for this user's own adaptation data
	}
	
	@Override
	protected void gradientByR2(_AdaptStruct user){		
		_CoRegLRAdaptStruct ui = (_CoRegLRAdaptStruct)user, uj;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoRegLRAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
	}
	
	//we will only update ui but keep uj as constant
	void gradientByR2(_AdaptStruct ui, _AdaptStruct uj, double sim) {
		double coef = 2 * sim * m_eta2, diff;
		int offset = (m_featureSize+1)*ui.getId();
		
		for(int k=0; k<m_featureSize+1; k++) {
			diff = coef  * (ui.getPWeight(k) - uj.getPWeight(k));
			
			// update ui's gradient only
			m_g[offset + k] += diff;
		}
	}
	
	protected double gradientTest(_AdaptStruct user) {
		int offset, uid = (m_featureSize+1)*user.getId();
		double mag = 0;
		
		for(int i=0; i<m_featureSize+1; i++){
			offset = uid + i;
			mag += m_g[offset] * m_g[offset];
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for user %d, b: %.5f\n", user.getId(), mag);
		return mag;
	}
	
	@Override
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationCacheSize();
	}
	
	//this is online training in each individual user
	@Override
	public double train() {
		double gNorm, gNormOld = Double.MAX_VALUE;
		int updateCount = 0;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		_CoRegLRAdaptStruct user;
		
		initLBFGS();
		init();
		for(int t=0; t<m_userOrder.length; t++) {
			user = (_CoRegLRAdaptStruct)m_userList.get(m_userOrder[t]);
			
			if(user.hasNextAdaptationIns()) {
				// test the latest model
				if (m_testmode!=TestMode.TM_batch && (doc = user.getLatestTestIns()) != null) {
					perfStat = user.getPerfStat();
					predL = predict(doc, user);
					trueL = doc.getYLabel();
					perfStat.addOnePredResult(predL, trueL);
				} // in batch mode we will not accumulate the performance during adaptation			
				
				// prepare to adapt: initialize gradient	
				Arrays.fill(m_g, 0); 
				calculateGradients(user);
				gNorm = gradientTest(user);
				
				if (m_displayLv==1) {
					if (gNorm<gNormOld)
						System.out.print("o");
					else
						System.out.print("x");
				}
				
				//gradient descent
				gradientDescent(user, m_initStepSize, 1.0);
				gNormOld = gNorm;
				
				if (m_displayLv>0 && ++updateCount%100==0)
					System.out.println();
			}			
		}
		
		if (m_displayLv>0)
			System.out.println();
		
		setPersonalizedModel();		
		return 0; // we do not evaluate function value
	}
	
	// update this current user only
	void gradientDescent(_CoRegLRAdaptStruct user, double initStepSize, double inc) {
		double a, stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int offset = (m_featureSize+1)*user.getId();
		for(int k=0; k<=m_featureSize; k++) {
			a = user.getPWeight(k) - stepSize * m_g[offset + k];
			user.setPWeight(k, a);
		}
		user.incUpdatedCount(inc);
	}
}
