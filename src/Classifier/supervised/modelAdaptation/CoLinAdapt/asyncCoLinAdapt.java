/**
 * 
 */
package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;

/**
 * @author Hongning Wang
 * asynchronized CoLinAdapt with zero order gradient update, i.e., we will only touch the current user's gradient
 */
public class asyncCoLinAdapt extends CoLinAdapt {
	double m_initStepSize = 1.50;
	int[] m_userOrder; // visiting order of different users during online learning
	
	public asyncCoLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		
		// all three test modes for asyncCoLinAdapt is possible, and default is online
		m_testmode = TestMode.TM_online;
	}
	
	@Override
	public String toString() {
		return String.format("asyncCoLinAdapt[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_eta2, m_eta3, m_eta4, m_topK, m_sType);
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
		_CoLinAdaptStruct uj, ui = (_CoLinAdaptStruct)user;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
	}
	
	//we will only update ui but keep uj as constant
	void gradientByR2(_CoLinAdaptStruct ui, _CoLinAdaptStruct uj, double sim) {
		double coef = 2 * sim, dA, dB;
		int offset = m_dim*2*ui.getId();
		
		for(int k=0; k<m_dim; k++) {
			dA = coef * m_eta3 * (ui.getScaling(k) - uj.getScaling(k));
			dB = coef * m_eta4 * (ui.getShifting(k) - uj.getShifting(k));
			
			// update ui's gradient
			m_g[offset + k] += dA;
			m_g[offset + k + m_dim] += dB;
		}
	}
	
	protected double gradientTest(_AdaptStruct user) {
		int offset, uid = 2*m_dim*user.getId();
		double magA = 0, magB = 0;
		
		for(int i=0; i<m_dim; i++){
			offset = uid + i;
			magA += m_g[offset]*m_g[offset];
			magB += m_g[offset+m_dim]*m_g[offset+m_dim];
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	@Override
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationCacheSize();
	}
	
	//this is online training in each individual user
	@Override
	public double train(){
		double gNorm, gNormOld = Double.MAX_VALUE;
		int updateCount = 0;
		_CoLinAdaptStruct user;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		
		initLBFGS();
		init();
		for(int t=0; t<m_userOrder.length; t++) {
			user = (_CoLinAdaptStruct)m_userList.get(m_userOrder[t]);

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
				//gradientDescent(user, asyncLinAdapt.getStepSize(initStepSize, user));
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
	void gradientDescent(_CoLinAdaptStruct user, double initStepSize, double inc) {
		double a, b, stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int offset = 2*m_dim*user.getId();
		for(int k=0; k<m_dim; k++) {
			a = user.getScaling(k) - stepSize * m_g[offset + k];
			user.setScaling(k, a);
			
			b = user.getShifting(k) - stepSize * m_g[offset + k + m_dim];
			user.setShifting(k, b);
		}
		user.incUpdatedCount(inc);
	}	
}
