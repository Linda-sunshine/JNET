/**
 * 
 */
package Classifier.supervised.modelAdaptation.RegLR;

import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import utils.Utils;

/**
 * @author Hongning Wang
 * online learning of RegLR
 */
public class asyncRegLR extends RegLR {
	double m_initStepSize = 0.50;
	
	public asyncRegLR(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		
		// all three test modes for asyncRegLR is possible, and default is online
		m_testmode = TestMode.TM_online;
	}
	
	public static double getStepSize(double initStepSize, _AdaptStruct user) {
		return (0.5+0.5*Math.random()) * initStepSize/(2.0+user.getUpdateCount());
	}
	
	public void setInitStepSize(double initStepSize) {
		m_initStepSize = initStepSize;
	}
	
	//this is online training in each individual user
	@Override
	public double train(){
		double gNorm, gNormOld = Double.MAX_VALUE;;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		
		initLBFGS();
		init();
		for(_AdaptStruct user:m_userList) {
			while(user.hasNextAdaptationIns()) {
				// test the latest model before model adaptation
				if (m_testmode != TestMode.TM_batch &&(doc = user.getLatestTestIns()) != null) {
					perfStat = user.getPerfStat();
					predL = predict(doc, user);
					trueL = doc.getYLabel();
					perfStat.addOnePredResult(predL, trueL);
				} // in batch mode we will not accumulate the performance during adaptation				
				
				// prepare to adapt: initialize gradient	
				Arrays.fill(m_g, 0);
				calculateGradients(user);
				gNorm = gradientTest();
				
				if (m_displayLv==1) {
					if (gNorm<gNormOld)
						System.out.print("o");
					else
						System.out.print("x");
				}
				
				//gradient descent
				gradientDescent(user, m_initStepSize, m_g);
				gNormOld = gNorm;
			}
			
			if (m_displayLv>0)
				System.out.println();
		}
		
		setPersonalizedModel();
		return 0;//we do not evaluate function value
	}
	
	// update this current user only
	public static void gradientDescent(_AdaptStruct user, double initStepSize, double[] g) {
		double stepSize = asyncRegLR.getStepSize(initStepSize, user);
		Utils.add2Array(user.getUserModel(), g, -stepSize);
		user.incUpdatedCount(1.0);
	}	
	
	@Override
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationCacheSize();
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review, 1.0);//equal weight for the user's own adaptation data
	}
}
