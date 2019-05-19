package Classifier.supervised.modelAdaptation.RegLR;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._SparseFeature;
import utils.Utils;
/**
 * The modified version of MT-SVM since it cannot be performed in online mode.
 * @author Lin
 */
public class asyncMTRegLR extends MTRegLR {//asyncRegLR
	double[] m_glbWeights; // shared global weights.
	double m_initStepSize = 0.05;
	
	public asyncMTRegLR(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_u = 1;
		m_glbWeights = new double[m_featureSize+1];
		System.arraycopy(m_gWeights, 0, m_glbWeights, 0, m_gWeights.length);//start from the old global model
	}
	
	@Override
	public String toString() {
		return String.format("asyncMTRegLR[u:%.2f,initStepSize: %.3f, eta1:%.3f]", m_u, m_initStepSize, m_eta1);
	}
	
	public void setInitStepSize(double initStepSize) {
		m_initStepSize = initStepSize;
	}
	
	@Override
	protected void initLBFGS(){
		m_eta1 = 1.0/m_userList.size();
		// This is asynchronized model update, at any time we will only touch one user together with the global model 
		if(m_g == null)
			m_g = new double[(m_featureSize+1)*2];
		Arrays.fill(m_g, 0);
		//no need to initialize m_diag, since this is asynconized model update
	}

	// Every user is represented by (u*global + individual)
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		int fid;
		// User bias and Global bias
		double sum = user.getPWeight(0) + m_u * m_glbWeights[0];
		for(_SparseFeature f:fvs){
			fid = f.getIndex()+1;
			// User model with Global model.
			sum += (user.getPWeight(fid) + m_u * m_glbWeights[fid]) * f.getValue();	
		}
		return Utils.logistic(sum);
	}
		
	@Override
	protected void gradientByFunc(_AdaptStruct user, _Doc review, double weight) {
		int n, offset = m_featureSize+1; // feature index
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[0] -= delta; //a[0] = w0*x0; x0=1
		m_g[offset] -= m_u*delta;// offset for the global part.
		
		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[n] -= delta * fv.getValue();// User part.
			m_g[offset + n] -= delta * m_u * fv.getValue(); // Global part.
		}
	}
	
	//should this R1 be |w_u+\mu w_g - w_0|, or just |w_u+\mu w_g|
	@Override
	protected void gradientByR1(_AdaptStruct user){
		int offset = m_featureSize+1;
		double v;
		//R1 regularization part
		for(int k=0; k<m_featureSize+1; k++){
			v = 2 * m_eta1 * (user.getPWeight(k) + m_u * m_glbWeights[k] - m_gWeights[k]);
//			v = 2 * m_eta1 * (user.getPWeight(k) + m_u * m_glbWeights[k]);
			m_g[k] += v;
			m_g[offset + k] += v * m_u;
		}
	}
	
	@Override
	public double train() {
		double gNorm, gNormOld = Double.MAX_VALUE;
		int predL, trueL;
		_Review doc;
		_AdaptStruct user;
		_PerformanceStat perfStat;
		double val;

		initLBFGS();
		init();
		try{			
			m_writer = new PrintWriter(new File(String.format("train_online_MTRegLR.txt")));
			for(int i=0; i<m_userList.size(); i++) {
				user = m_userList.get(i);
			
				while(user.hasNextAdaptationIns()) {
					// test the latest model before model adaptation
					if (m_testmode != TestMode.TM_batch && (doc = user.getLatestTestIns()) != null) {
						perfStat = user.getPerfStat();	
						val = logit(doc.getSparse(), user);
						predL = predict(doc, user);
						trueL = doc.getYLabel();
						perfStat.addOnePredResult(predL, trueL);
						m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
					} // in batch mode we will not accumulate the performance during adaptation				
				
					gradientDescent(user, m_initStepSize, 1.0);
				
					//test the gradient only when we want to debug
					if (m_displayLv>0) {
						gNorm = gradientTest();				
						if (m_displayLv==1) {
							if (gNorm<gNormOld)
								System.out.print("o");
							else
								System.out.print("x");
						}				
						gNormOld = gNorm;
					}
				}
				m_writer.flush();
				if (m_displayLv==1)
					System.out.println();
			} 
		}catch(IOException e){
			e.printStackTrace();
		}
		setPersonalizedModel();
		return 0;//we do not evaluate function value
	}
	
	// update this current user only
	protected void gradientDescent(_AdaptStruct user, double initStepSize, double inc) {
		double stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int offset = m_featureSize+1;		
		double[] pWeight = user.getPWeights();
		
		//get gradient
		Arrays.fill(m_g, 0);
		calculateGradients(user);

		for(int k=0; k<m_featureSize+1; k++) {
			//update the individual user
			pWeight[k] -= stepSize * m_g[k];
			
			//update the shared global part.
			m_glbWeights[k] -= stepSize * m_g[offset+k];
		}
		
		//update the record of updating history
		user.incUpdatedCount(inc);
	}
}