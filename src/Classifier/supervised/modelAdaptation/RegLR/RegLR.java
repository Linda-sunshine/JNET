/**
 * 
 */
package Classifier.supervised.modelAdaptation.RegLR;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._Doc.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

/**
 * @author Hongning Wang
 * Global model regularized logistic regression model
 */
public class RegLR extends ModelAdaptation {
	//Trade-off parameters	
	protected double m_eta1; // weight for scaling in R1.
		
	//shared space for LBFGS optimization
	protected double[] m_diag; //parameter used in lbfgs.
	protected double[] m_g;//optimized gradients. 
		
	protected PrintWriter m_writer;

	public RegLR(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		// default value of trade-off parameters
		m_eta1 = 0.5;
		
		// the only test mode for RegLR is batch
		m_testmode = TestMode.TM_batch;
	}
	
	public RegLR(int classNo, int featureSize, String globalModel) {
		super(classNo, featureSize, globalModel);
		// default value of trade-off parameters
		m_eta1 = 0.5;
		
		// the only test mode for RegLR is batch
		m_testmode = TestMode.TM_batch;
	}
	public void setR1TradeOff(double eta1) {
		m_eta1 = eta1;
	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList) {
			m_userList.add(new _AdaptStruct(user));
			user.initModel(m_featureSize+1);
		}
	}
	
	protected void initLBFGS(){
		if(m_g == null)
			m_g = new double[m_featureSize+1];
		if(m_diag == null)
			m_diag = new double[m_featureSize+1];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
	}
	
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		double sum = user.getPWeight(0); // bias term
		for(_SparseFeature f:fvs) 
			sum += user.getPWeight(f.getIndex()+1) * f.getValue();		
		return Utils.logistic(sum);
	}
	
	protected int predict(_Doc review, _AdaptStruct user) {
		if (review==null)
			return -1;
		else
			return logit(review.getSparse(), user)>0.5?1:0;
	}
	
	//Calculate the function value of the new added instance.
	protected double calcLogLikelihood(_AdaptStruct user){
		double L = 0; //log likelihood.
		double Pi = 0;
		
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue; // only touch the adaptation data
			
			Pi = logit(review.getSparse(), user);
			if(review.getYLabel() == 1) {
				if (Pi>0.0)
					L += Math.log(Pi);					
				else
					L -= Utils.MAX_VALUE;
			} else {
				if (Pi<1.0)
					L += Math.log(1 - Pi);					
				else
					L -= Utils.MAX_VALUE;
			}
		}
		if(getAdaptationSize(user) == 0)
			return 0;
		else {
			return m_LNormFlag ? L/getAdaptationSize(user) : L;
		}
	}
	
	//Calculate the function value of the new added instance.
	protected double calculateFuncValue(_AdaptStruct user){
		double L = calcLogLikelihood(user); //log likelihood.
		double R1 = 0;
		
		//Add regularization parts.
		for(int i=0; i<m_featureSize+1; i++)
			R1 += (user.getPWeight(i) - m_gWeights[i]) * (user.getPWeight(i) - m_gWeights[i]);//(w^u_i-w^g_i)^2
		return m_eta1*R1 - L;
	}

	protected void gradientByFunc(_AdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue;
			
			gradientByFunc(user, review, 1.0);//weight all the instances equally
		}
	}
	
	protected void gradientByFunc(_AdaptStruct user, _Doc review, double weight) {
		int n; // feature index
		int offset = (m_featureSize+1)*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), user));
		if (m_LNormFlag)
			delta /= getAdaptationSize(user);

		//Bias term.
		m_g[offset] -= delta; //a[0] = w0*x0; x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= delta * fv.getValue();
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR1(_AdaptStruct user){
		int offset = (m_featureSize+1)*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		//R1 regularization part
		for(int k=0; k<m_featureSize+1; k++)
			m_g[offset + k] += 2 * m_eta1 * (user.getPWeight(k) - m_gWeights[k]);// add 2*eta1*(w^u_k-w^g_k)
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void calculateGradients(_AdaptStruct user){
		gradientByFunc(user);
		gradientByR1(user);
	}
	
	protected double gradientTest() {
		double magG = Utils.L2Norm(m_g);
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude %.5f\n", magG);
		return magG;
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0, w[], oldFValue = Double.MAX_VALUE, totalFvalue = 0;
		
		init();
		for(_AdaptStruct user:m_userList) {			
			initLBFGS();
			iflag[0] = 0;
			try{
				w = user.getUserModel();
				oldFValue = Double.MAX_VALUE; 
				do{
					Arrays.fill(m_g, 0); // initialize gradient					
					fValue = calculateFuncValue(user);
					calculateGradients(user);
					
					if (m_displayLv==2) {
						System.out.println("Fvalue is " + fValue);
						gradientTest();
					} else if (m_displayLv==1) {
						if (fValue<oldFValue)
							System.out.print("o");
						else
							System.out.print("x");
					} 
					oldFValue = fValue;
					
					LBFGS.lbfgs(w.length, 6, w, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-32, iflag);//In the training process, A is updated.
				} while(iflag[0] != 0);
			} catch(ExceptionWithIflag e) {
				if (m_displayLv>0)
					System.out.print("X");
				else
					System.out.println("X");
			}
			
			if (m_displayLv>0)
				System.out.println();			
			
			totalFvalue += fValue;
		}
		
		setPersonalizedModel();
		return totalFvalue;
	}
	
	@Override
	protected void setPersonalizedModel() {
		//personalized model has already been set in each user
	}
}
