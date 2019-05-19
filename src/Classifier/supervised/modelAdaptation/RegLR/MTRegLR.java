package Classifier.supervised.modelAdaptation.RegLR;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
/***
 * This is the batch training of Multi-task regualized logistic regression.
 * @author lin
 */
public class MTRegLR extends RegLR {

	double m_u; // Trade-off paramters.
	double[] m_ws; // The weights of all the users.

	public MTRegLR(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_u = 1;
		m_eta1 = 0.001;
	}
	@Override
	public String toString() {
		return String.format("MTRegLR[u:%.2f,eta1:%.3f]", m_u, m_eta1);
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		super.loadUsers(userList);
		
		// Merge the weights of all the users and global part.
		m_ws = new double[(m_userList.size()+1)*(m_featureSize+1)];
		// Assume the last section is for global parts.
		System.arraycopy(m_gWeights, 0, m_ws, m_userList.size()*(m_featureSize+1), m_gWeights.length);//start from the old global model
		// Load users weights into one array for LBFGS.
//		for(_AdaptStruct u: m_userList)
//			System.arraycopy(u.getUserModel(), 0, m_ws, u.getId()*(m_featureSize+1), m_gWeights.length);//start from the old global model
	}

	@Override
	protected void initLBFGS(){
		// This is asynchronized model update, at any time we will only touch one user together with the global model 
		if(m_g == null)
			m_g = new double[(m_featureSize+1)*(m_userList.size()+1)];
		if(m_diag == null)
			m_diag = new double[m_g.length];
		Arrays.fill(m_g, 0);
		Arrays.fill(m_diag, 0);
	}
	
	public void setTradeOffParam(double u){
		m_u = Math.sqrt(u);
	}
	
	// Every user is represented by (u*global + individual)
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		int fid, uOffset, gOffset;
		uOffset = (m_featureSize+1)*user.getId();
		gOffset = (m_featureSize+1)*m_userList.size();
		// User bias and Global bias
		double sum = m_ws[uOffset] + m_u * m_ws[gOffset];
		for(_SparseFeature f:fvs){
			fid = f.getIndex()+1;
			// User model with Global model.
			sum += (m_ws[uOffset+fid] + m_u * m_ws[gOffset+fid]) * f.getValue();	
		}
		return Utils.logistic(sum);
	}
	
	//Calculate the function value of the new added instance.
	protected double calculateFuncValue(_AdaptStruct user){
		double L = super.calcLogLikelihood(user), R1 = 0; //log likelihood.
		int uOffset, gOffset;
		uOffset = (m_featureSize+1)*user.getId();
		gOffset = (m_featureSize+1)*m_userList.size();
		
		//Add regularization parts.
		for(int k=0; k<m_featureSize+1; k++)
			// (w_i+u*w_g-w_0)^2
			R1 += (m_ws[uOffset+k] + m_u * m_ws[gOffset+k] - m_gWeights[k])*(m_ws[uOffset+k] + m_u * m_ws[gOffset+k] - m_gWeights[k]);
		return m_eta1*R1 - L;
	}

	@Override
	protected void gradientByFunc(_AdaptStruct user, _Doc review, double weight) {
		int n, uOffset, gOffset;
		uOffset = (m_featureSize+1)*user.getId();
		gOffset = (m_featureSize+1)*m_userList.size();
		
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[uOffset] -= delta; //a[0] = w0*x0; x0=1
		m_g[gOffset] -= m_u*delta;// offset for the global part.
		
		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[uOffset + n] -= delta * fv.getValue();// User part.
			m_g[gOffset + n] -= delta * m_u * fv.getValue(); // Global part.
		}
	}
	//should this R1 be |w_u+\mu w_g - w_0|, or just |w_u+\mu w_g|
	@Override
	protected void gradientByR1(_AdaptStruct user){
		int uOffset, gOffset;
		uOffset = (m_featureSize+1)*user.getId();
		gOffset = (m_featureSize+1)*m_userList.size();
		
		double v;
		//R1 regularization part
		for(int k=0; k<m_featureSize+1; k++){
			v = 2 * m_eta1 * (m_ws[uOffset+k] + m_u * m_ws[gOffset+k] - m_gWeights[k]);
//			v = 2 * m_eta1 * (m_ws[uOffset+k] + m_u * m_ws[gOffset+k]);
			m_g[uOffset + k] += v;
			m_g[gOffset + k] += v * m_u;
		}
	} 
	
	//this is batch training in each individual user
	@Override
	public double train() {
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue, oldFValue = Double.MAX_VALUE;
		int displayCount = 0;
		_AdaptStruct user;

		initLBFGS();
		init();
		try {
			do {
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient

				// accumulate function values and gradients from each user
				for (int i = 0; i < m_userList.size(); i++) {
					user = (_AdaptStruct) m_userList.get(i);
					fValue += calculateFuncValue(user); // L + R^1(A_i)
					calculateGradients(user);
				}

				if (m_displayLv == 2) {
					System.out.format("Fvalue is %.3f\t", fValue);
					gradientTest();
				} else if (m_displayLv == 1) {
					if (fValue < oldFValue)
						System.out.print("o");
					else
						System.out.print("x");

					if (++displayCount % 100 == 0)
						System.out.println();
				}
				oldFValue = fValue;

				LBFGS.lbfgs(m_ws.length, 6, m_ws, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);// In the training process, A is updated.
			} while (iflag[0] != 0);
			System.out.println();
		} catch (ExceptionWithIflag e) {
			System.err.println("********lbfgs fails here!******");
			e.printStackTrace();
		}

		setPersonalizedModel();
		return oldFValue;
	}
	public void setPersonalizedModel(){
		double[] uWeights; // w_i
		double[] pWeights = new double[m_featureSize+1]; // personalzied weights, u*w_g + w_i
		double[] gWeights = Arrays.copyOfRange(m_ws, m_userList.size()*(m_featureSize+1), m_ws.length);

		int start, end;
		for(_AdaptStruct u: m_userList){
			start = u.getId()*(m_featureSize+1);
			end = (u.getId()+1)*(m_featureSize+1);
			uWeights = Arrays.copyOfRange(m_ws, start, end);
			
			for(int k=0; k<uWeights.length; k++)
				pWeights[k] = uWeights[k] + m_u*gWeights[k];
			u.setPersonalizedModel(pWeights);
		}
	}
}
