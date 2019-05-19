package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class LinAdapt extends RegLR {
	protected int m_dim;//The number of feature groups k, so the total number of dimensions of weights is 2(k+1).	
	protected int[] m_featureGroupMap; // bias term is at position 0
	
	//Trade-off parameters	
	protected double m_eta2; // weight for shifting in R2.	
	
	public LinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap){
		super(classNo, featureSize, featureMap, globalModel);
		m_userList = null;
		
		loadFeatureGroupMap(featureGroupMap);
		
		// default value of trade-off parameters
		m_eta1 = 0.5;
		m_eta2 = 0.5;
		
		// the only test mode for LinAdapt is batch
		m_testmode = TestMode.TM_batch;
	}  
	
	public LinAdapt(int classNo, int featureSize, String globalModel, String featureGroupMap){
		super(classNo, featureSize, globalModel);
		m_userList = null;
		
		loadFeatureGroupMap(featureGroupMap);
		
		// default value of trade-off parameters
		m_eta1 = 0.5;
		m_eta2 = 0.5;
		
		// the only test mode for LinAdapt is batch
		m_testmode = TestMode.TM_batch;
	}  
	
	@Override
	public String toString() {
		return String.format("LinAdapt[dim:%d,eta1:%.3f,eta2:%.3f]", m_dim, m_eta1, m_eta2);
	}
	
	public void setR1TradeOffs(double eta1, double eta2) {
		m_eta1 = eta1;
		m_eta2 = eta2;
	}
	
	/***When we do feature selection, we will group features and store them in file. 
	 * The index is the index of features and the corresponding number is the group index number.***/
	public void loadFeatureGroupMap(String filename){
		if(filename == null){
			m_dim = m_featureSize + 1;
			m_featureGroupMap = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<=m_featureSize; i++)
				m_featureGroupMap[i] = i;
			return;
		} else {		
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
				String[] groups = reader.readLine().split(",");//Group information of each feature.
				reader.close();
				
				m_featureGroupMap = new int[groups.length + 1]; //One more term for bias, 0->0.
				m_dim = 0;
				//Group index starts from 0, so add 1 for it.
				for(int i=0; i<groups.length; i++) {
					m_featureGroupMap[i+1] = Integer.valueOf(groups[i]) + 1;
					if (m_dim < m_featureGroupMap[i+1])
						m_dim = m_featureGroupMap[i+1];
				}
				m_dim ++;
				
				System.out.format("[Info]Feature group size %d\n", m_dim);
			} catch(IOException e){
				System.err.format("[Error]Fail to open file %s.\n", filename);
			}
		}
	}
	
	//Initialize the weights of the transformation matrix.
	@Override
	public void loadUsers(ArrayList<_User> userList){
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _LinAdaptStruct(user, m_dim));
		m_pWeights = new double[m_gWeights.length];
	}
	
	protected int getVSize() {
		return m_dim*2;
	}
	
	//this function will be repeatedly called in LinAdapt
	@Override
	protected void initLBFGS(){
		if(m_g == null)
			m_g = new double[getVSize()];
		if(m_diag == null)
			m_diag = new double[getVSize()];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
	}
	
	protected double linearFunc(_SparseFeature[] fvs, _AdaptStruct u) {
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		double value = user.getScaling(0)*m_gWeights[0] + user.getShifting(0);//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (user.getScaling(k)*m_gWeights[n] + user.getShifting(k)) * fv.getValue();
		}
		return value;
	}

	// We can do A*w*x at the same time to reduce computation.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){		
		return Utils.logistic(linearFunc(fvs, u));
	}
	
	//Calculate the function value of the new added instance.
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
	
	//shared gradient calculation by batch and online updating
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		
		int n, k; // feature index and feature group index		
		int offset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), user));
		if (m_LNormFlag)
			delta /= getAdaptationSize(user);

		//Bias term.
		m_g[offset] -= delta*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[offset + m_dim] -= delta;//b[0]

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= delta * m_gWeights[n] * fv.getValue();
			m_g[offset + m_dim + k] -= delta * fv.getValue();  
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR1(_AdaptStruct u){
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		int offset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		//R1 regularization part
		for(int k=0; k<m_dim; k++){
			m_g[offset + k] += 2 * m_eta1 * (user.getScaling(k)-1);// add 2*eta1*(a_k-1)
			m_g[offset + k + m_dim] += 2 * m_eta2 * user.getShifting(k); // add 2*eta2*b_k
		}
	}
	
	@Override
	protected double gradientTest() {
		double magA = 0, magB = 0 ;
		for(int i=0; i<m_dim; i++){
			magA += m_g[i]*m_g[i];
			magB += m_g[i+m_dim]*m_g[i+m_dim];
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	@Override
	protected void setPersonalizedModel() {
		int gid;
		_LinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_LinAdaptStruct)m_userList.get(i);
			
			if (m_personalized) {
				//set bias term
				m_pWeights[0] = user.getScaling(0) * m_gWeights[0] + user.getShifting(0);
				
				//set the other features
				for(int n=0; n<m_featureSize; n++) {
					gid = m_featureGroupMap[1+n];
					m_pWeights[1+n] = user.getScaling(gid) * m_gWeights[1+n] + user.getShifting(gid);
				}			
				
				user.setPersonalizedModel(m_pWeights);
			} else //otherwise, we will directly use the global model
				user.setPersonalizedModel(m_gWeights);
		}
	}
}
