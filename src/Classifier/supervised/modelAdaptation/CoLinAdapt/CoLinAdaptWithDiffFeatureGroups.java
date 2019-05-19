package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._RankItem;
import structures._Review;
import structures._Doc.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class CoLinAdaptWithDiffFeatureGroups extends CoLinAdapt{
	
	int m_dimA;
	int m_dimB;
	int[] m_featureGroupMapB; // bias term is at position 0
	double[] m_cache; // Used to store posterior, p(y|x)
	double m_g0 = 1, m_g1 = 1;
	
	public CoLinAdaptWithDiffFeatureGroups(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMapA, String featureGroupMapB) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMapA);
		
		m_dimA = m_dim;
		loadFeatureGroupMapB(featureGroupMapB); // Load the feature group map for the other class.
		m_cache = new double[m_classNo];
	}
	
	public void setGCoefficients(double a, double b){
		m_g0 = a;
		m_g1 = b;
	}
	
	@Override
	public String toString() {
		return String.format("CoLinAdaptWithDiffFvGroups[dimA:%d,dimB:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,k:%d,NB:%s]", m_dim, m_dimB, m_eta1, m_eta2, m_eta3, m_eta4, m_topK, m_sType);
	}
	
	int getASize(){
		return m_dimA*2*m_userList.size();
	}
	
	int getBSize(){
		return m_dimB*2*m_userList.size();
	}
	
	@Override
	protected int getVSize() {
		return getASize() + getBSize();
	} 
	
	@Override
	void constructUserList(ArrayList<_User> userList) {
		int ASize = 2*m_dimA;
		int BSize = 2*m_dimB;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptDiffFvGroupsStruct(user, m_dimA, i, m_topK, m_dimB));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptDiffFvGroupsStruct.sharedA = new double[getASize()];
		_CoLinAdaptDiffFvGroupsStruct.sharedB = new double[getBSize()];
		
		//step 2: copy each user's A and B to shared A and B in _CoLinAdaptStruct		
		_CoLinAdaptDiffFvGroupsStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptDiffFvGroupsStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptDiffFvGroupsStruct.sharedA, ASize*i, ASize);
			System.arraycopy(user.m_B, 0, _CoLinAdaptDiffFvGroupsStruct.sharedB, BSize*i, BSize);
		}
	}
	
	// Feature group map for the super user.
	void loadFeatureGroupMapB(String filename){
		// If there is no feature group for the super user.
		if(filename == null){
			m_dimB = m_featureSize + 1;
			m_featureGroupMapB = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<=m_featureSize; i++)
				m_featureGroupMapB[i] = i;
			return;
		} else{// If there is feature grouping for the super user, load it.
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
				String[] features = reader.readLine().split(",");//Group information of each feature.
				reader.close();
				
				m_featureGroupMapB = new int[features.length + 1]; //One more term for bias, bias->0.
				m_dimB = 0;
				//Group index starts from 0, so add 1 for it.
				for(int i=0; i<features.length; i++) {
					m_featureGroupMapB[i+1] = Integer.valueOf(features[i]) + 1;
					if (m_dimB < m_featureGroupMapB[i+1])
						m_dimB = m_featureGroupMapB[i+1];
				}
				m_dimB ++;
			} catch(IOException e){
				System.err.format("[Error]Fail to open super user group file %s.\n", filename);
			}
		}
		System.out.format("[Info]Feature group size for feature group B %d\n", m_dimB);
	}

	// There is still issue in calculating R2 since we don't know which set to use for a user.
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {		
		double fValue = super.calculateFuncValue(u), R1 = 0, R2 = 0, diffA, diffB;
		_CoLinAdaptDiffFvGroupsStruct ui = (_CoLinAdaptDiffFvGroupsStruct)u, uj;
		
		//Add R1 for another class.
		for(int i=0; i<m_dimB; i++){
			R1 += m_eta1 * (ui.getScalingB(i)-1) * (ui.getScalingB(i)-1);//(a[i]-1)^2
			R1 += m_eta2 * ui.getShiftingB(i) * ui.getShiftingB(i);//b[i]^2
		}
				
		//R2 regularization
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptDiffFvGroupsStruct)m_userList.get(nit.m_index);
			diffA = 0;
			diffB = 0;
			// We also need to sum over the other set of parameters.
			for(int k=0; k<m_dimB; k++){
				diffA += (ui.getScalingB(k) - uj.getScalingB(k)) * (ui.getScalingB(k) - uj.getScalingB(k));
				diffB += (ui.getShiftingB(k) - uj.getShiftingB(k)) * (ui.getShiftingB(k) - uj.getShiftingB(k));
			}
			R2 += nit.m_value * (m_eta3*diffA + m_eta4*diffB);
		}
		return fValue + R1 + R2;
	}
	
	//Calculate the function value of the new added instance.
	@Override
	protected double calcLogLikelihood(_AdaptStruct user){
		double L = 0; //log likelihood.
		double Pi = 0;
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue; // only touch the adaptation data
			
			calcPosterior(review.getSparse(), user);
			Pi = m_cache[review.getYLabel()];
			
			if (Pi>0.0)
				L += Math.log(Pi);					
			else
				L -= Utils.MAX_VALUE;
		}
		return L/getAdaptationSize(user);
	}

	public void calcPosterior(_SparseFeature[] fvs, _AdaptStruct u){

		// We want get p(y=0|x) and p(y=1|x) based on ylabel.
		_CoLinAdaptDiffFvGroupsStruct user = (_CoLinAdaptDiffFvGroupsStruct)u;
		double exp0 = 0, exp1 = 0;
		int n = 0, k = 0; // feature index and feature group index
		
		// w0*x
		exp0 = user.getScaling(0)*m_gWeights[0]*m_g0 + user.getShifting(0);// Bias term: w0*a0+b0.
		exp1 = user.getScalingB(0)*m_gWeights[0]*m_g1 + user.getShiftingB(0); // Bias term.
		
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			
			k = m_featureGroupMap[n];
			exp0 += (user.getScaling(k)*m_gWeights[n]*m_g0 + user.getShifting(k)) * fv.getValue();
			
			k = m_featureGroupMapB[n];
			exp1 += (user.getScalingB(k)*m_gWeights[n]*m_g1 + user.getShiftingB(k)) * fv.getValue();
		}
		
		exp0 = Math.exp(exp0);
		exp1 = Math.exp(exp1);
		
		m_cache[0] = exp0/(exp0+exp1);
		m_cache[1] = exp1/(exp0+exp1);
	}
	
	//shared gradient calculation by batch and online updating
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptDiffFvGroupsStruct user = (_CoLinAdaptDiffFvGroupsStruct)u;
		
		int n, k; // feature index and feature group index		
		int offsetA = 2*m_dimA*user.getId(), offsetB = getASize() + 2*m_dimB*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double deltaA, deltaB;
		
		if(review.getYLabel() == 0) {
			deltaA = 1.0 - m_cache[0];
			deltaB = -m_cache[1];
		} else {
			deltaA = -m_cache[0];
			deltaB = 1.0 - m_cache[1];
		}
		
		if (m_LNormFlag) {
			deltaA /= getAdaptationSize(user);
			deltaB /= getAdaptationSize(user);
		}
		
		//Bias term.
		m_g[offsetA] -= weight*deltaA*m_gWeights[0]*m_g0; //a[0] = w0*x0; x0=1
		m_g[offsetA + m_dimA] -= weight*deltaA;//b[0]

		m_g[offsetB] -= weight*deltaB*m_gWeights[0]*m_g1; // a[0]
		m_g[offsetB + m_dimB] -= weight*deltaB; // b[0]
		
		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			
			k = m_featureGroupMap[n];
			m_g[offsetA + k] -= weight * deltaA * m_gWeights[n]*m_g0 * fv.getValue();
			m_g[offsetA + m_dimA + k] -= weight * deltaA * fv.getValue();  
			
			k = m_featureGroupMapB[n];
			m_g[offsetB + k] -= weight * deltaB * m_gWeights[n]*m_g1 * fv.getValue();
			m_g[offsetB + m_dimB + k] -= weight * deltaB * fv.getValue();  
		}
	}
	
	@Override
	protected void gradientByR1(_AdaptStruct u){
		super.gradientByR1(u);
		_CoLinAdaptDiffFvGroupsStruct user = (_CoLinAdaptDiffFvGroupsStruct)u;
		int offset = getASize() + 2*m_dimB*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt

		//R1 regularization part for another class.
		for(int k=0; k<m_dimB; k++){
			m_g[offset + k] += 2 * m_eta1 * (user.getScalingB(k)-1);// add 2*eta1*(a_k-1)
			m_g[offset + k + m_dimB] += 2 * m_eta2 * user.getShiftingB(k); // add 2*eta2*b_k
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR2(_AdaptStruct user){
		// Part 1, gradients from class 0.
		super.gradientByR2(user);
		
		_CoLinAdaptDiffFvGroupsStruct ui = (_CoLinAdaptDiffFvGroupsStruct)user, uj;
		
		int Asize = getASize(), offseti = Asize + m_dimB*2*ui.getId(), offsetj;
		
		double coef, dA, dB;
		// Part 2, gradients from class 1.
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptDiffFvGroupsStruct)m_userList.get(nit.m_index);
			offsetj = Asize + m_dimB*2*uj.getId();
			coef = 2 * nit.m_value;
			
			for(int k=0; k<m_dimB; k++) {
				dA = coef * m_eta3 * (ui.getScalingB(k) - uj.getScalingB(k));
				dB = coef * m_eta4 * (ui.getShiftingB(k) - uj.getShiftingB(k));
				
				// update ui's gradient
				m_g[offseti + k] += dA;
				m_g[offseti + k + m_dimB] += dB;
			
				// update uj's gradient
				m_g[offsetj + k] -= dA;
				m_g[offsetj + k + m_dimB] -= dB;
			}			
		}
	}
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = getVSize(), displayCount = 0, lengthA = getASize(), lengthB = getBSize();
		_CoLinAdaptDiffFvGroupsStruct user;
			
		initLBFGS();
		init();
		double[] sharedAB = _CoLinAdaptDiffFvGroupsStruct.getSharedAB();
		
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
					
				// accumulate function values and gradients from each user
				for(int i=0; i<m_userList.size(); i++) {
					user = (_CoLinAdaptDiffFvGroupsStruct)m_userList.get(i);
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
					
				if (m_displayLv==2) {
					gradientTest();
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
				
				LBFGS.lbfgs(vSize, 5, sharedAB, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				// We need to update the learned A and B.
				System.arraycopy(sharedAB, 0, _CoLinAdaptDiffFvGroupsStruct.sharedA, 0, lengthA);
				System.arraycopy(sharedAB, lengthA, _CoLinAdaptDiffFvGroupsStruct.sharedB, 0, lengthB);
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}		
		
		setPersonalizedModel();
		return oldFValue;
	}
	
	@Override
	public void setPersonalizedModel(){
		int gid;
		_CoLinAdaptDiffFvGroupsStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptDiffFvGroupsStruct)m_userList.get(i);
			
			//set bias term
			m_pWeights[0] = user.getScalingB(0) * m_gWeights[0]*m_g1 + user.getShiftingB(0);
			m_pWeights[0] -= user.getScaling(0) * m_gWeights[0]*m_g0 + user.getShifting(0);

			//set the other features
			for(int n=0; n<m_featureSize; n++) {
				gid = m_featureGroupMapB[1+n];
				m_pWeights[1+n] = user.getScalingB(gid) * m_gWeights[1+n]*m_g1 + user.getShiftingB(gid);
				
				gid = m_featureGroupMap[1+n];
				m_pWeights[1+n] -= user.getScaling(gid) * m_gWeights[1+n]*m_g0 + user.getShifting(gid);
			}
			user.setPersonalizedModel(m_pWeights);			
		}
	}
	
	//will be used in asynchornized model update
	@Override
	protected int predict(_Doc review, _AdaptStruct user) {
		if (review==null)
			return -1;
		else{
			_SparseFeature[] fvs = review.getSparse();
			calcPosterior(fvs, user);
			return Utils.argmax(m_cache);
		}
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dimA, offset, uid, base;
		double magA = 0, magB = 0, magC = 0, magD = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dimA; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dimA]*m_g[offset+m_dimA];
			}
		}
		
		vSize = 2*m_dimB;
		base = getASize();
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dimB; i++){
				offset = base + uid + i;
				magC += m_g[offset]*m_g[offset];
				magD += m_g[offset+m_dimB]*m_g[offset+m_dimB];
			}
		}
		
		if (m_displayLv==2)
			System.out.format("Gradients a:%.5f,b:%.5f,c:%.5f,d:%.5f\n", magA, magB, magC, magD);
		return magA + magB + magC + magD;
	}
}
