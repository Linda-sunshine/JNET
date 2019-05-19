package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._SparseFeature;
import structures._User;

public class MTLinAdapt extends CoLinAdapt {

	double[] m_A; // [A_0, A_1, A_2,..A_s]Transformation matrix shared by super user and individual users.

	// feature grouping for super user (this could be different from individual users' feature grouping)
	int m_dimSup;
	int[] m_featureGroupMap4SupUsr; // bias term is at position 0
	double[] m_sWeights; // Weights for the super user.

//	double m_lambda1; // Scaling coefficient for R^1(A_s)
//	double m_lambda2; // Shifting coefficient for R^1(A_s)
	
	boolean m_LNormFlag; // Decide if we will normalize the likelihood.
	int m_lbfgs = 1; // m_lbfgs = 0, fails; m_lbfgs = 1, succeed.
	
	// The constructor only constructs feature group for individual users, not super user.
	public MTLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
			int topK, String globalModel, String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		loadFeatureGroupMap4SupUsr(featureGroup4Sup);

		m_LNormFlag = true;
	}
	
	public void setLNormFlag(boolean b){
		m_LNormFlag = b;
	}	
	
	@Override
	public String toString() {
		return String.format("MT-LinAdapt[dim:%d, supDim:%d, eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f, personalized:%b]", 
				m_dim, m_dimSup, m_eta1, m_eta2, m_eta3, m_eta4, m_personalized);
	}
	
	@Override
	protected int getVSize() {
		return m_userList.size()*m_dim*2 + m_dimSup*2;
	}
	
	// Feature group map for the super user.
	public void loadFeatureGroupMap4SupUsr(String filename){
		// If there is no feature group for the super user.
		if(filename == null){
			m_dimSup = m_featureSize + 1;
			m_featureGroupMap4SupUsr = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<=m_featureSize; i++)
				m_featureGroupMap4SupUsr[i] = i;
			return;
		} else{// If there is feature grouping for the super user, load it.
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
				String[] features = reader.readLine().split(",");//Group information of each feature.
				reader.close();
				
				m_featureGroupMap4SupUsr = new int[features.length + 1]; //One more term for bias, bias->0.
				m_dimSup = 0;
				//Group index starts from 0, so add 1 for it.
				for(int i=0; i<features.length; i++) {
					m_featureGroupMap4SupUsr[i+1] = Integer.valueOf(features[i]) + 1;
					if (m_dimSup < m_featureGroupMap4SupUsr[i+1])
						m_dimSup = m_featureGroupMap4SupUsr[i+1];
				}
				m_dimSup ++;
			} catch(IOException e){
				System.err.format("[Error]Fail to open super user group file %s.\n", filename);
			}
		}
		System.out.format("[Info]Feature group size for super user %d\n", m_dimSup);
	}
	
	@Override
	void constructUserList(ArrayList<_User> userList) {
		super.constructUserList(userList);
		
		m_A = _CoLinAdaptStruct.sharedA;
		
		// Init m_sWeights with global weights;
		m_sWeights = new double[m_featureSize + 1];
		System.arraycopy(m_gWeights, 0, m_sWeights, 0, m_gWeights.length);;
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList){
		int vSize = 2*m_dim;
		constructUserList(userList);
		// Init A_s with [1,1,1,..,0,0,0,...].
		for(int i=m_userList.size()*vSize; i<m_userList.size()*vSize+m_dimSup; i++)
			m_A[i] = 1;
	}
	
	// We can do A_i*A_s*w_g*x at the same time to reduce computation.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		int n = 0, k = 0; // feature index and feature group index
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		double value = ui.getScaling(0)*getSupWeights(0) + ui.getShifting(0);//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (ui.getScaling(k)*getSupWeights(n) + ui.getShifting(k)) * fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	
	//Calculate the function value of the new added instance.
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		double L = calcLogLikelihood(u); //log likelihood.
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;

		if(!m_LNormFlag)
			L *= ui.getAdaptationSize();

		//Add regularization parts.
		double R1 = 0;
		for(int k=0; k<m_dim; k++){
			R1 += m_eta1 * (ui.getScaling(k)-1) * (ui.getScaling(k)-1);//(a[i]-1)^2
			R1 += m_eta2 * ui.getShifting(k) * ui.getShifting(k);//b[i]^2
		}
		return (R1 - L);
	}
	
	@Override
	// Since I cannot access the method in LinAdapt or in RegLR, I Have to rewrite.
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u);
	}
	
	// Calculate the R1 for the super user, As.
	protected double calculateRs(){
		int offset = m_userList.size()*m_dim*2; // Access the As.
		double rs = 0;
		for(int i=0; i < m_dimSup; i++){
			rs += m_eta3 * (m_A[offset + i] - 1) * (m_A[offset + i] - 1); // Get scaling of super user.
			rs += m_eta4 * m_A[offset + i + m_dimSup] * m_A[offset + i + m_dimSup]; // Get shifting of super user.
		}
		return rs;
	}
	
	// Gradients for the gs.
	protected void gradientByRs(){
		int offset = m_userList.size() * m_dim * 2;
		for(int i=0; i < m_dimSup; i++){
			m_g[offset + i] += 2 * m_eta3 * (m_A[offset + i] - 1);
			m_g[offset + i + m_dimSup] += 2 * m_eta4 * m_A[offset + i + m_dimSup];
		}
	}
	
	// Gradients from loglikelihood, contributes to both individual user's gradients and super user's gradients.
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		int n, k, s; // feature index and feature group index		
		int offset = 2*m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int offsetSup = 2*m_dim*m_userList.size();
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);

		// Bias term for individual user.
		m_g[offset] -= delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		m_g[offset + m_dim] -= delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= delta*ui.getScaling(0)*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		m_g[offsetSup + m_dimSup] -= delta*ui.getScaling(0); //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			m_g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
			
			s = m_featureGroupMap4SupUsr[n];
			m_g[offsetSup + s] -= delta*ui.getScaling(k)*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			m_g[offsetSup + m_dimSup + s] -= delta*ui.getScaling(k)*fv.getValue(); // a_i*x_di
		}
	}
	
	//this is batch training in each individual user
	@Override
	public double train() {
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue, oldFValue = Double.MAX_VALUE;
		int vSize = getVSize(), displayCount = 0;
		_LinAdaptStruct user;

		initLBFGS();
		init();
		try {
			do {
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient

				// accumulate function values and gradients from each user
				for (int i = 0; i < m_userList.size(); i++) {
					user = (_LinAdaptStruct) m_userList.get(i);
					fValue += calculateFuncValue(user); // L + R^1(A_i)
					calculateGradients(user);
				}
				// The contribution from R^1(A_s) to both function value and gradients.
				fValue += calculateRs(); // + R^1(A_s)
				gradientByRs(); // Gradient from R^1(A_s)

				if (m_displayLv == 2) {
					System.out.format("Fvalue is %.3f", fValue);

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
				LBFGS.lbfgs(vSize, 6, m_A, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);// In the training process, A is updated.
			} while (iflag[0] != 0);
			System.out.println();
		} catch (ExceptionWithIflag e) {
			System.err.println("********lbfgs fails here!******");
			e.printStackTrace();
			m_lbfgs = 0;
		}

		setPersonalizedModel();
		return oldFValue;
	}
	
	@Override
	// In the algorithm, each individual user's model is A_i*A_s*w_g.
	protected void setPersonalizedModel() {
		int gid;
		_CoLinAdaptStruct ui;
		
		//get the model weight for super user
		for(int n=0; n<=m_featureSize; n++)
			m_sWeights[n] = getSupWeights(n);
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			
			if(m_personalized){
				//set the other features
				for(int n=0; n<=m_featureSize; n++) {
					gid = m_featureGroupMap[n];
					m_pWeights[n] = ui.getScaling(gid) * m_sWeights[n] + ui.getShifting(gid);
				}
				ui.setPersonalizedModel(m_pWeights);
			} else// Set super user == general user.
				ui.setPersonalizedModel(m_sWeights);
		}
	}
	
	// w_s = A_s * w_g
	public double getSupWeights(int index){
		int gid = m_featureGroupMap4SupUsr[index], offsetSup = m_userList.size() * 2 * m_dim;
		return m_A[offsetSup + gid] * m_gWeights[index] + m_A[offsetSup + gid + m_dimSup];
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, offsetSup;
		double magA = 0, magB = 0;
		for(int n=0; n<m_userList.size(); n++) {
			offset = n*vSize;
			for(int i=0; i<m_dim; i++){
				magA += m_g[offset+i]*m_g[offset+i];
				magB += m_g[offset+m_dim+i]*m_g[offset+m_dim+i];
			}
		}

		double magASup = 0, magBSup = 0;
		offsetSup = vSize * m_userList.size();
		for(int i=0; i<m_dimSup; i++){
			magASup += m_g[offsetSup+i] * m_g[offsetSup+i];
			magBSup += m_g[offsetSup+m_dimSup+i] * m_g[offsetSup + m_dimSup+i];
		}
		
		if (m_displayLv==2)
			System.out.format("\tuser(%.4f,%.4f), super user(%.4f,%.4f)\n", magA, magB, magASup, magBSup);
		return magA + magB;
	}
	
	public double[] getSupWeights(){
		return m_sWeights;
	}
	
	public double[] getGlobalWeights(){
		return m_gWeights;
	}
	
	public int getLBFGSFlag(){
		return m_lbfgs;
	}
	
	// Print out super user's weights.
	public void saveSupModel(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			writer.write(m_sWeights[0]+"\n");
			for(int i=1; i<m_sWeights.length; i++){
				writer.write(m_sWeights[i]+"\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
