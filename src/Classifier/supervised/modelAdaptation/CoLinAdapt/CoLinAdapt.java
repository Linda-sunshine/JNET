/**
 * 
 */
package Classifier.supervised.modelAdaptation.CoLinAdapt;

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
 * synchronized CoLinAdapt algorithm
 */
public class CoLinAdapt extends LinAdapt {

	double m_eta3; // weight for scaling in R2.
	double m_eta4; // weight for shifting in R2.
	int m_topK;
	SimType m_sType = SimType.ST_BoW;// default neighborhood by BoW
	
	public CoLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_eta3 = 0.5;
		m_eta4 = 0.5;
		m_topK = topK; // when topK<0, we will use a fully connected graph 
		
		// the only possible test modes for CoLinAdapt is batch mode
		m_testmode = TestMode.TM_batch;
	}
	
	@Override
	public String toString() {
		return String.format("CoLinAdapt[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_eta2, m_eta3, m_eta4, m_topK, m_sType);
	}

	public void setR2TradeOffs(double eta3, double eta4) {
		m_eta3 = eta3;
		m_eta4 = eta4;
	}
	
	public void setSimilarityType(SimType sType) {
		m_sType = sType;
	}
	
	@Override
	protected int getVSize() {
		return 2*m_dim*m_userList.size();
	}
	
	void constructUserList(ArrayList<_User> userList) {
		int vSize = 2*m_dim;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, m_dim, i, m_topK));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptStruct.sharedA = new double[getVSize()];
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
		}
	}

	@Override
	public void loadUsers(ArrayList<_User> userList){	
		//step 1: construct the user list structures
		constructUserList(userList);
		
		//step 2: construct neighborhood graph
		constructNeighborhood(m_sType);
	}
	
	//this will be only called once in CoLinAdapt
	@Override
	protected void initLBFGS(){ 
		int vSize = getVSize();
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {		
		double fValue = super.calculateFuncValue(u);
		double R2 = calculateR2(u);
		return fValue + R2;
	}
	
	public double calculateR2(_AdaptStruct u){
		//R2 regularization
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
		double R2 = 0, diffA, diffB;
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			diffA = 0;
			diffB = 0;
			for(int k=0; k<m_dim; k++) {
				diffA += (ui.getScaling(k) - uj.getScaling(k)) * (ui.getScaling(k) - uj.getScaling(k));
				diffB += (ui.getShifting(k) - uj.getShifting(k)) * (ui.getShifting(k) - uj.getShifting(k));
			}
			R2 += nit.m_value * (m_eta3*diffA + m_eta4*diffB);
//			R2 += 0.1 * (m_eta3*diffA + m_eta4*diffB);
//			R2 += (nit.m_value / simSum) * (m_eta3*diffA + m_eta4*diffB);
		}
		return R2;
	}
	@Override
	protected void calculateGradients(_AdaptStruct u){
		super.calculateGradients(u);
		gradientByR2(u);
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR2(_AdaptStruct user){		
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)user, uj;
		int offseti = m_dim*2*ui.getId(), offsetj;
		double coef, dA, dB;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			offsetj = m_dim*2*uj.getId();
			coef = 2 * nit.m_value;
			
			for(int k=0; k<m_dim; k++) {
				dA = coef * m_eta3 * (ui.getScaling(k) - uj.getScaling(k));
				dB = coef * m_eta4 * (ui.getShifting(k) - uj.getShifting(k));
				
				// update ui's gradient
				m_g[offseti + k] += dA;
				m_g[offseti + k + m_dim] += dB;
				
				// update uj's gradient
				m_g[offsetj + k] -= dA;
				m_g[offsetj + k + m_dim] -= dB;
			}			
		}
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, uid;
		double magA = 0, magB = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dim]*m_g[offset+m_dim];
			}
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	protected void initPerIter() {
		Arrays.fill(m_g, 0); // initialize gradient
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int displayCount = 0;
		_LinAdaptStruct user;
		
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				initPerIter();			
				
				// accumulate function values and gradients from each user
				for(int i=0; i<m_userList.size(); i++) {
					user = (_LinAdaptStruct)m_userList.get(i);
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
				
				LBFGS.lbfgs(m_g.length, 5, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}		
		
		setPersonalizedModel();
		return oldFValue;
	}
}
