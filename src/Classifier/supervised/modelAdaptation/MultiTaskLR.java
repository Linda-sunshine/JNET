package Classifier.supervised.modelAdaptation;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc.rType;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class MultiTaskLR extends ModelAdaptation {

	double[] m_beta; // store all users' weights
	double[] m_g, m_diag; // variables used in lbfgs
	double m_u; // w_user = w_g + u * w_i
	double m_lambda;
	
	public MultiTaskLR(int classNo, int featureSize) {
		super(classNo, featureSize);
		m_u = 1;
		m_lambda = 0.1;
		m_testmode = TestMode.TM_batch;

	}
	
	public void setLambda(double l){
		m_lambda = l;
	}
	@Override
	public String toString() {
		return String.format("MT-LR[mu:%.3f,lambda:%.3f]", m_u, m_lambda);
	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList) 
			m_userList.add(new _AdaptStruct(user));
		loadGlobalModel("./data/mtsvm_global.txt");
	}
	
	//Load global model from file.
	public void loadGlobalModel(String filename){
		if (filename==null)
			return;
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			ArrayList<Double> weights = new ArrayList<Double>();
			while((line=reader.readLine()) != null)
				weights.add(Double.parseDouble(line));
			
			reader.close();
			if(weights.size() == m_featureSize + 1){
				m_gWeights = new double[weights.size()];
				for(int i=0; i<weights.size(); i++)
					m_gWeights[i] = weights.get(i);
			} else
				System.out.println("Wrong dimension of global weights!");
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	@Override
	protected void init() {
		m_beta = new double[(m_featureSize+1)*m_userList.size()];
		m_g = new double[m_beta.length];
		m_diag = new double[m_g.length];
	}
	
	@Override
	public double train() {
		init();
		
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue = 0;
		int fSize = m_beta.length;
		
		try{
			do {
				fValue = calcFuncGradient();
				System.out.println(fValue);
				LBFGS.lbfgs(fSize, 6, m_beta, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
		setPersonalizedModel();
		return fValue;
	}
	
	//This function is used to calculate the value and gradient with the new beta.
	protected double calcFuncGradient() {		
		double gValue = 0, fValue = 0;
		
		// Add the L2 regularization.
		double L2 = 0, b;
		for(int i = 0; i < m_beta.length; i++) {
			b = m_beta[i];
			m_g[i] = 2 * m_lambda * b;
			L2 += b * b;
		}
		
		//The computation complexity is n*classNo.
		int Yi;
		_SparseFeature[] fv;
		double wx = 0, p1 = 0;
		_AdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = m_userList.get(i);
			for(_Review r: user.getReviews()){
				if(r.getType() == rType.TEST){
					Yi = r.getYLabel();
					fv = r.getSparse();
					
					//compute P(y=1|x)
					wx = dotProduct(m_beta, fv, i*(m_featureSize+1));
					p1 = 1/(1 + Math.exp(-wx));
					if(Yi == 1){
						if(p1 > 0.0)
							fValue += Math.log(p1);
						else
							fValue -= Utils.MAX_VALUE;
					} else{
						if(p1 < 1.0)
							fValue += Math.log(1-p1);
						else
							fValue -= Utils.MAX_VALUE;
					}
					//fValue += Yi == 1 ? Math.log(p1) : Math.log(1-p1);
					//System.out.println(String.format("wx:%.5f,p1:%.5f,logP:%.5f", wx, p1, Yi==1?Math.log(p1):Math.log(1-p1)));
					gValue = Yi == 1 ? (p1-1) : p1;
					gValue *= m_u;
					int offset = i * (m_featureSize + 1);
					m_g[offset] += gValue;
					//(y-p1)*x
					for(_SparseFeature sf: fv)
						m_g[offset + sf.getIndex() + 1] += gValue * sf.getValue();
				}
			}
		}
		return m_lambda*L2 - fValue;
	}
		
		
	//The function defines the dot product of beta and sparse Vector of a document.
	public double dotProduct(double[] beta, _SparseFeature[] sf, int offset){
		double sum = m_gWeights[0] + m_u * beta[offset];
		for(int i = 0; i < sf.length; i++){
			int index = sf[i].getIndex() + offset + 1;
			sum += (m_gWeights[sf[i].getIndex() + 1] + m_u * beta[index]) * sf[i].getValue();
		}
		return sum;
	}
	@Override
	protected void setPersonalizedModel() {
		_AdaptStruct user;
		m_pWeights = new double[m_gWeights.length];
		int offset = 0;
		for(int i=0; i<m_userList.size(); i++){
			user = m_userList.get(i);
//			pWeights = user.getPWeights();
			offset = (m_featureSize + 1) *i;
			for(int v=0; v<m_featureSize+1; v++){
				m_pWeights[v] = m_gWeights[v] + m_u * m_beta[offset+v];
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
}
