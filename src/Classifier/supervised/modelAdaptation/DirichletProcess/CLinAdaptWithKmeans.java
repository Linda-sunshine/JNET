package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._SparseFeature;
import structures._User;

/***
 * In this class, we would like to incorporate the cluster information 
 * to see how it influences the final performance. 
 * @author lin
 */
public class CLinAdaptWithKmeans extends LinAdapt{
	int m_clusterSize; // The cluster number.
	// Parameters for different parts.
	double m_u = 1; // global parts.
	double m_c = 1; // cluster parts.
	double m_i = 1; // individual parts.
	
	double m_eta3 = 0.1, m_eta4 = 0.1;// coefficients for cluster and global parts.
	int[] m_userClusterIndex; // The index is user index, the value is corresponding cluster no.

	public CLinAdaptWithKmeans(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, int clusterNo, int[] userClusterIndex) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_clusterSize = clusterNo;
		m_userClusterIndex = userClusterIndex;
	}
	
	public void setParameters(double i, double c, double g){
		m_i = i;
		m_c = c;
		m_u = g;
	}
	
	public void setRcRgTradeOffs(double eta3, double eta4) {
		m_eta3 = eta3;
		m_eta4 = eta4;
	}
	
	//Initialize the weights of the transformation matrix.
	@Override
	public void loadUsers(ArrayList<_User> userList){
		int totalUserSize = userList.size();
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CLinAdaptStruct(user, m_dim, i, totalUserSize, m_clusterSize));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		// step1: init the shared A: individual + cluster + global
		_CLinAdaptStruct.sharedA = new double[getVSize()];
		for(int i=0; i<m_userList.size()+m_clusterSize; i++){
			for(int j=0; j<m_dim; j++){
				_CLinAdaptStruct.sharedA[i*m_dim*2+j] = 1;
			}
		}
	}

	@Override
	protected int getVSize() {
		return m_dim*2*(m_userList.size() + m_clusterSize + 1);
	}
	
	@Override
	protected double linearFunc(_SparseFeature[] fvs, _AdaptStruct u) {
		_CLinAdaptStruct user = (_CLinAdaptStruct)u;
		int clusterIndex = m_userClusterIndex[user.getId()];
		double scaling, shifting;
		scaling = m_u*user.getGlobalScaling(0) + m_c*user.getClusterScaling(clusterIndex, 0) + m_i*user.getScaling(0);
		shifting = m_u*user.getGlobalShifting(0) + m_c*user.getClusterShifting(clusterIndex, 0) + m_i*user.getShifting(0);
		double value = scaling*m_gWeights[0] + shifting;//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			scaling = m_u*user.getGlobalScaling(k) + m_c*user.getClusterScaling(clusterIndex, k) + m_i*user.getScaling(k);
			shifting = m_u*user.getGlobalShifting(k) + m_c*user.getClusterShifting(clusterIndex, k) + m_i*user.getShifting(k);
			value += (scaling*m_gWeights[n] + shifting) * fv.getValue();
		}
		return value;
	}	
//	@Override
//	protected double calculateFuncValue(_AdaptStruct u){
//		return  super.calculateFuncValue(u);
//	}
	
	protected double calculateRcRg(){
		double RcRg = 0;
		int offset = 0;
		// Add regularization parts for clusters.
		for(int c=0; c<m_clusterSize; c++){
			offset = m_dim*2*(m_userList.size()+c);
			for(int k=0; k<m_dim; k++){
				RcRg += m_eta3*(_CLinAdaptStruct.sharedA[offset+k]-1)*(_CLinAdaptStruct.sharedA[offset+k]-1);//(a[i]-1)^2
				RcRg += m_eta4 *_CLinAdaptStruct.sharedA[offset+k+m_dim]*_CLinAdaptStruct.sharedA[offset+k+m_dim];//b[i]^2
			}
		}
		// Add regularization for global parts.
		offset = m_dim*2*(m_userList.size()+m_clusterSize);
		for(int k=0; k<m_dim; k++){
			RcRg += m_eta3*(_CLinAdaptStruct.sharedA[offset+k]-1)*(_CLinAdaptStruct.sharedA[offset+k]-1);//(a[i]-1)^2
			RcRg += m_eta4 *_CLinAdaptStruct.sharedA[offset+k+m_dim]*_CLinAdaptStruct.sharedA[offset+k+m_dim];//b[i]^2
		}		
		return RcRg;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CLinAdaptStruct user = (_CLinAdaptStruct)u;

		int n, k; // feature index and feature group index		
		int clusterIndex = m_userClusterIndex[user.getId()];
		int iOffset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int cOffset = 2*m_dim*(m_userList.size() + clusterIndex);
		int gOffset = 2*m_dim*(m_userList.size() + m_clusterSize);
		double delta = (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term for individual part.
		m_g[iOffset] -= weight*delta*m_i*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[iOffset + m_dim] -= weight*delta*m_i;//b[0]

		//Bias term for cluster part.
		m_g[cOffset] -= weight*delta*m_c*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[cOffset + m_dim] -= weight*delta*m_c;//b[0]
		
		//Bias term for global part.
		m_g[gOffset] -= weight*delta*m_u*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[gOffset + m_dim] -= weight*delta*m_u;//b[0]
				
		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			
			//Individual part.
			m_g[iOffset + k] -= weight*delta*m_i*m_gWeights[n]*fv.getValue();
			m_g[iOffset + m_dim + k] -= weight*delta*m_i*fv.getValue(); 
			
			//Cluster part.
			m_g[cOffset + k] -= weight*delta*m_c*m_gWeights[n]*fv.getValue();
			m_g[cOffset + m_dim + k] -= weight*delta*m_c*fv.getValue(); 
			
			//Global part.
			m_g[gOffset + k] -= weight*delta*m_u*m_gWeights[n]*fv.getValue();
			m_g[gOffset + m_dim + k] -= weight*delta*m_u*fv.getValue(); 
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
	
	// Gradients brought by regularization of clusters.
	public void gradientByRcRg(){
		int offset = 0;
		for(int c=0; c<m_clusterSize; c++){
			offset = m_dim*2*(m_userList.size()+c);
			for(int k=0; k<m_dim; k++){
				m_g[offset+k] += 2*m_eta3*(_CLinAdaptStruct.sharedA[offset+k]-1);//2*(a[i]-1)
				m_g[offset+k+m_dim] += 2*m_eta4 *_CLinAdaptStruct.sharedA[offset+k+m_dim];//2*b[i]
			}
		}
		offset = m_dim*2*(m_userList.size()+m_clusterSize);
		for(int k=0; k<m_dim; k++){
			m_g[offset+k] += 2*m_eta3*(_CLinAdaptStruct.sharedA[offset+k]-1);//2*(a[i]-1)
			m_g[offset+k+m_dim] += 2*m_eta4 *_CLinAdaptStruct.sharedA[offset+k+m_dim];//2*b[i]
		}
	}
	
	protected void initPerIter() {
		Arrays.fill(m_g, 0); // initialize gradient
	}
	
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0, oldFValue = Double.MAX_VALUE, totalFvalue = 0;
		int displayCount = 0;
		_LinAdaptStruct user;

		init();
		initLBFGS();

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
				fValue += calculateRcRg();
				gradientByRcRg();
					
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
				LBFGS.lbfgs(m_g.length, 6, _CLinAdaptStruct.sharedA, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}
		setPersonalizedModel();
		return totalFvalue;
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, iOffset, cOffset;
		double magA = 0, magB = 0, magC = 0, magD = 0;
		// gradients for the individual parts.
		for(int n=0; n<m_userList.size(); n++) {
			for(int i=0; i<m_dim; i++){
				iOffset = n*vSize + i;
				magA += m_g[iOffset]*m_g[iOffset];
				magB += m_g[iOffset+m_dim]*m_g[iOffset+m_dim];
			}
		}
		// gradients for the cluster part.
		for(int c=0; c<m_clusterSize; c++){
			for(int i=0; i<m_dim; i++){
				cOffset = (c + m_userList.size())*vSize + i;
				magC += m_g[cOffset]*m_g[cOffset];
				magD += m_g[cOffset+m_dim]*m_g[cOffset+m_dim];
			}
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for total:%.5f,a:%.5f,b:%.5f,c:%.5f,d:%.5f\n", magA+magB+magC+magD, magA, magB, magC, magD);
		return magA + magB;
	}
	
	@Override
	protected void setPersonalizedModel() {
		int gid;
		_CLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CLinAdaptStruct)m_userList.get(i);
			
			int clusterIndex = m_userClusterIndex[user.getId()];
			double scaling, shifting;
			// Bias term.
			scaling = m_u*user.getGlobalScaling(0) + m_c*user.getClusterScaling(clusterIndex, 0) + m_i*user.getScaling(0);
			shifting = m_u*user.getGlobalShifting(0) + m_c*user.getClusterShifting(clusterIndex, 0) + m_i*user.getShifting(0);
			m_pWeights[0] = scaling*m_gWeights[0] + shifting;//Bias term: w0*a0+b0.
			
			for(int n=0; n<m_featureSize; n++){
				gid = m_featureGroupMap[1+n];
				scaling = m_u*user.getGlobalScaling(gid) + m_c*user.getClusterScaling(clusterIndex, gid) + m_i*user.getScaling(gid);
				shifting = m_u*user.getGlobalShifting(gid) + m_c*user.getClusterShifting(clusterIndex, gid) + m_i*user.getShifting(gid);
				m_pWeights[1+n] = scaling* m_gWeights[1+n] + shifting;
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
	
	@Override
	public String toString() {
		return String.format("CLinAdaptWithKmeans[dim:%d,i:%.2f,c:%.2f,g:%.2f,kmeans:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.4f]",m_dim,m_i,m_c,m_u,m_clusterSize, m_eta1, m_eta2,m_eta3,m_eta4);
	}
}
