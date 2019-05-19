package Classifier.supervised.modelAdaptation;

import java.util.HashMap;

import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Problem;
import structures._Review;
import structures._SparseFeature;


public class MultiTaskSVMWithClusters extends MultiTaskSVM {
	
	int m_clusterNo; // The number of clusters.
	int[] m_userClusterIndex; // The index is user index, the value is corresponding cluster no.
	double m_c; // The coefficient in front of cluster.
	double m_i; // The coefficient in front of individual user.
	
	HashMap<Integer, Integer> m_userIndexClusterIndexMap; // Find each user's corresponding cluster no.
	public MultiTaskSVMWithClusters(int classNo, int featureSize, int clusterNo, int[] userClusterIndex) {
		super(classNo, featureSize);
		m_clusterNo = clusterNo;
		m_userClusterIndex = userClusterIndex;
		m_c = 1;
		m_i = 1;
	}
	public void setAllParams(double g, double c, double i){
		setGlobalParam(g);
		setClusterParam(c);
		setIndividualParam(i);
	}
	public void setGlobalParam(double g){
		m_u = g;
	}
	public void setClusterParam(double c){
		m_c = c;
	}
	public void setIndividualParam(double i){
		m_i = i;
	}
	@Override
	public String toString() {
		return String.format("MT-SVM-Clusters[mu:%.3f,mc:%.3f,mi:%.3f,C:%.3f,clusters:%d,bias:%b]", m_u, m_c, m_i, m_C, m_clusterNo, m_bias);
	}
	
	@Override
	public void setLibProblemDimension(Problem libProblem){
		if (m_bias) {
			libProblem.n = (m_featureSize + 1) * (m_userSize + m_clusterNo + 1); // including bias term; global model + user models
			libProblem.bias = 1;// bias term in liblinear.
		} else {
			libProblem.n = m_featureSize * (m_userSize + m_clusterNo + 1);
			libProblem.bias = -1;// no bias term in liblinear.
		}
	}
	
	//create a training instance of svm with cluster information.
	//for MT-SVM feature vector construction: we put user models in front of global model
	@Override
	public Feature[] createLibLinearFV(_Review r, int userIndex){
		int fIndex, clusterIndex = m_userClusterIndex[userIndex]; 
		double fValue;
		_SparseFeature fv;
		_SparseFeature[] fvs = r.getSparse();
		
		int userOffset, clusterOffset, globalOffset;		
		Feature[] node;//0-th: x//sqrt(u); t-th: x.
		
		if (m_bias) {
			userOffset = (m_featureSize + 1) * userIndex;
			clusterOffset = (m_featureSize + 1) * (m_userSize + clusterIndex);
			globalOffset = (m_featureSize + 1) * (m_userSize + m_clusterNo);
			node = new Feature[(1+fvs.length) * 3]; // It consists of three parts.
		} else {
			userOffset = m_featureSize * userIndex;
			clusterOffset = m_featureSize * (m_userSize + clusterIndex);
			globalOffset = m_featureSize * (m_userSize + m_clusterNo);
			node = new Feature[fvs.length * 3];
		}
		
		for(int i = 0; i < fvs.length; i++){
			fv = fvs[i];
			fIndex = fv.getIndex() + 1;//liblinear's feature index starts from one
			fValue = fv.getValue();
			
			//Construct the user part of the training instance.			
			node[i] = new FeatureNode(userOffset + fIndex, fValue*m_i);
			
			//Construct the cluster and global part of the training instance.
			if (m_bias){
				node[i + fvs.length + 1] = new FeatureNode(clusterOffset + fIndex, m_c==0?0:fValue/m_c); // cluster part
				node[i + 2 * fvs.length + 2] = new FeatureNode(globalOffset + fIndex, m_u==0?0:fValue/m_u); // global part
			} else{
				node[i + fvs.length] = new FeatureNode(clusterOffset + fIndex, m_c==0?0:fValue/m_c); // cluster part
				node[i + 2 * fvs.length] = new FeatureNode(globalOffset + fIndex, m_u==0?0:fValue/m_u); // global part
			}
		}
		
		if (m_bias) {//add the bias term		
			node[fvs.length] = new FeatureNode((m_featureSize + 1) * (userIndex + 1), m_i==0?0:1.0/m_i);//user model's bias
			node[2*fvs.length+1] = new FeatureNode((m_featureSize + 1) * (m_userSize + clusterIndex + 1), m_c==0?0:1.0 / m_c);//cluster model's bias
			node[3*fvs.length+2] = new FeatureNode((m_featureSize + 1) * (m_userSize + m_clusterNo + 1), m_u==0?0:1.0 / m_u);//global model's bias
		}
		return node;
	}
	
	@Override
	protected void setPersonalizedModel() {
		double[] weight = m_libModel.getWeights();//our model always assume the bias term
		int class0 = m_libModel.getLabels()[0];
		double sign = class0 > 0 ? 1 : -1;
		int userOffset = 0, clusterOffset = 0;
		int globalOffset = (m_bias?(m_featureSize+1):m_featureSize)*(m_userSize + m_clusterNo);
		_AdaptStruct user;
		
		for(int u=0; u<m_userList.size(); u++) {
			user = m_userList.get(u);
			userOffset = (m_bias?(m_featureSize + 1):m_featureSize) * u;
			clusterOffset = (m_bias?(m_featureSize+1):m_featureSize)*(m_userSize + m_userClusterIndex[u]);

			if(m_personalized){
				for(int i=0; i<m_featureSize; i++) 
					m_pWeights[i+1] = sign*(weight[globalOffset+i]*m_u + weight[clusterOffset+i]*m_c + weight[userOffset+i]*m_i);
				
				if (m_bias) 
					m_pWeights[0] = sign*(weight[globalOffset+m_featureSize]*m_u + weight[clusterOffset+m_featureSize]*m_c + weight[userOffset+m_featureSize]*m_i);
			
			} else {
				for(int i=0; i<m_featureSize; i++) // no personal model since no adaptation data
					m_pWeights[i+1] = sign*weight[globalOffset+i]*m_u;
				
				if (m_bias)
					m_pWeights[0] = sign*weight[globalOffset+m_featureSize]*m_u;
			}
			
			user.setPersonalizedModel(m_pWeights);//our model always assume the bias term
		}
	}}
