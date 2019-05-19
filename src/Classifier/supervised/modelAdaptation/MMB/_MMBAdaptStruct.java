package Classifier.supervised.modelAdaptation.MMB;

import java.util.Collection;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
import structures._Doc;
import structures._HDPThetaStar;
import structures._MMBNeighbor;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class _MMBAdaptStruct extends _HDPAdaptStruct {
	
	// This is cluster and edge size map.
	// key: global component parameter; val: edge size.
	protected HashMap<_HDPThetaStar, int[]> m_hdpThetaEdgeSizeMap;
	
	// This is the uj and neighbor map.
	// key: uj; val: group parameter-_HDPThetaStar.
	protected HashMap<_HDPAdaptStruct, _MMBNeighbor> m_neighborMap;
	// the mixture over global components
	protected double[] m_mixture; 
	

	public _MMBAdaptStruct(_User user) {
		super(user);
		m_hdpThetaEdgeSizeMap = new HashMap<_HDPThetaStar, int[]>();
		m_neighborMap = new HashMap<_HDPAdaptStruct, _MMBNeighbor>();
	}

	public _MMBAdaptStruct(_User user, int dim){
		super(user, dim);
		m_hdpThetaEdgeSizeMap = new HashMap<_HDPThetaStar, int[]>();
		m_neighborMap = new HashMap<_HDPAdaptStruct, _MMBNeighbor>();
	}
	// used in link prediction
	public _MMBAdaptStruct(_User user, double[] mix){
		super(user);
		m_mixture = mix;
	}
	
	
	/********Functions used in MMB model.********/
	public Collection<_HDPThetaStar> getHDPTheta4Edge(){
		return m_hdpThetaEdgeSizeMap.keySet();
	}
	
	// Return the total number of edges (0 and 1) in the given thetaStar.
	public int getHDPThetaEdgeSize(_HDPThetaStar s){
		if(m_hdpThetaEdgeSizeMap.containsKey(s))
			return Utils.sumOfArray(m_hdpThetaEdgeSizeMap.get(s));
		else 
			return 0;
	}

	// Return the number of one type of edges in the given thetaStar
	public int getHDPThetaOneEdgeSize(_HDPThetaStar s, int e){
		if(m_hdpThetaEdgeSizeMap.containsKey(s))
			return m_hdpThetaEdgeSizeMap.get(s)[e];
		else 
		return 0;
	}
	
	// Update the size of the edges belong to the group.
	public void incHDPThetaStarEdgeSize(_HDPThetaStar s, int v, int e){
		if (v==0)
			return;
		else if(v > 0){
			if(!m_hdpThetaEdgeSizeMap.containsKey(s)){
				m_hdpThetaEdgeSizeMap.put(s, new int[]{0, 0});
			}
			m_hdpThetaEdgeSizeMap.get(s)[e] += v;
		} else{
			if(!m_hdpThetaEdgeSizeMap.containsKey(s)){
				System.err.println("[Error]This theta does not exist!");
			}
			int[] edgeSize = m_hdpThetaEdgeSizeMap.get(s);
			edgeSize[e] += v;
			if(edgeSize[e] < 0)
				System.err.println("[Error]Invalid value in updating theta!");
			else if(edgeSize[e] == 0 && Utils.sumOfArray(edgeSize) == 0)
				m_hdpThetaEdgeSizeMap.remove(s);
		}
		
//		if (v>0)
//			m_hdpThetaEdgeSizeMap.put(s, v);
//		else
//			m_hdpThetaEdgeSizeMap.remove(s);
	}
	
	public int getEdge(_HDPAdaptStruct uj){
		return m_neighborMap.get(uj).getEdge();
	}
	
	public boolean hasEdge(_HDPAdaptStruct uj){
		return m_neighborMap.containsKey(uj);
	}
	// Add a neighbor, update the <Neighbor, ThetaStar> map and <Neighbor, edge_value> map.
	public void addNeighbor(_HDPAdaptStruct uj, _HDPThetaStar theta, int e){
		m_neighborMap.put(uj, new _MMBNeighbor(uj, theta, e));
	}
	
	// Remove one neighbor, 
	public void rmNeighbor(_HDPAdaptStruct uj){
		m_neighborMap.remove(uj);
	}
	
	// Get the group membership for the edge between i->j.
	public _HDPThetaStar getThetaStar(_HDPAdaptStruct uj){
		return m_neighborMap.get(uj).getHDPThetaStar();
	}
	
	public _MMBNeighbor getOneNeighbor(_HDPAdaptStruct u){
		return m_neighborMap.get(u);
	}
	public HashMap<_HDPAdaptStruct, _MMBNeighbor> getNeighbors(){
		return m_neighborMap;
	}
	
	@Override
	public double evaluate(_Doc doc) {
		_Review r = (_Review) doc;
		double prob = 0, sum = 0;
		double[] probs = r.getCluPosterior();
		int n, m, k;

		//not adaptation based
		if (m_dim==0) {
			for(k=0; k<probs.length; k++) {
				sum = Utils.dotProduct(CLRWithMMB.m_hdpThetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				if(MTCLRWithMMB.m_supWeights != null && CLRWithDP.m_q != 0)
					sum += CLRWithDP.m_q*Utils.dotProduct(MTCLRWithMMB.m_supWeights, doc.getSparse(), 0);
								
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		} else {
			double As[];
			for(k=0; k<probs.length; k++) {
				As = CLRWithMMB.m_hdpThetaStars[k].getModel();
				sum = As[0]*CLinAdaptWithMMB.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: doc.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*CLinAdaptWithMMB.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
				}
				
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		}
		
		//accumulate the prediction results during sampling procedure
		doc.m_pCount ++;
		doc.m_prob += Math.exp(prob); //>0.5?1:0;
		return Math.exp(prob);
	}	
	
	public double evaluateIndependently(_Doc doc) {
		_Review r = (_Review) doc;
		double prob = 0, sum = 0;
		double[] probs = r.getCluPosterior();
		int n, m, k;

		//not adaptation based
		if (m_dim==0) {
			for(k=0; k<probs.length; k++) {
				sum = Utils.dotProduct(CLRWithMMB.m_hdpThetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				if(MTCLRWithMMB.m_supWeights != null && CLRWithDP.m_q != 0)
					sum += CLRWithDP.m_q*Utils.dotProduct(MTCLRWithMMB.m_supWeights, doc.getSparse(), 0);
								
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		} else {
			double As[];
			for(k=0; k<probs.length; k++) {
				As = CLRWithMMB.m_hdpThetaStars[k].getModel();
				sum = As[0]*CLinAdaptWithMMB.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: doc.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*CLinAdaptWithMMB.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
				}
				
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		}
		return Math.exp(prob);
	}	
	protected void setMixture(double[] m){
		m_mixture = m;
	}
	
	public double[] getMixture(){
		return m_mixture;
	}
	
	// predict the label based on the current user's model
	public int predictIndependently(_Doc doc){
		double prob = evaluateIndependently(doc);
		return prob>=0.5 ? 1:0;
	}	
}
