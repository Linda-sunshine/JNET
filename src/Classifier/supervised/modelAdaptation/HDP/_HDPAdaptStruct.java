package Classifier.supervised.modelAdaptation.HDP;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
import structures._Doc;
import structures._HDPThetaStar;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class _HDPAdaptStruct extends _DPAdaptStruct {
	
	_HDPThetaStar  m_hdpThetaStar = null;
	
	// This is cluster and documents size map.
	// key: global component parameter; val: document member size
	protected HashMap<_HDPThetaStar, Integer> m_hdpThetaMemSizeMap;
	
	public _HDPAdaptStruct(_User user) {
		super(user);
		m_hdpThetaMemSizeMap = new HashMap<_HDPThetaStar, Integer>();
	}

	public _HDPAdaptStruct(_User user, int dim){
		super(user, dim);
		m_hdpThetaMemSizeMap = new HashMap<_HDPThetaStar, Integer>();
	}

	//Return the number of members in the given thetaStar.
	public int getHDPThetaMemSize(_HDPThetaStar s){
		if(m_hdpThetaMemSizeMap.containsKey(s))
			return m_hdpThetaMemSizeMap.get(s);
		else 
			return 0;
	}

	//will remove the key if the updated value is zero
	public void incHDPThetaStarMemSize(_HDPThetaStar s, int v){
		if (v==0)
			return;
		
		if(m_hdpThetaMemSizeMap.containsKey(s))
			v += m_hdpThetaMemSizeMap.get(s);
		
		if (v>0)
			m_hdpThetaMemSizeMap.put(s, v);
		else
			m_hdpThetaMemSizeMap.remove(s);
	}
	
	/***Functions and variables used in confidence learning.***/
	// We need the reviews to calculate the weighted sum for one thetastar.
	protected HashMap<_HDPThetaStar, ArrayList<_Review>> m_hdpThetaMemMap = new HashMap<_HDPThetaStar, ArrayList<_Review>>();

	// Return the weighted sum of members in the given thetaStar, i.e., each count is treated differently regarding their weight.
	public int getWeightedHDPThetaMemSize(_HDPThetaStar s){
		double wSum = 0;
		if(m_hdpThetaMemMap.containsKey(s)){
			for(_Review r: m_hdpThetaMemMap.get(s))
				wSum += r.getConfidence();
			return (int) wSum;
		} else 
			return 0;
	}

	// Add one review to one thetastar.
	protected void addHDPThetaStarMem(_HDPThetaStar s, _Review r){
		if(!m_hdpThetaMemMap.containsKey(s))
			m_hdpThetaMemMap.put(s, new ArrayList<_Review>());
		m_hdpThetaMemMap.get(s).add(r);
	}
	// Remove one review from one thetastar.
	protected void rmHDPThetaStarMem(_HDPThetaStar s, _Review r){
		if(!m_hdpThetaMemMap.containsKey(s)){
			System.err.println("The Theta star doesn't exist!");
			return;
		}
		m_hdpThetaMemMap.get(s).remove(r);
		if(m_hdpThetaMemMap.get(s).size() == 0)
			m_hdpThetaMemMap.remove(s);
	}
	
	public Collection<_HDPThetaStar> getHDPTheta4Rvw(){
		return m_hdpThetaMemSizeMap.keySet();
	}
	
	public void setThetaStar(_HDPThetaStar theta){
		m_hdpThetaStar = theta;
	}
	@Override
	public _HDPThetaStar getThetaStar(){
		return m_hdpThetaStar;
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
				sum = Utils.dotProduct(CLRWithHDP.m_hdpThetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				if(MTCLRWithHDP.m_supWeights != null && MTCLRWithHDP.m_q != 0)
					sum += CLRWithDP.m_q*Utils.dotProduct(MTCLRWithHDP.m_supWeights, doc.getSparse(), 0);
								
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		} else {
			double As[];
			for(k=0; k<probs.length; k++) {
				As = CLRWithHDP.m_hdpThetaStars[k].getModel();
				sum = As[0]*CLinAdaptWithHDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: doc.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*CLinAdaptWithHDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
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
		return prob;
	}	
	
	// Evaluate the performance of the global part.
	public double evaluateG(_Doc doc){
		_Review r = (_Review) doc;
		double prob = 0, sum = 0;
		double[] probs = r.getCluPosterior();
		int n, k;

		for(k=0; k<probs.length; k++) {
			//As = CLRWithHDP.m_hdpThetaStars[k].getModel();
			sum = CLinAdaptWithHDP.m_supWeights[0];//Bias term: w_s0*a0+b0.
			for(_SparseFeature fv: doc.getSparse()){
				n = fv.getIndex() + 1;
				sum += CLinAdaptWithHDP.m_supWeights[n] * fv.getValue();
			}
				
			//to maintain numerical precision, compute the expectation in log space as well
			if (k==0)
				prob = probs[k] + Math.log(Utils.logistic(sum));
			else
				prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
		}
		
		//accumulate the prediction results during sampling procedure
		doc.m_pCount_g++;
		doc.m_prob_g += Math.exp(prob); //>0.5?1:0;
		return prob;
	}
}
