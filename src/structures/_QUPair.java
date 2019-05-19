package structures;

import java.util.ArrayList;

import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import utils.Utils;

public class _QUPair implements Comparable<_QUPair>{
	public int m_y;
	public double m_score;//ranking score for this QUPair
	
	//for LambdaRank
	public ArrayList<_QUPair> m_betterURLs, m_worseURLs;
	
	//feature part (dense representation)
	public double[] m_rankFv;
	
	public _QUPair(int y, double[] features){//entrance for LoadHRS and LoadUser
		m_y = y;
		m_worseURLs = null;
		m_betterURLs = null;
		m_score = 0;
		m_rankFv = features;
	}
	
	@Override
	public String toString() {
		return String.format("%d:%.4f", m_y, m_score);
	}
	
	public void addWorseURL(_QUPair qu){
		if (m_worseURLs==null)
			m_worseURLs = new ArrayList<_QUPair>();
		m_worseURLs.add(qu);
	}
	
	public void addBetterURL(_QUPair qu){
		if (m_betterURLs==null)
			m_betterURLs = new ArrayList<_QUPair>();
		m_betterURLs.add(qu);
	}
	
	//for RankSVM
	Feature[] getDiffFv(_QUPair dj) {
		ArrayList<Feature> fvs = new ArrayList<Feature>();
		double value;
		for(int i=0; i<m_rankFv.length; i++) {
			value = m_rankFv[i] - dj.m_rankFv[i]; 
			if (value != 0)
				fvs.add(new FeatureNode(i+1, value));
		}
		
		if (fvs.size()==0)
			return null;
		return fvs.toArray(new Feature[fvs.size()]);
	}
	
	public double score(double[] w){
		m_score = Utils.dotProduct(w, m_rankFv);
		return m_score;
	}
	
	//rank by predicted score
	public int compareTo (_QUPair qu){
		if (this.m_score > qu.m_score)
			return -1;
		else if (this.m_score < qu.m_score)
			return 1;
		else 
			return 0;
	}
}
