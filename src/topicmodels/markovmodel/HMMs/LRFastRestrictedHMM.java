package topicmodels.markovmodel.HMMs;

import structures._Doc;
import utils.Utils;

/**
 * 
 * @author Hongning Wang
 * Extension of FastRestrictedHMM by using logistic regression to determine the topic transition probabilities
 */

public class LRFastRestrictedHMM extends FastRestrictedHMM {
	
	double[] m_omega; // feature weight for topic transition
	double[] m_epsilons; // topic transition for each sentence
	
	public LRFastRestrictedHMM (double[] omega, int maxSeqSize, int topicSize) {
		super(-1, maxSeqSize, topicSize, 2);//no global epsilon
		m_omega = omega;
		m_epsilons  = new double[maxSeqSize];
	}

	@Override
	public double ForwardBackward(_Doc d, double[][] emission) {
		initEpsilons(d);		
		return super.ForwardBackward(d, emission);
	}
	
	//all epsilons in real space!!
	void initEpsilons(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++)
			m_epsilons[t] = Utils.logistic(d.getSentence(t-1).getTransitFvs(), m_omega);//first sentence does not have features
	}
	
	@Override
	double getEpsilon(int t) {
		return m_epsilons[t];
	}
	
	@Override
	public void BackTrackBestPath(_Doc d, double[][] emission, int[] path) {
		initEpsilons(d);		
		super.BackTrackBestPath(d, emission, path);
	}	
}
