package topicmodels.markovmodel.HMMs;

import structures._Doc;
import utils.Utils;

/**
 * 
 * @author Hongning Wang
 * Extension of FastRestrictedHMM_sentiment by using logistic regression to determine the topic transition and sentiment transition probabilities
 */
public class LRFastRestrictedHMM_sentiment extends FastRestrictedHMM_sentiment {
	
	double[] m_omega; // feature weight for topic transition
	double[] m_delta; // feature weight for sentiment transition
	double[] m_epsilons; // topic transition for each sentence
	double[] m_sigmas;
	
	public LRFastRestrictedHMM_sentiment (double[] omega, double[] delta, int maxSeqSize, int topicSize) {
		super(-1, -1, maxSeqSize, topicSize);//no global epsilon, epsilon = -1,sigma = -1
		m_omega = omega;
		m_delta = delta;
		m_epsilons  = new double[maxSeqSize];
		m_sigmas = new double[maxSeqSize];
	}

	@Override
	public double ForwardBackward(_Doc d, double[][] emission) {
		initEpsilons(d);
		initSigmas(d);
		return super.ForwardBackward(d, emission);
	}
	
	//all epsilon in real space!!
	void initEpsilons(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++)
			m_epsilons[t] = Utils.logistic(d.getSentence(t-1).getTransitFvs(), m_omega);
	}
	
	//all sigma in real space!!
	void initSigmas(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++)
			m_sigmas[t] = Utils.logistic(d.getSentence(t-1).getSentiTransitFvs(), m_delta);
	}

	@Override
	double getEpsilon(int t) {
		return m_epsilons[t];
	}
	
	@Override
	double getSigma(int t) {
		return m_sigmas[t];
	}
	
	@Override
	public void BackTrackBestPath(_Doc d, double[][] emission, int[] path) {
		initEpsilons(d);
		initSigmas(d);
		super.BackTrackBestPath(d, emission, path);
	}
}
