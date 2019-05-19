package topicmodels;

import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * two-topic Topic Modeling 
 */

public class twoTopic extends TopicModel {
	private final double[] m_theta;//p(w|\theta) - the only topic for each document
	protected double[] m_sstat;//c(w,d)p(z|w) - sufficient statistics for each word under topic
	
	/*p (w|theta_b) */
	protected double[] background_probability;
	protected double m_lambda; //proportion of background topic in each document
	
	public twoTopic(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			double lambda) {//arguments for 2topic topic model
		super(number_of_iteration, converge, beta, c);
		
		m_lambda = lambda;
		background_probability = c.getBackgroundProb();
		
		m_theta = new double[vocabulary_size];
		m_sstat = new double[vocabulary_size];
	}
	
	@Override
	public String toString() {
		return String.format("2Topic Model[lambda:%.2f]", m_lambda);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {}
	
	@Override
	protected void initTestDoc(_Doc d) {
		Utils.randomize(m_theta, d_beta);
    	Arrays.fill(m_sstat, 0);
	};

	@Override
	protected void init() {}
	
	@Override
	public double calculate_E_step(_Doc d) {
		for(_SparseFeature fv:d.getSparse()) {
			int wid = fv.getIndex();
			m_sstat[wid] = (1-m_lambda)*m_theta[wid];
			m_sstat[wid] = fv.getValue() * m_sstat[wid]/(m_sstat[wid]+m_lambda*background_probability[wid]);//compute the expectation
		}
		
		return calculate_log_likelihood(d);
	}
	
	@Override
	public void calculate_M_step(int iter) {		
		double sum = Utils.sumOfArray(m_sstat) + vocabulary_size * (d_beta-1.0);//with smoothing
		for(int i=0;i<vocabulary_size;i++)
			m_theta[i] = (d_beta - 1.0 + m_sstat[i]) / sum;
	}
	
	@Override
	protected double calculate_log_likelihood(_Doc d) {		
		double logLikelihood = 0.0;
		for(_SparseFeature fv:d.getSparse()) {
			int wid = fv.getIndex();
			logLikelihood += fv.getValue() * Math.log(m_lambda*background_probability[wid] + (1-m_lambda)*m_theta[wid]);
		}
		
		return logLikelihood;
	}

	@Override
	public void printTopWords(int k) {
		//we only have one topic to show
		MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
		for(int i=0; i<m_theta.length; i++) 
			fVector.add(new _RankItem(m_corpus.getFeature(i), m_theta[i]));
		
		for(_RankItem it:fVector)
			System.out.format("%s(%.3f)\t", it.m_name, it.m_value);
		System.out.println();
	}
	
	//this function can only estimate the document-specific random variables
	@Override
	protected void estThetaInDoc(_Doc d) {
		calculate_M_step(0);
	}

	@Override
	protected void finalEst() {	}

	@Override
	public void printTopWords(int k, String topWordPath) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initial() {
		// TODO Auto-generated method stub
		
	}
}
