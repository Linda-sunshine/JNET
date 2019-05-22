package topicmodels.markovmodel;

import java.util.Arrays;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._Stn;
import topicmodels.markovmodel.HMMs.FastRestrictedHMM_sentiment;

/**
 * 
 * @author Hongning Wang
 * Implementation of Hidden Sentiment Markov Model
 * Rahman, Md Mustafizur, and Hongning Wang. "Hidden Topic Sentiment Model." Proceedings of the 25th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2016.
 * 
 */
public class HTSM extends HTMM {
	
	double sigma;
	double sigma_lot; // sufficient statistic about sigma (sentiment transition)
	
	public HTSM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha) {//arguments for HTMM	
		super(number_of_iteration, converge, beta, c, number_of_topics, alpha);
		
		if (number_of_topics%2!=0) {
			System.err.println("[Error]In HTSM the number of topics has to be even!");
			System.exit(-1);
		}
		
		this.constant = 3;// we will have three sets of latent states: sentiment transit by topic transit 
		this.sigma = Math.random();		
	}
	
	@Override
	protected void createSpace() {
		super.createSpace();
		m_hmm = new FastRestrictedHMM_sentiment(epsilon, sigma, m_corpus.getLargestSentenceSize(), this.number_of_topics); 
	}
	
	@Override
	// Construct the emission probabilities for sentences under different topics in a particular document.
	void ComputeEmissionProbsForDoc(_Doc d) {			
		for(int i=0; i<d.getSenetenceSize(); i++) {
			_Stn stn = d.getSentence(i);
			Arrays.fill(emission[i], 0);
			
			int start = 0, end = this.number_of_topics;				
			if(i==0 && d.getSourceType()==2){ // first sentence is specially handled for newEgg
				//get the sentiment label of the first sentence
				int sentimentLabel = stn.getStnSentiLabel();
				if(sentimentLabel==0) {// positive sentiment in the first half						
					end = this.number_of_topics / 2;
					for(int k=end; k<this.number_of_topics; k++)
						emission[i][k] = Double.NEGATIVE_INFINITY;							
				} else if(sentimentLabel==1) { // negative sentiment in the second half
					start = this.number_of_topics / 2;
					for(int k=0; k<start; k++)
						emission[i][k] = Double.NEGATIVE_INFINITY;
				}
			}
			
			for(int k=start; k<end; k++) {
				for(_SparseFeature w:stn.getFv()) {
					emission[i][k] += w.getValue() * topic_term_probabilty[k][w.getIndex()];//all in log-space
				}
			}
		}			
	}
	
	@Override
	public double calculate_E_step(_Doc d) {
		double logLikelihood = super.calculate_E_step(d);
		
		if (m_collectCorpusStats)
			accSigmaStat(d);
		
		return logLikelihood;
	}
	
	//probabilities of sentiment switch
	void accSigmaStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				this.sigma_lot += this.p_dwzpsi[t][i];
		}
	}
	
	@Override
	public void calculate_M_step(int iter) {
		super.calculate_M_step(iter);
		
		if (iter>0) {
			this.sigma = this.sigma_lot / this.total;
			((FastRestrictedHMM_sentiment)m_hmm).setSigma(this.sigma);
		}
	}
	
	protected void init() {
		super.init();
		this.sigma_lot = 0.0; // sufficient statistics for sigma	
	}
	
	@Override
	public String toString() {
		return String.format("HTSM[k:%d, alpha:%.3f, beta:%.3f]", number_of_topics, d_alpha, d_beta);
	}
}
