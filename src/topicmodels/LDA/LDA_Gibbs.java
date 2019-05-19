package topicmodels.LDA;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import structures._Corpus;
import structures._Doc;
import structures._Word;
import topicmodels.pLSA.pLSA;
import utils.Utils;

/**
 * 
 * @author hongning
 * Gibbs sampling for Latent Dirichlet Allocation model
 * Griffiths, Thomas L., and Mark Steyvers. "Finding scientific topics."
 */
public class LDA_Gibbs extends pLSA {
	protected Random m_rand;
	protected int m_burnIn; // discard the samples within burn in period
	protected int m_lag; // lag in accumulating the samples
	
	protected double[] m_topicProbCache;
	//all computation here is not in log-space!!!
	public LDA_Gibbs(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha);
		
		m_rand = new Random();
		m_burnIn = (int) (burnIn * number_of_iteration);
		m_lag = lag;
		m_topicProbCache = new double[number_of_topics];
	}
	
	@Override
	protected void createSpace() {
		super.createSpace();
		m_sstat = new double[number_of_topics];
	}

	@Override
	public String toString() {
		return String.format("LDA[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		Arrays.fill(m_topicProbCache, 0);
		
		// initialize topic-word allocation, p(w|z)
		for(_Doc d:collection) {
			d.setTopics4Gibbs(number_of_topics, d_alpha);//allocate memory and randomize it
			for(_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()] ++;
				m_sstat[w.getTopic()] ++;
			}
		}
		
		imposePrior();		
	}
	
	@Override
	protected void imposePrior() {		
		if (word_topic_prior!=null) {//we have enforced that the topic size is at least as many as prior seed words
			if (m_sentiAspectPrior) {
				int size = word_topic_prior.length/2, shift = number_of_topics/2;//if it is sentiment aspect prior, the size must be even
				for(int k=0; k<size; k++) {
					for(int n=0; n<vocabulary_size; n++) {
						word_topic_sstat[k][n] += word_topic_prior[k][n];
						m_sstat[k] += word_topic_prior[k][n];
						
						word_topic_sstat[k + shift][n] += word_topic_prior[k + size][n];
						m_sstat[k + shift] += word_topic_prior[k + size][n];
					}
				}
			} else {//no symmetric property
				for(int k=0; k<word_topic_prior.length; k++) {
					for(int n=0; n<vocabulary_size; n++) {
						word_topic_sstat[k][n] += word_topic_prior[k][n];
						m_sstat[k] += word_topic_prior[k][n];
					}
				}
			}
		}
	}
	
	@Override
	protected void init() {
		//we just simply permute the training instances here
		int t;
		_Doc tmpDoc;
		for(int i=m_trainSet.size()-1; i>1; i--) {
			t = m_rand.nextInt(i);
			
			tmpDoc = m_trainSet.get(i);
			m_trainSet.set(i, m_trainSet.get(t));
			m_trainSet.set(t, tmpDoc);			
		}
	}
	
	@Override
	protected void initTestDoc(_Doc d) {
		d.setTopics4Gibbs(number_of_topics, d_alpha);//allocate memory and randomize it
	}
	
	@Override
	public double calculate_E_step(_Doc d) {	
		d.permutation();
		double p;
		int wid, tid;
		for(_Word w:d.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			
			//remove the word's topic assignment
			d.m_sstat[tid] --;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}

			//perform random sampling
			p = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				m_topicProbCache[tid] = topicInDocProb(tid, d)
						* wordByTopicProb(tid, wid);
				p += m_topicProbCache[tid];
			}

			p *= m_rand.nextDouble();
			
			tid = -1;
			while(p>0 && tid<number_of_topics-1) {
				tid ++;
				p -= m_topicProbCache[tid];
			}
			
			//assign the selected topic to word
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
		
		return 0;
//		if (m_collectCorpusStats == false || m_converge>0)
//			return 0;
////			return calculate_log_likelihood(d);
//		else
//			return 0;
	}
	
	protected double wordByTopicProb(int tid, int wid) {
		return word_topic_sstat[tid][wid] / m_sstat[tid];
	}

	protected double topicInDocProb(int tid, _Doc d) {
		return (d.m_sstat[tid]);
	}

	@Override
	public void calculate_M_step(int iter) {	
		//literally we do not have M-step in Gibbs sampling		
		if (iter>m_burnIn && iter%m_lag == 0) {
			//accumulate p(w|z)
			for(int i=0; i<this.number_of_topics; i++) {
				for(int v=0; v<this.vocabulary_size; v++) {
					topic_term_probabilty[i][v] += word_topic_sstat[i][v]; // accumulate the samples during sampling iterations
				}
			}
			
			//accumulate p(z|d)
			for(_Doc d:m_trainSet)
				collectStats(d);
		}
	}
	
	protected void collectStats(_Doc d) {
		for(int k=0; k<this.number_of_topics; k++)
			d.m_topics[k] += d.m_sstat[k];
	}
	
	// perform inference of topic distribution in the document
	@Override
	public double inference(_Doc d) {
		initTestDoc(d);//this is not a corpus level estimation
		
		double likelihood = Double.NEGATIVE_INFINITY, count = 0;
		int  i = 0;
		do {
			calculate_E_step(d);
			if (i>m_burnIn && i%m_lag==0){
				collectStats(d);
				likelihood = Utils.logSum(likelihood, calculate_log_likelihood(d));				
				count ++;
			}
		} while (++i<this.number_of_iteration);
		
		estThetaInDoc(d);
		return likelihood - Math.log(count); // this is average joint probability!
	}
	
	@Override
	protected void finalEst() {	
		//estimate p(w|z) from all the collected samples
		for(int i=0; i<this.number_of_topics; i++)
			Utils.L1Normalization(topic_term_probabilty[i]); 
		
		//estimate p(z|d) from all the collected samples
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
	}
	
	@Override
	protected void estThetaInDoc(_Doc d) {
		Utils.L1Normalization(d.m_topics);
	}
	
	@Override
	protected double docThetaLikelihood(_Doc d) {
		double norm = Utils.sumOfArray(d.m_topics);
		double logLikelihood = 0; //Utils.lgamma(number_of_topics * d_alpha) - number_of_topics*Utils.lgamma(d_alpha);
		for(int i=0; i<this.number_of_topics; i++)
			logLikelihood += (d_alpha-1) * Math.log(d.m_topics[i]/norm);
		return logLikelihood;
	}
	
	@Override
	protected double calculate_log_likelihood(_Doc d) {
		int tid, wid;
		double logLikelihood = docThetaLikelihood(d), docSum = Utils
				.sumOfArray(d.m_sstat);
		
		for (_Word w : d.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			logLikelihood += Math.log(d.m_sstat[tid] / docSum
					* word_topic_sstat[tid][wid] / m_sstat[tid]);
		}
		return logLikelihood;
	}
	
	@Override
	protected double calculate_log_likelihood() {		
		// prior from Dirichlet distributions
		double logLikelihood = number_of_topics
				* (Utils.lgamma(vocabulary_size * d_beta) - vocabulary_size
						* Utils.lgamma(d_beta));
		for (int tid = 0; tid < this.number_of_topics; tid++) {
			for (int wid = 0; wid < this.vocabulary_size; wid++)
				logLikelihood += Utils.lgamma(word_topic_sstat[tid][wid]);
			logLikelihood -= Utils.lgamma(m_sstat[tid]);
		}
		
		return logLikelihood;
	}
}
