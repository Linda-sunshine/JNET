/**
 * 
 */
package topicmodels.multithreads.HTSM;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._Stn;
import topicmodels.markovmodel.LRHTSM;
import topicmodels.markovmodel.HMMs.FastRestrictedHMM_sentiment;
import topicmodels.markovmodel.HMMs.LRFastRestrictedHMM_sentiment;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;

/**
 * @author Hongning Wang
 * Multi-thread implementation of LRHTSM
 * Due to complex inheritance relation in HTMM-HTSM-LRHTSM, we reimplement most of functions in HTMM again 
 */
public class LRHTSM_multithread extends LRHTSM {

	class LRHTSM_worker extends TopicModel_worker {		// cache structure for sufficient statistics
		
		double[][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w)
	 	double[][] emission;  // emission probability of p(s|z)
	 	
	 	//hmm-style inferencer
	 	FastRestrictedHMM_sentiment m_hmm; 
	 	
	 	public LRHTSM_worker(int number_of_topics, int vocabulary_size, int maxSeqSize) {
	 		super(number_of_topics, vocabulary_size);
	 		
	 		//cache in order to avoid frequently allocating new space
			p_dwzpsi = new double[maxSeqSize][constant * number_of_topics]; // max|S_d| * (2*K)
			emission = new double[maxSeqSize][number_of_topics]; // max|S_d| * K
			
			m_hmm = new LRFastRestrictedHMM_sentiment(m_omega, m_delta, maxSeqSize, number_of_topics); 
	 	}
		
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
			//Step 1: pre-compute emission probability
			if (m_collectCorpusStats)//indicates this is training
				ComputeEmissionProbsForDoc(d);
			
			//Step 2: use forword/backword algorithm to compute the posterior
			double logLikelihood = m_hmm.ForwardBackward(d, emission);
			
			//Step 3: collection expectations from the posterior distribution
			m_hmm.collectExpectations(p_dwzpsi);//expectations will be in the original space	
			accTheta(d);
			estThetaInDoc(d);//get the posterior of theta
			
			if (m_collectCorpusStats) {
				accEpsilonStat(d);
				accSigmaStat(d);
				accPhiStat(d);
			}
			
			return logLikelihood + docThetaLikelihood(d);
		}
		
		@Override
		public double inference(_Doc d) {
			initTestDoc(d);//this is not a corpus level estimation
			ComputeEmissionProbsForDoc(d);//to avoid repeatedly computing emission probability 
			
			double delta, last = 1, current;
			int  i = 0;
			do {
				current = calculate_E_step(d);
				estThetaInDoc(d);			
				
				delta = (last - current)/last;
				last = current;
			} while (Math.abs(delta)>m_converge && ++i<number_of_iteration);
			
			int path[] = get_MAP_topic_assignment(d);
			_Stn[] sentences = d.getSentences();
			for(i=0; i<path.length; i++)
				sentences[i].setTopic(path[i]);
			return current;
		}
		
		void accTheta(_Doc d) {
			for(int t=0; t<d.getSenetenceSize(); t++) {
				for(int i=0; i<number_of_topics; i++) 
					for(int j=0; j<constant-1; j++)
						d.m_sstat[i] += this.p_dwzpsi[t][i + j*number_of_topics];//only consider \psi=1
			}
		}
		
		void accEpsilonStat(_Doc d) {
			for(int t=1; t<d.getSenetenceSize(); t++) {
				double s = 0;
				for(int i=0; i<(constant-1)*number_of_topics; i++) 
					s += this.p_dwzpsi[t][i];
				d.getSentence(t).setTransitStat(s); //store the statistics at position t!!
			}
		}
		
		void accSigmaStat(_Doc d) {
			for(int t=1; t<d.getSenetenceSize(); t++) {
				double s = 0;
				for(int i=0; i<number_of_topics; i++) 
					s += this.p_dwzpsi[t][i];
				d.getSentence(t).setSentiTransitStat(s); //store the statistics at position t!!
			}
		}
		
		//probabilities of topic assignment
		void accPhiStat(_Doc d) {
			double prob;
			for(int t=0; t<d.getSenetenceSize(); t++) {
				_Stn s = d.getSentence(t);
				for(_SparseFeature f:s.getFv()) {
					int wid = f.getIndex();
					double v = f.getValue();//frequency
					for(int i=0; i<number_of_topics; i++) {
						prob = this.p_dwzpsi[t][i];
						for(int j=1; j<constant; j++)
							prob += this.p_dwzpsi[t][i + j*number_of_topics];
						this.sstat[i][wid] += v * prob;
					}
				}
			}
		}	
		
		int[] get_MAP_topic_assignment(_Doc d) {
			int path [] = new int[d.getSenetenceSize()];
			m_hmm.BackTrackBestPath(d, emission, path);
			return path;
		}
	}
	
	public LRHTSM_multithread(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha, //arguments for pLSA	
			double lambda) {//arguments for LR-HTMM		
		super(number_of_iteration, converge, beta, c, 
				number_of_topics, alpha,
				lambda);
		
		m_multithread = true;
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];
		m_workers = new LRHTSM_worker[cores];
		
		for(int i=0; i<cores; i++)
			m_workers[i] = new LRHTSM_worker(number_of_topics, vocabulary_size, m_corpus.getLargestSentenceSize());
		
		int workerID = 0;
		for(_Doc d:collection) {
			m_workers[workerID%cores].addDoc(d);
			workerID++;
		}
		
		super.initialize_probability(collection);
	}
	
	@Override
	public String toString() {
		return String.format("multi-thread LR-HTSM[k:%d, alpha:%.3f, beta:%.3f, lambda:%.2f]", number_of_topics, d_alpha, d_beta, m_lambda);
	}

	@Override
	protected void init() { // clear up for next iteration
		super.init();
		for(TopicModelWorker worker:m_workers)
			worker.resetStats();
	}
}
