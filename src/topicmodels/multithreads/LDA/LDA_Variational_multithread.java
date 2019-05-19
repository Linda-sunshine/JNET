package topicmodels.multithreads.LDA;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import topicmodels.LDA.LDA_Variational;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;
import utils.Utils;

public class LDA_Variational_multithread extends LDA_Variational {

	public class LDA_worker extends TopicModel_worker {
		protected double[] alphaStat;
		
		public LDA_worker(int number_of_topics, int vocabulary_size) {
			super(number_of_topics, vocabulary_size);
			alphaStat = new double[number_of_topics];
		}
		
		@Override
		public double calculate_E_step(_Doc d) {	
			double last = 1;			
			if (m_varConverge>0)
				last = calculate_log_likelihood(d);
			
			double current = last, converge, logSum, v;
			int iter = 0, wid;
			_SparseFeature[] fv = d.getSparse();
			
			do {
				//variational inference for p(z|w,\phi)
				for(int n=0; n<fv.length; n++) {
					wid = fv[n].getIndex();
					v = fv[n].getValue();
					for(int i=0; i<number_of_topics; i++)
						d.m_phi[n][i] = topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]);
					
					logSum = Utils.logSum(d.m_phi[n]);
					for(int i=0; i<number_of_topics; i++)
						d.m_phi[n][i] = Math.exp(d.m_phi[n][i] - logSum);
				}
				
				//variational inference for p(\theta|\gamma)
				System.arraycopy(m_alpha, 0, d.m_sstat, 0, m_alpha.length);
				for(int n=0; n<fv.length; n++) {
					v = fv[n].getValue();
					for(int i=0; i<number_of_topics; i++)
						d.m_sstat[i] += d.m_phi[n][i] * v;// 
				}
				
				if (m_varConverge>0) {
					current = calculate_log_likelihood(d);			
					converge = Math.abs((current - last)/last);
					last = current;
					
					if (converge<m_varConverge)
						break;
				}
			} while(++iter<m_varMaxIter);
			
			//collect the sufficient statistics after convergence
			if (m_collectCorpusStats) {
				this.collectStats(d);			
				return current;
			} else 
				return calculate_log_likelihood(d);
		}
		
		protected void collectStats(_Doc d) {
			_SparseFeature[] fv = d.getSparse();
			int wid;
			double v; 
			for(int n=0; n<fv.length; n++) {
				wid = fv[n].getIndex();
				v = fv[n].getValue();
				for(int i=0; i<number_of_topics; i++)
					sstat[i][wid] += v*d.m_phi[n][i];
			}
			
			double diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
			for(int i=0; i<number_of_topics; i++)
				alphaStat[i] += Utils.digamma(d.m_sstat[i]) - diGammaSum;
		}
		
		// this is directly copied from LDA_Variational.java
		@Override
		public double inference(_Doc d) {
			initTestDoc(d);		
			double likelihood = calculate_E_step(d);
			estThetaInDoc(d);
			return calc_term_log_likelihood(d);
		}
		
		@Override
		public double accumluateStats(double[][] word_topic_sstat) {
			for(int k=0; k<number_of_topics; k++)
				m_alphaStat[k] += alphaStat[k];

			return super.accumluateStats(word_topic_sstat);
		}
		
		@Override
		public void resetStats() {
			Arrays.fill(alphaStat, 0);
			super.resetStats();
		}
	}
	
	public LDA_Variational_multithread(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, 
			int number_of_topics, double alpha, int varMaxIter, double varConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge);
		m_multithread = true;
	}

	@Override
	public String toString() {
		return String.format("multithread LDA[k:%d, alpha:%.3f, beta:%.3f, Variational]", number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];
		m_workers = new LDA_worker[cores];
		
		for(int i=0; i<cores; i++)
			m_workers[i] = new LDA_worker(number_of_topics, vocabulary_size);
		
		int workerID = 0;
		for(_Doc d:collection) {
			m_workers[workerID%cores].addDoc(d);
			workerID++;
		}
		
		super.initialize_probability(collection);
	}
	
	@Override
	protected void init() { // clear up for next iteration
		super.init();
		for(TopicModelWorker worker:m_workers)
			worker.resetStats();
	}
}
