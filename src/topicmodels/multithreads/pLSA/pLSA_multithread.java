package topicmodels.multithreads.pLSA;

import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import topicmodels.multithreads.TopicModel_worker;
import topicmodels.pLSA.pLSA;

public class pLSA_multithread extends pLSA {

	public class pLSA_worker extends TopicModel_worker {
		
		public pLSA_worker(int number_of_topics, int vocabulary_size) {
			super(number_of_topics, vocabulary_size);
		}
		
		public double calculate_E_step(_Doc d) {	
			double propB; // background proportion
			double exp; // expectation of each term under topic assignment
			for(_SparseFeature fv:d.getSparse()) {
				int j = fv.getIndex(); // jth word in doc
				double v = fv.getValue();
				
				//-----------------compute posterior----------- 
				double sum = 0;
				for(int k=0;k<number_of_topics;k++)
					sum += d.m_topics[k]*topic_term_probabilty[k][j];//shall we compute it in log space?
				
				propB = m_lambda * background_probability[j];
				propB /= propB + (1-m_lambda) * sum;//posterior of background probability
				
				//-----------------compute and accumulate expectations----------- 
				for(int k=0;k<number_of_topics;k++) {
					exp = v * (1-propB)*d.m_topics[k]*topic_term_probabilty[k][j]/sum;
					d.m_sstat[k] += exp;
					
					if (m_collectCorpusStats)
						sstat[k][j] += exp;
				}
			}
			
			if (m_collectCorpusStats==false || m_converge>0)
				return calculate_log_likelihood(d);
			else
				return 1;//no need to compute likelihood
		}

		// this is directly copied from TopicModel.java
		@Override
		public double inference(_Doc d) {
			initTestDoc(d);//this is not a corpus level estimation
			
			double delta, last = 1, current;
			int  i = 0;
			do {
				current = calculate_E_step(d);
				estThetaInDoc(d);			
				
				delta = (last - current)/last;
				last = current;
			} while (Math.abs(delta)>m_converge && ++i<number_of_iteration);
			return current;
		}
	}	
	
	public pLSA_multithread(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			double lambda, //arguments for 2topic topic model
			int number_of_topics, double alpha) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha);
		m_multithread = true;
	}
	
	@Override
	public String toString() {
		return String.format("multi-thread pLSA[k:%d, lambda:%.2f]", number_of_topics, m_lambda);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);
		
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];
		m_workers = new pLSA_worker[cores];
		
		for(int i=0; i<cores; i++)
			m_workers[i] = new pLSA_worker(number_of_topics, vocabulary_size);
		
		int workerID = 0;
		for(_Doc d:collection) {//evenly allocate the work load
			m_workers[workerID%cores].addDoc(d);
			workerID++;
		}
	}
	
	@Override
	protected void init() { // clear up for next iteration
		super.init();
		for(int i=0; i<m_workers.length; i++)
			m_workers[i].resetStats();
	}
}
