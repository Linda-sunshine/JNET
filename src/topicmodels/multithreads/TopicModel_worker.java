package topicmodels.multithreads;

import java.util.ArrayList;
import java.util.Arrays;

import structures._Doc;

//E-step needs to be implemented by different models
public abstract class TopicModel_worker implements TopicModelWorker {

	public enum RunType {
		RT_inference,
		RT_EM
	}
	
	protected ArrayList<_Doc> m_corpus;
	protected double m_likelihood;
	protected double m_perplexity;
	protected int m_totalWords;
	
	protected double[][] sstat; // p(w|z)
	protected int number_of_topics;
	protected int vocabulary_size;
	protected RunType m_type = RunType.RT_EM;//EM is the default type
	
	public TopicModel_worker(int number_of_topics, int vocabulary_size) {
		this.number_of_topics = number_of_topics;
		this.vocabulary_size = vocabulary_size;
		
		m_corpus = new ArrayList<_Doc>();
		sstat = new double[number_of_topics][vocabulary_size];
	}
	
	public void setType(RunType type) {
		m_type = type;
	}
	
	@Override
	public double getLogLikelihood() {
		return m_likelihood;
	}
	
	@Override
	public double getPerplexity() {
		return m_perplexity;
	}

	public double getTotalWords(){
		return m_totalWords;
	}

	@Override
	public void run() {
		m_likelihood = 0;
		m_perplexity = 0;
		m_totalWords = 0;
		
		double loglikelihood = 0, log2 = Math.log(2.0);
//		System.out.println("[Info]Thread corpus size\t" + m_corpus.size());
		long eStartTime = System.currentTimeMillis();

		for(_Doc d:m_corpus) {
			if (m_type == RunType.RT_EM)
				m_likelihood += calculate_E_step(d);
			else if (m_type == RunType.RT_inference) {
				loglikelihood = inference(d);
				m_perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);//this assumes the likelihood is only contributed by the words in documents 
				m_likelihood += loglikelihood;
				m_totalWords += d.getTotalDocLength();
			}
		}
		long eEndTime = System.currentTimeMillis();

//		System.out.format("[Info]Execution time in E-step time %.2fs\n", (eEndTime - eStartTime)/1000.0);
	}
	
	@Override
	public void addDoc(_Doc d) {
		m_corpus.add(d);
	}
	
	public void clearCorpus() {
		m_corpus.clear();
	}
	
	@Override
	public void resetStats() {
		for(int i=0; i<sstat.length; i++)
			Arrays.fill(sstat[i], 0);			
	}

	@Override
	public double accumluateStats(double[][] word_topic_sstat) {
		for(int k=0; k<number_of_topics; k++) {
			for(int v=0; v<vocabulary_size; v++)
				word_topic_sstat[k][v] += sstat[k][v];
		}
		return m_likelihood;
	}	
}
