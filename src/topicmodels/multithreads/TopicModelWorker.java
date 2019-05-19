package topicmodels.multithreads;

import structures._Doc;
import topicmodels.multithreads.TopicModel_worker.RunType;

public interface TopicModelWorker extends Runnable {

	public void setType(RunType type);
	
	public void addDoc(_Doc d);
	
	public void clearCorpus();
	
	public double calculate_E_step(_Doc d);
	
	public double inference(_Doc d);
	
	public double accumluateStats(double[][] word_topic_sstat);
	
	public void resetStats();
	
	public double getLogLikelihood();
	
	public double getPerplexity();

	public double getTotalWords();
}
