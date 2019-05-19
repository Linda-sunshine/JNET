package topicmodels.multithreads;

import topicmodels.multithreads.TopicModel_worker.RunType;

public interface EmbedModelWorker extends Runnable {
    public void setType(RunType type);

    public void addObject(Object o);

    public void clearObjects();

    public double calculate_E_step(Object o);

    public double accumluateStats();

    public void resetStats();

    public double getLogLikelihood();

    public double getPerplexity();
}
