package topicmodels.multithreads;

import java.util.ArrayList;
import java.util.List;

import topicmodels.multithreads.TopicModel_worker.RunType;

public abstract class EmbedModel_worker implements EmbedModelWorker{
    protected List<Object> m_objects;
    protected double m_likelihood;
    protected double m_perplexity;
    protected int number_of_topics;
    protected int vocabulary_size;
    protected RunType m_type = RunType.RT_EM;//EM is the default type

    protected int m_dim;

    public EmbedModel_worker(int number_of_topics, int vocabulary_size){
        this.number_of_topics = number_of_topics;
        this.vocabulary_size = vocabulary_size;
        m_objects = new ArrayList<>();
    }

    public EmbedModel_worker(int dim){
        m_dim = dim;
        m_objects = new ArrayList<>();
    }

    @Override
    public void setType(RunType type) {
        m_type = type;
    }

    @Override
    public void addObject(Object o){ m_objects.add(o); }

    @Override
    public double getLogLikelihood() {
        return m_likelihood;
    }

    @Override
    public double getPerplexity() {
        return m_perplexity;
    }

    @Override
    public void clearObjects() {
        m_objects.clear();
    }
}
