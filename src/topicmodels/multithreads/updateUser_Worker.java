package topicmodels.multithreads;

public abstract class updateUser_Worker implements TopicModelWorker{
    public enum RunType {
        RT_inference,
        RT_EM
    }


}
