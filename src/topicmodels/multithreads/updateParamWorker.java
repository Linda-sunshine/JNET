package topicmodels.multithreads;

import topicmodels.multithreads.updateParam_worker.RunType;

public interface updateParamWorker extends Runnable{
	
	public void setType(RunType type);
	
	public void addParameter(double[]param, int paramIndex);
	
	public void clearParameter();
	
	public void calculate_M_step();
	
	public void returnParameter(double[] param, int index);
	
	public double getLogLikelihood();
}
