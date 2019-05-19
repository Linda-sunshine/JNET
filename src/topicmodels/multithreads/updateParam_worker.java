package topicmodels.multithreads;

import java.util.Arrays;


public abstract class updateParam_worker implements updateParamWorker{
	public enum RunType {
		RT_inference,
		RT_EM
	}
	
	RunType m_type = RunType.RT_EM;//EM is the default type 	

	double[] m_param;
	protected double m_paramIndex;
	protected double m_likelihood;
	
	public updateParam_worker(){
		
	}
	
	public void addParameter(double[] param, int paramIndex){
		int paramLen = param.length;
		m_param = new double[paramLen];
		System.arraycopy(param, 0, m_param, 0, paramLen);
	}
	
	public void clearParameter(){
		Arrays.fill(m_param, 0);
	}
	
	public void returnParameter(double[] param, int index){
		if(param.length==m_param.length)
			System.arraycopy(param, 0, m_param, 0, m_param.length);
	}
	
	public double getLogLikelihood(){
		return m_likelihood;
	}
	
	public void calculate_M_step(){

	}
	
	public void setType(RunType type) {
		m_type = type;
	}
	
	public void run(){
		m_likelihood = 0;

		calculate_M_step();
	}
}
