package topicmodels.multithreads.DCMLDA;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Doc4DCMLDA;
import topicmodels.DCM.DCMLDA;
import topicmodels.multithreads.updateParamWorker;
import topicmodels.multithreads.updateParam_worker;
import topicmodels.multithreads.updateParam_worker.RunType;
import utils.Utils;

public class DCMLDA_multithread extends DCMLDA{
	
	protected updateParamWorker[] m_updateParamWorkers;
	
	public class DCMLDA_worker extends updateParam_worker{
		protected ArrayList<double[]> m_param;
		protected ArrayList<Integer> m_paramIndex;
		
		public DCMLDA_worker(){
			super();
			m_param = new ArrayList<double[]>();
			m_paramIndex = new ArrayList<Integer>();
		}
		
		public void addParameter(double[] t_param, int t_index){
			int paramLen = t_param.length;
			double[] param = new double[paramLen];
			System.arraycopy(t_param, 0, param, 0, paramLen);
			m_param.add(param);
			m_paramIndex.add(t_index);
		}

		public void clearParameter(){
			for(double[] param: m_param)
				Arrays.fill(param, 0);
		}
		
		public void returnParameter(double[] param, int index){
			if(param.length == m_param.get(index).length){
				System.arraycopy(param, 0, m_param.get(index), 0, param.length);
			}
		}
		
		public void run(){
			System.out.println("running thread");
//			
//			for(int i=0; i<10; i++){
//				System.out.println(i);
//			}
			for(int i=0; i<m_paramIndex.size(); i++){
				calculate_M_step(m_param.get(i), m_paramIndex.get(i));
			}
			
			for(int i=0; i<m_paramIndex.size(); i++){
				int paramLength = m_param.get(i).length;
				int paramIndex = m_paramIndex.get(i);
				System.arraycopy(m_param.get(i), 0, m_beta[paramIndex], 0, paramLength);
			}
			
		}
		
		public void calculate_M_step(double[] param, int tid){
			// System.out.println("topic optimization \t"+tid);
			double diff = 0;
			int iteration = 0;
			double smoothingBeta = 0.01;
			double totalBeta = 0;
			do{
				diff = 0;
				double deltaBeta = 0;
				double wordNum = 0;
				double[] wordNum4V = new double[vocabulary_size];
				double totalBetaDenominator = 0;
				double[] totalBetaNumerator = new double[vocabulary_size];
				Arrays.fill(totalBetaNumerator, 0);
				Arrays.fill(wordNum4V, 0);
				
				totalBeta = Utils.sumOfArray(param);
				double digBeta = Utils.digamma(totalBeta);
				
				for(_Doc d:m_trainSet){
					_Doc4DCMLDA doc = (_Doc4DCMLDA)d;
					totalBetaDenominator += Utils.digamma(totalBeta+d.m_sstat[tid])-digBeta;
					for(int v=0; v<vocabulary_size; v++){
						wordNum += doc.m_wordTopic_stat[tid][v];
						wordNum4V[v] += doc.m_wordTopic_stat[tid][v];
						
						totalBetaNumerator[v] += Utils.digamma(param[v]+doc.m_wordTopic_stat[tid][v]);
						totalBetaNumerator[v] -= Utils.digamma(param[v]);
					}
				}
				
				for(int v=0; v<vocabulary_size; v++){
					if(wordNum==0)
						break;
					if(wordNum4V[v]==0){
						deltaBeta = 0;
					} else {
						deltaBeta = totalBetaNumerator[v]
								/ totalBetaDenominator;

					}
					
					double newBeta = param[v] * deltaBeta + d_beta;

					double t_diff = Math.abs(param[v] - newBeta);
					if (t_diff > diff)
						diff = t_diff;

					param[v] = newBeta;
				
				}
				

				iteration ++;
				if(iteration > m_newtonIter)
					break;
				// System.out.println("beta iteration\t"+iteration);
			}while(diff>m_newtonConverge);
			
			System.out.println("iteration\t"+iteration);

		}

	}
	
	public DCMLDA_multithread(int number_of_iteration, double converge, 
			double beta, _Corpus c, double lambda, int number_of_topics, 
			double alpha, double burnIn, int lag, int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag,newtonIter, newtonConverge);
	
		m_multithread = false;
	}
	
	public String toString(){
		return String.format("multithread DCM LDA[k:%d]", number_of_topics);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		super.initialize_probability(collection);
		
		int cores = Runtime.getRuntime().availableProcessors();
		
		m_threadpool = new Thread[cores];
		m_updateParamWorkers = new DCMLDA_worker[cores];
		
		for(int i=0; i<cores; i++)
			m_updateParamWorkers[i] = new DCMLDA_worker();
		
		int workerID = 0;
		for(int k=0; k<number_of_topics; k++){
			m_updateParamWorkers[workerID%cores].addParameter(m_beta[k], k);
			workerID ++;		
		}
		
	}
	
	protected void updateBeta(){
		for(int i=0; i<m_updateParamWorkers.length; i++){
			m_updateParamWorkers[i].setType(RunType.RT_EM);
			m_threadpool[i] = new Thread(m_updateParamWorkers[i]);
			m_threadpool[i].start();
		}
		
		for(Thread thread:m_threadpool){
			try{
				thread.join();
			}catch(InterruptedException e) {
				e.printStackTrace();
			}
		}
		
	}

	protected void updateParameter(int iter, File weightIterFolder) {
		initialAlphaBeta();
		updateAlpha();

		updateBeta();

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
		saveParameter2File(weightIterFolder, fileName);

	}
}