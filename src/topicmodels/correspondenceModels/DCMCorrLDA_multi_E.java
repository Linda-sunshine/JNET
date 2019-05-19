package topicmodels.correspondenceModels;

import java.io.File;
import java.util.Collection;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._Word;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;
import topicmodels.multithreads.TopicModel_worker.RunType;

public class DCMCorrLDA_multi_E extends DCMCorrLDA{
	public class DCMCorrLDA_worker extends TopicModel_worker{
		protected double[] alphaStat;
		
		public DCMCorrLDA_worker(int number_of_topics, int vocabulary_size){
			super(number_of_topics, vocabulary_size);
			
			alphaStat = new double[number_of_topics];
		}
		
		public double calculate_E_step(_Doc d){
			d.permutation();
			
			if(d instanceof _ParentDoc){
				sampleInParentDoc((_ParentDoc)d);
			}else if(d instanceof _ChildDoc){
				sampleInChildDoc((_ChildDoc)d);
			}
			
			return 0;
		}
		
		protected void sampleInParentDoc(_Doc d){
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
			int wid, tid;
			double normalizedProb;
			
			for(_Word w:pDoc.getWords()){
				tid = w.getTopic();
				wid = w.getIndex();
				
				pDoc.m_sstat[tid] --;
				pDoc.m_topic_stat[tid] --;
				pDoc.m_wordTopic_stat[tid][wid] --;
				
				normalizedProb = 0;
				
				for(tid=0; tid<number_of_topics; tid++){
					double pWordTopic = parentWordByTopicProb(tid, wid, pDoc);
					double pTopicPDoc = parentTopicInDocProb(tid, pDoc);
					double pTopicCDoc = parentChildInfluenceProb(tid, pDoc);
					
					alphaStat[tid] = pWordTopic * pTopicPDoc * pTopicCDoc;
					normalizedProb += alphaStat[tid];
				}
				
				normalizedProb *= m_rand.nextDouble();
				for(tid=0; tid<number_of_topics; tid++){
					normalizedProb -= alphaStat[tid];
					if(normalizedProb <= 0)
						break;
				}
				
				if(tid==number_of_topics)
					tid --;
				
				w.setTopic(tid);
				pDoc.m_sstat[tid] ++;
				pDoc.m_topic_stat[tid] ++;
				pDoc.m_wordTopic_stat[tid][wid] ++;
			}
		}
		
		protected void sampleInChildDoc(_ChildDoc d){
			int wid, tid;
			double normalizedProb;
			
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d.m_parentDoc;
			
			for(_Word w:d.getWords()){
				tid = w.getTopic();
				wid = w.getIndex();
				
				pDoc.m_wordTopic_stat[tid][wid] --;
				pDoc.m_topic_stat[tid] --;
				d.m_sstat[tid] --;
				
				normalizedProb = 0;
				for(tid=0; tid<number_of_topics; tid++){
					double pWordTopic = childWordByTopicProb(tid, wid, pDoc);
					double pTopic = childTopicInDocProb(tid, d, pDoc);
					
					alphaStat[tid] = pWordTopic * pTopic;
					normalizedProb += alphaStat[tid];
				}
				
				normalizedProb *= m_rand.nextDouble();
				for (tid = 0; tid < number_of_topics; tid++) {
					normalizedProb -= alphaStat[tid];
					if (normalizedProb <= 0)
						break;
				}

				if (tid == number_of_topics)
					tid--;

				w.setTopic(tid);
				d.m_sstat[tid]++;
				pDoc.m_topic_stat[tid]++;
				pDoc.m_wordTopic_stat[tid][wid]++;
			}
		}

		@Override
		public double inference(_Doc d) {
			// TODO Auto-generated method stub
			return 0;
		}
		
	} 
	
	public DCMCorrLDA_multi_E(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, 
			double alpha, double alphaC, double burnIn, double ksi, double tau,
			int lag, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, alphaC, burnIn, ksi, tau, lag, newtonIter,
				newtonConverge);
		m_multithread = true;
	}
	
	public String toString(){
		return String.format("multithread DCMCorrLDA[k:%d, alpha:%.2f, beta:%.2f, Variational]", number_of_topics, d_alpha, d_beta);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];
		m_workers = new DCMCorrLDA_worker[cores];
		
		for(int i=0; i<cores; i++)
			m_workers[i] = new DCMCorrLDA_worker(number_of_topics, vocabulary_size);
		
		int workerID = 0;
		for(_Doc d:collection){
			if (d instanceof _ParentDoc) {
				m_workers[workerID % cores].addDoc(d);
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
				for (_ChildDoc cDoc : pDoc.m_childDocs) {
					m_workers[workerID % cores].addDoc(cDoc);
				}
				workerID++;
			}
		}
		
		super.initialize_probability(collection);
		
	}
	
	protected void init(){
		// super.init();
		for(TopicModelWorker worker:m_workers){
			worker.resetStats();
		}
	}
	
	public void EM(){
		System.out.format("Starting %s...\n", toString());

		long starttime = System.currentTimeMillis();

		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);

		String filePrefix = "./data/results/DCMCorrLDA";
		File weightFolder = new File(filePrefix + "");
		if (!weightFolder.exists()) {
			// System.out.println("creating directory for weight"+weightFolder);
			weightFolder.mkdir();
		}

		double delta = 0, last = 0, current = 0;
		int i = 0, displayCount = 0;
		do {

			long eStartTime = System.currentTimeMillis();
			for (int j = 0; j < number_of_iteration; j++) {
				init();
				multithread_E_step();
			}
			long eEndTime = System.currentTimeMillis();

			System.out.println("per iteration e step time\t"
					+ (eEndTime - eStartTime));

			long mStartTime = System.currentTimeMillis();
			calculate_M_step(i, weightFolder);
			long mEndTime = System.currentTimeMillis();

			System.out.println("per iteration m step time\t"
					+ (mEndTime - mStartTime));

			if (m_converge > 0
					|| (m_displayLap > 0 && i % m_displayLap == 0 && displayCount > 6)) {
				// required to display log-likelihood
				current = calculate_log_likelihood();
				// together with corpus-level log-likelihood

				if (i > 0)
					delta = (last - current) / last;
				else
					delta = 1.0;
				last = current;
			}

			if (m_displayLap > 0 && i % m_displayLap == 0) {
				if (m_converge > 0) {
					System.out.format(
							"Likelihood %.3f at step %s converge to %f...\n",
							current, i, delta);
					infoWriter.format(
							"Likelihood %.3f at step %s converge to %f...\n",
							current, i, delta);

				} else {
					System.out.print(".");
					if (displayCount > 6) {
						System.out.format("\t%d:%.3f\n", i, current);
						infoWriter.format("\t%d:%.3f\n", i, current);
					}
					displayCount++;
				}
			}

			if (m_converge > 0 && Math.abs(delta) < m_converge)
				break;// to speed-up, we don't need to compute likelihood in
						// many cases
		} while (++i < this.number_of_iteration);

		finalEst();

		long endtime = System.currentTimeMillis() - starttime;
		System.out
				.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n",
						current, i, delta, endtime / 1000);
		infoWriter
				.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n",
						current, i, delta, endtime / 1000);
	}
	
	protected double multithread_E_step(){
		for(int i=0; i<m_workers.length; i++){
			m_workers[i].setType(RunType.RT_EM);
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		for(Thread thread:m_threadpool){
			try{
				thread.join();
			}catch(InterruptedException e){
				e.printStackTrace();
			}
		}
		return 0;
	}

}