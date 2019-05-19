package topicmodels.correspondenceModels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._RankItem;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class DCMCorrLDA extends DCMLDA4AC {
	
	protected double[] m_alpha_c;
	protected double m_totalAlpha_c;
	protected double[] m_alphaAuxilary;
	
	public DCMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha_a,
			double alpha_c, double burnIn, double ksi, double tau, int lag,
			int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha_c, burnIn, lag, ksi, tau, newtonIter, newtonConverge);
		

	}
	
	public String toString(){
		return String.format("DCMCorrLDA[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	protected void initialize_probability(Collection<_Doc>collection){
		m_alpha_c = new double[number_of_topics];
		m_alphaAuxilary = new double[number_of_topics];
		
		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalAlpha_c = 0;
		m_totalBeta = new double[number_of_topics];
		
		m_topic_word_prob = new double[number_of_topics][vocabulary_size];

		for(_Doc d:collection){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size);
				
				for (_Stn stnObj : d.getSentences()) {
					stnObj.setTopicsVct(number_of_topics);
				}

				for(_ChildDoc cDoc: pDoc.m_childDocs){
					cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
					for(_Word w:cDoc.getWords()){
						int wid = w.getIndex();
						int tid = w.getTopic();
						
						pDoc.m_wordTopic_stat[tid][wid] ++;
						pDoc.m_topic_stat[tid]++;
					}
					computeMu4Doc(cDoc);
				}
			}
			
		}
		
		initialAlphaBeta();
		imposePrior();
	}
	
	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine(tempParent.getSparse(), d.getSparse());
		mu = 0.5;
		d.setMu(mu);
	}
	
	protected void computeTestMu4Doc(_ChildDoc d){
		_ParentDoc pDoc = d.m_parentDoc;
		
		double mu = Utils.cosine(d.getSparseVct4Infer(), pDoc.getSparseVct4Infer());
		mu = 0.05;
		d.setMu(mu);
	}
	
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	protected void sampleInParentDoc(_Doc d){
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
		int wid, tid;
		double normalizedProb;
		
		for (_Word w : pDoc.getWords()) {
			tid = w.getTopic();
			wid = w.getIndex();
			
			pDoc.m_sstat[tid]--;
			pDoc.m_topic_stat[tid]--;
			pDoc.m_wordTopic_stat[tid][wid]--;
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid, pDoc);
				double pTopicPDoc = parentTopicInDocProb(tid, pDoc);
				double pTopicCDoc = parentChildInfluenceProb(tid, pDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==number_of_topics)
				tid --;
			
			w.setTopic(tid);
			pDoc.m_sstat[tid]++;
			pDoc.m_topic_stat[tid]++;
			pDoc.m_wordTopic_stat[tid][wid]++;
		}
		
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		int wid, tid;
		double normalizedProb;
		
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d.m_parentDoc;

		for(_Word w:d.getWords()){
			tid = w.getTopic();
			wid = w.getIndex();
			
			pDoc.m_wordTopic_stat[tid][wid]--;
			pDoc.m_topic_stat[tid] --;
			d.m_sstat[tid] --;

			normalizedProb = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				double pWordTopic = childWordByTopicProb(tid, wid, pDoc);
				double pTopic = childTopicInDocProb(tid, d, pDoc);
				
				m_topicProbCache[tid] = pWordTopic * pTopic;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for (tid = 0; tid < m_topicProbCache.length; tid++) {
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid--;
			
			w.setTopic(tid);
			d.m_sstat[tid]++;
			pDoc.m_topic_stat[tid]++;
			pDoc.m_wordTopic_stat[tid][wid]++;
		}
	}

	protected double parentWordByTopicProb(int tid, int wid, _ParentDoc4DCM d) {
		double prob = 0;
		prob = (d.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (d.m_topic_stat[tid] + m_totalBeta[tid]);
		
		return prob;
	}
	
	protected double parentTopicInDocProb(int tid, _ParentDoc4DCM d) {
		double prob = 0;
		
		prob = (d.m_sstat[tid] + m_alpha[tid])
				/ (d.getDocInferLength() + m_totalAlpha);

		return prob;
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d) {
		double term = 1.0;

		if (tid == 0)
			return term;
		
		for (_ChildDoc cDoc : d.m_childDocs) {
			double muDp = cDoc.getMu()/d.getTotalDocLength();
			term *= gammaFuncRatio((int) cDoc.m_sstat[tid], muDp, d_alpha
					+ d.m_sstat[tid] * muDp)
					/ gammaFuncRatio((int) cDoc.m_sstat[0], muDp, d_alpha
							+ d.m_sstat[0] * muDp);
		}

		return term;

	}

	protected double gammaFuncRatio(int nc, double muDp, double alphaMuDp) {
		if (nc == 0)
			return 1.0;

		double result = 1.0;
		for (int n = 1; n <= nc; n++) {
			result *= 1 + muDp / (alphaMuDp + n - 1);
		}

		return result;
	}
	
	protected double childWordByTopicProb(int tid, int wid, _ParentDoc4DCM d) {
		double prob = 0;
		prob = (d.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (d.m_topic_stat[tid] + m_totalBeta[tid]);
		return prob;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d, _ParentDoc4DCM pDoc) {
		double prob = 0;
		double childTopicSum = Utils.sumOfArray(d.m_sstat);
		double parentTopicSum = Utils.sumOfArray(pDoc.m_sstat);
			
		double muDp = d.getMu() / parentTopicSum;
		prob = (m_alpha_c[tid] + muDp * pDoc.m_sstat[tid] + d.m_sstat[tid])
				/ (m_totalAlpha_c + muDp * parentTopicSum + childTopicSum);
		
		return prob;
	}
	
	protected void updateParameter(int iter, File weightIterFolder) {
		initialAlphaBeta();
		updateAlpha();
		updateAlphaC();

		for (int k = 0; k < number_of_topics; k++)
			updateBeta(k);

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
		saveParameter2File(weightIterFolder, fileName);

	}

	protected void initialAlphaBeta(){
		
		double parentDocNum = 0;
		double childDocNum = 0;
		
		Arrays.fill(m_sstat, 0);
		Arrays.fill(m_alphaAuxilary, 0);
		for(int k=0; k<number_of_topics; k++){
			Arrays.fill(topic_term_probabilty[k], 0);
			Arrays.fill(word_topic_sstat[k], 0);
		}
		
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				for(int k=0; k<number_of_topics; k++){
					double tempProb = pDoc.m_sstat[k]/pDoc.getTotalDocLength();
					m_sstat[k] += tempProb;
					
					if(pDoc.m_sstat[k] == 0)
						continue;
					for(int v=0; v<vocabulary_size; v++){
						tempProb = pDoc.m_wordTopic_stat[k][v]/pDoc.m_topic_stat[k];
						topic_term_probabilty[k][v] += tempProb;
					}
				}
				parentDocNum += 1;
			
				for(_ChildDoc cDoc:pDoc.m_childDocs){
					for(int k=0; k<number_of_topics; k++){
						double tempProb = cDoc.m_sstat[k]/cDoc.getTotalDocLength();
						m_alphaAuxilary[k] += tempProb;
					}
					childDocNum += 1;
				}
			}
		}
		
		for(int k=0; k<number_of_topics; k++){
			m_sstat[k] /= parentDocNum;
			m_alphaAuxilary[k] /= childDocNum;
			for(int v=0; v<vocabulary_size; v++){
				topic_term_probabilty[k][v] /= (parentDocNum+childDocNum);
			}
		}
		
		for(int k=0; k<number_of_topics; k++){
			m_alpha[k] = m_sstat[k];
			m_alpha_c[k] = m_alphaAuxilary[k];
			for(int v=0; v<vocabulary_size; v++)
				m_beta[k][v] = topic_term_probabilty[k][v]+d_beta;
		}
		
		m_totalAlpha = Utils.sumOfArray(m_alpha);
		m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);
		for(int k=0; k<number_of_topics; k++){
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}
		
	}
	
	protected void updateAlpha(){
		double diff = 0;
		int iteration = 0;
		
		do{
			diff = 0;
			
			double totalAlphaDenominator = 0;
			m_totalAlpha = Utils.sumOfArray(m_alpha);
			double digAlpha = Utils.digamma(m_totalAlpha);
			
			double deltaAlpha = 0;
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					totalAlphaDenominator += Utils.digamma(d.getTotalDocLength()+m_totalAlpha)-digAlpha;
				}
			}
			
			for(int k=0; k<number_of_topics; k++){
				double totalAlphaNumerator = 0;
				
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc)
						totalAlphaNumerator += Utils.digamma(m_alpha[k]+d.m_sstat[k])-Utils.digamma(m_alpha[k]);
				}
				
				deltaAlpha = totalAlphaNumerator*1.0/totalAlphaDenominator;
				
				double newAlpha = m_alpha[k]*deltaAlpha;
				double t_diff = Math.abs(m_alpha[k]-newAlpha);
				
				if(t_diff>diff)
					diff = t_diff;
				
				m_alpha[k] = newAlpha;
			}
			
			iteration ++;
			// System.out.println("alpha parentDoc iteration\t" + iteration);
			
			if(iteration>m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);
		
		// System.out.println("iteration\t"+iteration);
		m_totalAlpha = 0;
		for(int k=0; k<number_of_topics; k++){
			m_totalAlpha += m_alpha[k];
		}
	}
	
	protected void updateAlphaC(){
		double diff = 0;
		int iteration = 0;
		
		do{
			diff = 0;
			double totalAlphaDenominator = 0;
			double[] totalAlphaNumerator = new double[number_of_topics];
			Arrays.fill(totalAlphaNumerator, 0);
			m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);
						
			double deltaAlpha = 0;
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
					
					double pDocLen = pDoc.getTotalDocLength();
					for(_ChildDoc cDoc:pDoc.m_childDocs){
						double muDp = cDoc.getMu()/pDocLen;
						double t_totalAlpha_c = m_totalAlpha_c+cDoc.getMu();
						double digAlpha = Utils.digamma(t_totalAlpha_c);
						totalAlphaDenominator += Utils.digamma(cDoc.getTotalDocLength()+t_totalAlpha_c)-digAlpha;
						
						for(int k=0; k<number_of_topics; k++)
							totalAlphaNumerator[k] += Utils.digamma(m_alpha_c[k]+muDp*pDoc.m_sstat[k]+cDoc.m_sstat[k])-Utils.digamma(m_alpha_c[k]+muDp*pDoc.m_sstat[k]);
					}
				}
			}
			
			for(int k=0; k<number_of_topics; k++){
				deltaAlpha = totalAlphaNumerator[k]*1.0/totalAlphaDenominator;
				
				double newAlpha = m_alpha_c[k]*deltaAlpha;
				double t_diff = Math.abs(m_alpha_c[k]-newAlpha);
				if(t_diff>diff)
					diff = t_diff;
				
				m_alpha_c[k] = newAlpha;
			}
			
			iteration ++;
			// System.out.println("alpha iteration\t" + iteration);
			
			if(iteration > m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);
		
		// System.out.println("iteration\t" + iteration);
		m_totalAlpha_c = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha_c += m_alpha_c[k];
		}
	}
	
	protected void updateBeta(int tid){
		double diff = 0;
		double smoothingBeta = 0.1;
		
		int iteration = 0;
		do{
			diff = 0;
			
			double deltaBeta = 0;
			double wordNum4Tid = 0;
			
			double[] wordNum4Tid4V = new double[vocabulary_size];
			double totalBetaDenominator = 0;
			double[] totalBetaNumerator = new double[vocabulary_size];
			
			Arrays.fill(totalBetaNumerator, 0);
			Arrays.fill(wordNum4Tid4V, 0);
			
			m_totalBeta[tid] = Utils.sumOfArray(m_beta[tid]);
			double digBeta4Tid = Utils.digamma(m_totalBeta[tid]);
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
					totalBetaDenominator += Utils.digamma(m_totalBeta[tid]+pDoc.m_topic_stat[tid])-digBeta4Tid;
					for(int v=0; v<vocabulary_size; v++){
						wordNum4Tid += pDoc.m_wordTopic_stat[tid][v];
						wordNum4Tid4V[v] += pDoc.m_wordTopic_stat[tid][v];
						
						totalBetaNumerator[v] += Utils.digamma(m_beta[tid][v]+pDoc.m_wordTopic_stat[tid][v]);
						totalBetaNumerator[v] -= Utils.digamma(m_beta[tid][v]);
					}
				}
				
			}
			
			for(int v=0; v<vocabulary_size; v++){
				if(wordNum4Tid == 0)
					break;
				if(wordNum4Tid4V[v] == 0){
					deltaBeta = 0;
				}else{
					deltaBeta = totalBetaNumerator[v]/totalBetaDenominator;
				}
					
				double newBeta = m_beta[tid][v]*deltaBeta+d_beta;
				double t_diff = Math.abs(m_beta[tid][v]-newBeta);
				if(t_diff>diff)
					diff = t_diff;
				
				m_beta[tid][v] = newBeta;
			}
			
			iteration ++;
			
			// System.out.println("beta iteration\t"+iteration);
		}while(diff > m_newtonConverge);
		
		// System.out.println("beta iteration\t" + iteration);
	}
	
	protected double calculate_log_likelihood(){
		double logLikelihood = 0.0;
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc)
				logLikelihood += calculate_log_likelihood((_ParentDoc4DCM)d);
		}
		
		return logLikelihood;
	}
	
	protected void collectStats(_Doc d){
		if(d instanceof _ParentDoc4DCM){
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
			for(int k=0; k<number_of_topics; k++){
				pDoc.m_topics[k] += pDoc.m_sstat[k]+m_alpha[k];
				for(int v=0; v<vocabulary_size; v++){
					pDoc.m_wordTopic_prob[k][v] += pDoc.m_wordTopic_stat[k][v]+m_beta[k][v];
				}
			}
		}else if(d instanceof _ChildDoc){
			_ChildDoc cDoc = (_ChildDoc)d;
			_ParentDoc pDoc = cDoc.m_parentDoc;
			double muDp = cDoc.getMu()/pDoc.getTotalDocLength();
			for(int k=0; k<number_of_topics; k++){
				cDoc.m_topics[k] += cDoc.m_sstat[k]+m_alpha_c[k]+muDp*pDoc.m_sstat[k];
			}
		}
	}
	
	protected void saveParameter2File(File fileFolder, String fileName){
		try{
			File paramFile = new File(fileFolder, fileName);
			
			PrintWriter pw = new PrintWriter(paramFile);
			pw.println("alpha");
			
			for(int k=0; k<number_of_topics; k++){
				pw.print(m_alpha[k]+"\t");
			}
			pw.println();
			pw.println("alpha c");
			for(int k=0; k<number_of_topics; k++){
				pw.print(m_alpha_c[k]+"\t");
			}
			pw.println();
			pw.println("beta");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print("topic" + k + "\t");
				for (int v = 0; v < vocabulary_size; v++) {
					pw.print(m_beta[k][v] + "\t");
				}
				pw.println();
			}
			pw.flush();
			pw.close();
		}catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	protected double calculate_log_likelihood(_ParentDoc4DCM d){
		double docLogLikelihood = 0;
		int docID = d.getID();
		
		double parentDocLength = d.getTotalDocLength();
		
		for(int k=0; k<number_of_topics; k++){
			double term = Utils.lgamma(d.m_sstat[k]+m_alpha[k]);
			docLogLikelihood += term;
			
			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;
		}
		
		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(parentDocLength+m_totalAlpha);
		
		for(int k=0; k<number_of_topics; k++){
			for(int v=0; v<vocabulary_size; v++){
				double term = Utils.lgamma(d.m_wordTopic_stat[k][v]+m_beta[k][v]);
				docLogLikelihood += term;
				
				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;
			}
			
			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(d.m_topic_stat[k]+m_totalBeta[k]);
		}
		
		for(_ChildDoc cDoc:d.m_childDocs){
			double muDp = cDoc.getMu()/parentDocLength;
			docLogLikelihood += Utils.digamma(m_totalAlpha_c+cDoc.getMu());
			docLogLikelihood += Utils.digamma(m_totalAlpha_c+cDoc.getMu()+cDoc.getTotalDocLength());
			for(int k=0; k<number_of_topics; k++){
				double term = Utils.digamma(m_alpha_c[k]+muDp*d.m_sstat[k]+cDoc.m_sstat[k]);
				term -= Utils.digamma(m_alpha_c[k]+muDp*d.m_sstat[k]);
				docLogLikelihood += term;
			}
		}
		
		return docLogLikelihood;
	}
	
	protected void estThetaInDoc(_Doc d) {

		if (d instanceof _ParentDoc4DCM) {
			for (int i = 0; i < number_of_topics; i++)
				Utils.L1Normalization(((_ParentDoc4DCM) d).m_wordTopic_prob[i]);
		}
		Utils.L1Normalization(d.m_topics);

	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
		for(_Stn stnObj:pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength, vocabulary_size);
		
		sampleTestSet.add(pDoc);
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
			
			for(_Word w:cDoc.getWords()){
				int wid = w.getIndex();
				int tid = w.getTopic();
				
				pDoc.m_wordTopic_stat[tid][wid] ++;
				pDoc.m_topic_stat[tid] ++;
			}
			
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();
			//cDoc computeMu
			computeTestMu4Doc(cDoc);
		}
		
	}
	
	protected double cal_logLikelihood_partial4Child(_ChildDoc cDoc) {
		double childLikelihood = 0;
		
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)cDoc.m_parentDoc;
		
		for(_Word w:cDoc.getTestWords()){
			int wid= w.getIndex();
			
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = cDoc.m_topics[k]*pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			childLikelihood += Math.log(wordLogLikelihood);
		}
		
		return childLikelihood;
	}
	
	public void printTopWords(int k, String betaFile) {
		try {
			PrintWriter topWordWriter = new PrintWriter(new File(betaFile));

			for (int i = 0; i < m_beta.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_beta[i][j]));

				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}
}

