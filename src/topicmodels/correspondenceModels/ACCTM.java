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
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM extends corrLDA_Gibbs {
	
	protected double[] m_topicProbCache;
	protected double m_kAlpha;	
	protected boolean m_statisticsNormalized = false;//a warning sign of normalizing statistics before collecting new ones
	
	public ACCTM(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);
		
		m_topicProbCache = new double[number_of_topics];
		m_kAlpha = d_alpha * number_of_topics;
	}
	
	@Override
	public String toString(){
		return String.format("ACCTM [k:%d, alpha:%.2f, beta:%.2f, training proportion:%.2f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta, m_testWord4PerplexityProportion);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct(number_of_topics);				
			} else if(d instanceof _ChildDoc){
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
				computeMu4Doc((_ChildDoc) d);
			}
			
			for (_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()]++;
				m_sstat[w.getTopic()]++;
			}			

		}
	
		imposePrior();
		
		m_statisticsNormalized = false;
	}
	
	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine(tempParent.getSparse(), d.getSparse());
		d.setMu(mu);
	}
	
	protected void computeTestMu4Doc(_ChildDoc d){
		_ParentDoc pDoc = d.m_parentDoc;
		
		double mu = Utils.cosine(d.getSparseVct4Infer(), pDoc.getSparseVct4Infer());
		mu = 1e32;
		d.setMu(mu);
	}

	protected void sampleInParentDoc(_Doc d) {
		_ParentDoc pDoc = (_ParentDoc) d;
		int wid, tid;
		double normalizedProb;
		
		for (_Word w : pDoc.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			
			pDoc.m_sstat[tid]--;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopicPDoc = parentTopicInDocProb(tid, pDoc);
				double pTopicCDoc = parentChildInfluenceProb(tid, pDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid==number_of_topics)
				tid --;
			
			w.setTopic(tid);
			pDoc.m_sstat[tid]++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		double topicSum = Utils.sumOfArray(pDoc.m_sstat);

		for(_ChildDoc cDoc: pDoc.m_childDocs){
			double muDp = cDoc.getMu() / topicSum;
			term *= gammaFuncRatio((int)cDoc.m_sstat[tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp)
					/ gammaFuncRatio((int)cDoc.m_sstat[0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);
		}
		
		return term;
	}
	
	protected double gammaFuncRatio(int nc, double muDp, double alphaMuNp) {
		if (nc==0)
			return 1.0;
		
		double result = 1.0;
		for (int n = 1; n <= nc; n++)
			result *= 1 + muDp / (alphaMuNp + n - 1);
		return result;
	}
	
	protected void sampleInChildDoc(_Doc d) {

		_ChildDoc cDoc = (_ChildDoc) d;
		int wid, tid;
		double normalizedProb;
		
		for (_Word w : cDoc.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			
			cDoc.m_sstat[tid]--;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = childWordByTopicProb(tid, wid);
				double pTopicCDoc = childTopicInDocProb(tid, cDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid == number_of_topics)
				tid --;
			
			w.setTopic(tid);
			cDoc.m_sstat[tid]++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}

	protected double childTopicInDocProb(int tid, _ChildDoc d){

		_ParentDoc pDoc = (_ParentDoc) (d.m_parentDoc);
		double pDocTopicSum = Utils.sumOfArray(pDoc.m_sstat);
		double cDocTopicSum = Utils.sumOfArray(d.m_sstat);

		return (d_alpha + d.getMu() * d.m_parentDoc.m_sstat[tid] / pDocTopicSum + d.m_sstat[tid])
				/ (m_kAlpha + d.getMu() + cDocTopicSum);
	
	}

	protected void collectChildStats(_Doc d) {
		_ChildDoc cDoc = (_ChildDoc) d;
		_ParentDoc pDoc = cDoc.m_parentDoc;
		double pDocTopicSum = Utils.sumOfArray(pDoc.m_sstat);
		for (int k = 0; k < this.number_of_topics; k++) 
			cDoc.m_topics[k] += cDoc.m_sstat[k] + d_alpha + cDoc.getMu()
					* pDoc.m_sstat[k] / pDocTopicSum;

	}
	
	protected void estThetaInDoc(_Doc d){
		Utils.L1Normalization(d.m_topics);
		if(d instanceof _ParentDoc){
			((_ParentDoc)d).estStnTheta();
		}
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
		sampleTestSet.add(pDoc);
		pDoc.createSparseVct4Infer();

		for(_ChildDoc cDoc: pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();

			computeTestMu4Doc(cDoc);
		}
	}

	protected double cal_logLikelihood_partial(_Doc d) {
		if(d instanceof _ParentDoc)
			return cal_logLikelihood_partial4Parent(d);
		else
			return cal_logLikelihood_partial4Child(d);
	}	
	
	@Override
	public void printTopWords(int k, String betaFile) {
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < m_sstat.length; i++) {
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];	
				if (Double.isNaN(d.m_topics[i]))
					System.out.println("nan name\t" + d.getName());
			}
		}
		
		Utils.L1Normalization(m_sstat);
		
		try {
			System.out.println("beta file");
			PrintWriter betaOut = new PrintWriter(new File(betaFile));
			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));

				betaOut.format("Topic %d(%.3f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector) {
					betaOut.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
						m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				betaOut.println();
				System.out.println();
			}
			betaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}
	
	protected double calculate_log_likelihood4Child(_Doc d) {
		_ChildDoc cDoc = (_ChildDoc) d;
		double docLogLikelihood = 0.0;

		// prepare compute the normalizers
		_SparseFeature[] fv = cDoc.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
						* childTopicInDocProb(k, cDoc);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += value * wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
		
}
