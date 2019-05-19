package topicmodels.correspondenceModels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._ChildDoc4BaseWithPhi_Hard;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_CHard extends ACCTM_C {
	
	public ACCTM_CHard(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma);
	
		m_topicProbCache = new double[number_of_topics+1];
	}
	
	@Override
	public String toString(){
		return String
				.format("ACCTM_C with hard differentiate[k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopic(number_of_topics);
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				((_ChildDoc4BaseWithPhi_Hard) d).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
				((_ChildDoc4BaseWithPhi_Hard) d).setTopics4Gibbs(number_of_topics, 0);
				computeMu4Doc((_ChildDoc) d);
			}
			
			if(d instanceof _ParentDoc){
				for (_Word w:d.getWords()) {
					word_topic_sstat[w.getTopic()][w.getIndex()]++;
					m_sstat[w.getTopic()]++;
				}	
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}
				}
			}
		}
		
		imposePrior();	
		m_statisticsNormalized = false;
	}

	@Override
	protected void sampleInChildDoc(_Doc d) {
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi)d;
		int wid, tid, xid;
		double normalizedProb;
		
		for(_Word w:cDoc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				cDoc.m_xTopicSstat[xid][tid] --;
				cDoc.m_xSstat[xid] --;
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				cDoc.m_xTopicSstat[xid][wid]--;
				cDoc.m_xSstat[xid] --;
				cDoc.m_childWordSstat --;
			}
			
			_ParentDoc pDocObj = cDoc.m_parentDoc;
			if(Utils.indexOf(pDocObj.getSparse(), wid) != -1){
				normalizedProb = 0;
				for(tid=0; tid<number_of_topics; tid++){
					double pWordTopic = childWordByTopicProb(tid, wid);
					double pTopic = childTopicInDocProb(tid, cDoc);
					
					m_topicProbCache[tid] = pWordTopic*pTopic;
					normalizedProb += m_topicProbCache[tid];
				}
				
				normalizedProb *= m_rand.nextDouble();
				for(tid=0; tid<m_topicProbCache.length; tid++){
					normalizedProb -= m_topicProbCache[tid];
					if(normalizedProb<=0)
						break;
				}
				
				if(tid==m_topicProbCache.length)
					tid --;
				
				if(tid<number_of_topics){
					xid = 0;
					w.setX(xid);
					w.setTopic(tid);
					cDoc.m_xTopicSstat[xid][tid]++;
					cDoc.m_xSstat[xid]++;
					
					if(m_collectCorpusStats){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
	 				}
					
				}else if(tid==(number_of_topics)){
					System.out.println("error on hard differentiate");
					
				}
				
				
			}else{
				normalizedProb = 0;
				double pLambdaZero = childXInDocProb(0, cDoc);
				double pLambdaOne = childXInDocProb(1, cDoc);
				
				for(tid=0; tid<number_of_topics; tid++){
					double pWordTopic = childWordByTopicProb(tid, wid);
					double pTopic = childTopicInDocProb(tid, cDoc);
					
					m_topicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
					normalizedProb += m_topicProbCache[tid];
				}
					
				double pWordTopic = childLocalWordByTopicProb(wid, cDoc);
				m_topicProbCache[tid] = pWordTopic*pLambdaOne;
				normalizedProb += m_topicProbCache[tid];
				
				normalizedProb *= m_rand.nextDouble();
				for(tid=0; tid<m_topicProbCache.length; tid++){
					normalizedProb -= m_topicProbCache[tid];
					if(normalizedProb<=0)
						break;
				}
				
				if(tid==m_topicProbCache.length)
					tid --;
				
				if(tid<number_of_topics){
					xid = 0;
					w.setX(xid);
					w.setTopic(tid);
					cDoc.m_xTopicSstat[xid][tid]++;
					cDoc.m_xSstat[xid]++;
					
					if(m_collectCorpusStats){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
	 				}
					
				}else if(tid==(number_of_topics)){
					xid = 1;
					w.setX(xid);
					w.setTopic(tid);
					cDoc.m_xTopicSstat[xid][wid]++;
					cDoc.m_xSstat[xid]++;
					cDoc.m_childWordSstat ++;
					
				}
				
			}		
			
		}
	}

	@Override
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		int testLength = (int)(m_testWord4PerplexityProportion*d.getTotalDocLength());
		testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);		
		sampleTestSet.add(pDoc);
		pDoc.createSparseVct4Infer();

		for(_ChildDoc cDoc: pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());

			((_ChildDoc4BaseWithPhi_Hard) cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
			((_ChildDoc4BaseWithPhi_Hard)cDoc).setTopics4GibbsTest(number_of_topics, 0, testLength);
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();

			computeTestMu4Doc(cDoc);
		}
	}
	
	@Override
	protected double calculate_log_likelihood4Child(_Doc d) {
//		System.out.println("likelihood in child doc in base with phi");
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gamma);
		double cDocXSum = Utils.sumOfArray(cDoc.m_xSstat);
		
		// prepare compute the normalizers
		_SparseFeature[] fv = cDoc.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();
			
			double wordLogLikelihood = 0;

			if (Utils.indexOf(cDoc.m_parentDoc.getSparse(), wid) != -1) {
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
			}else{
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
							* childTopicInDocProb(k, cDoc)
							* childXInDocProb(0, cDoc) / (cDocXSum + gammaLen);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				double wordPerTopicLikelihood = childLocalWordByTopicProb(wid,
						cDoc)
						* childXInDocProb(1, cDoc)
						/ (cDocXSum + gammaLen);
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
	
	@Override
	protected double cal_logLikelihood_partial4Child(_Doc d) {
		_ChildDoc4BaseWithPhi_Hard cDoc = (_ChildDoc4BaseWithPhi_Hard) d;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gamma);
		double cDocXSum = Utils.sumOfArray(cDoc.m_xSstat);

		for (_Word w : cDoc.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			
			if (Utils.indexOf(cDoc.m_parentDoc.getSparse(), wid) != -1) {
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
			}else{
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
							* childTopicInDocProb(k, cDoc)
							* childXInDocProb(0, cDoc) / (cDocXSum + gammaLen);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				double wordPerTopicLikelihood = childLocalWordByTopicProb(wid,
						cDoc)
						* childXInDocProb(1, cDoc)
						/ (cDocXSum + gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;
			}

			if (Math.abs(wordLogLikelihood) < 1e-10) {
				System.out.println("wordLoglikelihood\t" + wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += wordLogLikelihood;
		}

		return docLogLikelihood;
	}
}
