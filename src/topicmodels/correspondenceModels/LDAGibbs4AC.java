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
import topicmodels.LDA.LDA_Gibbs;
import utils.Utils;

public class LDAGibbs4AC extends LDA_Gibbs {
	public LDAGibbs4AC(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag) {
		super( number_of_iteration,  converge,  beta, c, lambda, number_of_topics,  alpha,  burnIn,  lag);
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		createSpace();
		for (int i = 0; i < number_of_topics; i++) {
			Arrays.fill(topic_term_probabilty[i], 0);
			Arrays.fill(word_topic_sstat[i], d_beta);
		}
		Arrays.fill(m_sstat, d_beta * vocabulary_size);

		for (_Doc d : collection) {
			if (d instanceof _ParentDoc) {
				for (_Stn stnObj : d.getSentences()) {
					stnObj.setTopicsVct(number_of_topics);
				}
				d.setTopics4Gibbs(number_of_topics, d_alpha);
			} else if (d instanceof _ChildDoc) {
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, d_alpha);
			}

			for (_Word w : d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()]++;
				m_sstat[w.getTopic()]++;
			}
		}

		imposePrior();
	}

	protected void collectStats(_Doc d) {
		if (d instanceof _ParentDoc) {
			collectParentStats(d);
		}else{
			collectChildStats(d);
		}
	}

	protected void collectParentStats(_Doc d) {
		_ParentDoc pDoc = (_ParentDoc) d;
		for (int k = 0; k < this.number_of_topics; k++)
			pDoc.m_topics[k] += pDoc.m_sstat[k];
		pDoc.collectTopicWordStat();
	}
	
	protected void collectChildStats(_Doc d) {
		_ChildDoc cDoc = (_ChildDoc) d;
		for (int k = 0; k < this.number_of_topics; k++)
			cDoc.m_topics[k] += cDoc.m_sstat[k];
	}
	
	protected void estThetaInDoc(_Doc d) {
		super.estThetaInDoc(d);
		if (d instanceof _ParentDoc) {
			// estParentStnTopicProportion((_ParentDoc)d);
			((_ParentDoc) d).estStnTheta();
		}
	}

	public void crossValidation(int k) {
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();

		double[] perf = null;

		_Corpus parentCorpus = new _Corpus();
		ArrayList<_Doc> docs = m_corpus.getCollection();
		ArrayList<_ParentDoc> parentDocs = new ArrayList<_ParentDoc>();
		for (_Doc d : docs) {
			if (d instanceof _ParentDoc) {
				parentCorpus.addDoc(d);
				parentDocs.add((_ParentDoc) d);
			}
		}

		System.out.println("size of parent docs\t" + parentDocs.size());

		parentCorpus.setMasks();
		if (m_randomFold == true) {
			perf = new double[k];
			parentCorpus.shuffle(k);
			int[] masks = parentCorpus.getMasks();

			for (int i = 0; i < k; i++) {
				for (int j = 0; j < masks.length; j++) {
					if (masks[j] == i) {
						m_testSet.add(parentDocs.get(j));
					} else {
						m_trainSet.add(parentDocs.get(j));
						for (_ChildDoc d : parentDocs.get(j).m_childDocs) {
							m_trainSet.add(d);
						}
					}

				}

				// writeFile(i, m_trainSet, m_testSet);
				System.out.println("Fold number " + i);
				infoWriter.println("Fold number " + i);

				System.out.println("Train Set Size " + m_trainSet.size());
				infoWriter.println("Train Set Size " + m_trainSet.size());

				System.out.println("Test Set Size " + m_testSet.size());
				infoWriter.println("Test Set Size " + m_testSet.size());

				long start = System.currentTimeMillis();
				EM();
				perf[i] = Evaluation(i);

				System.out.format(
						"%s Train/Test finished in %.2f seconds...\n",
						this.toString(),
						(System.currentTimeMillis() - start) / 1000.0);
				infoWriter.format(
						"%s Train/Test finished in %.2f seconds...\n",
						this.toString(),
						(System.currentTimeMillis() - start) / 1000.0);

				if (i < k - 1) {
					m_trainSet.clear();
					m_testSet.clear();
				}
			}

		}
		double mean = Utils.sumOfArray(perf) / k, var = 0;
		for (int i = 0; i < perf.length; i++)
			var += (perf[i] - mean) * (perf[i] - mean);
		var = Math.sqrt(var / k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
		infoWriter.format("Perplexity %.3f+/-%.3f\n", mean, var);
	}

	public double Evaluation(int i) {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, totalWords = 0, sumLikelihood = 0;

		System.out.println("In Normal");

		for (_Doc d : m_testSet) {
			loglikelihood = inference(d);
			sumLikelihood += loglikelihood;
			perplexity += loglikelihood;
			totalWords += d.getDocTestLength();
			for (_ChildDoc cDoc : ((_ParentDoc) d).m_childDocs) {
				totalWords += cDoc.getDocTestLength();
			}
		}
		System.out.println("total Words\t" + totalWords + "perplexity\t"
				+ perplexity);
		infoWriter.println("total Words\t" + totalWords + "perplexity\t"
				+ perplexity);
		perplexity /= totalWords;
		perplexity = Math.exp(-perplexity);
		sumLikelihood /= m_testSet.size();

		System.out.format(
				"Test set perplexity is %.3f and log-likelihood is %.3f\n",
				perplexity, sumLikelihood);
		infoWriter.format(
				"Test set perplexity is %.3f and log-likelihood is %.3f\n",
				perplexity, sumLikelihood);
		return perplexity;
	}

	public double inference(_Doc pDoc) {
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();

		initTest(sampleTestSet, pDoc);
		
		double logLikelihood = 0.0, count = 0;
		
		logLikelihood = inference4Doc(sampleTestSet);

		return logLikelihood;
	}
	
	protected double inference4Doc(ArrayList<_Doc> sampleTestSet) {
		double logLikelihood = 0, count = 0;
		int iter = 0;
		do{
			int t;
			
			_Doc tempDoc;
			for(int i=sampleTestSet.size()-1; i>1; i--){
				t = m_rand.nextInt(i);
				
				tempDoc = sampleTestSet.get(i);
				sampleTestSet.set(i, sampleTestSet.get(t));
				sampleTestSet.set(t, tempDoc);
			}
			
			for(_Doc d:sampleTestSet)
				calculate_E_step(d);
			
			if(iter>m_burnIn && iter%m_lag==0){
				for(_Doc d:sampleTestSet){
					collectStats(d);
				}
			}
			
		}while(++iter<number_of_iteration);
		
		for(_Doc d:sampleTestSet){
			estThetaInDoc(d);
			if(d instanceof _ChildDoc){
				logLikelihood += cal_logLikelihood_partial4Child((_ChildDoc)d);
			}
			
		}
			
		return logLikelihood;
	}

	protected double cal_logLikelihood_partial(_Doc d) {
		if (d instanceof _ParentDoc)
			return cal_logLikelihood_partial4Parent(d);
		else
			return cal_logLikelihood_partial4Child(d);
	}

	protected double cal_logLikelihood_partial4Parent(_Doc d) {
		double docLogLikelihood = 0.0;

		for (_Word w : d.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = d.m_topics[k]
						* topic_term_probabilty[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}

	protected double cal_logLikelihood_partial4Child(_Doc d) {
		double docLogLikelihood = 0.0;

		for (_Word w : d.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = d.m_topics[k]
						* topic_term_probabilty[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}

	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d) {

		_ParentDoc pDoc = (_ParentDoc) d;
		for (_Stn stnObj : pDoc.getSentences()) {
			stnObj.setTopicsVct(number_of_topics);
		}

		int testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
		sampleTestSet.add(pDoc);

		pDoc.createSparseVct4Infer();

		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			testLength = (int) (m_testWord4PerplexityProportion * cDoc
					.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();

		}
	}

	protected double calculate_log_likelihood() {
		double logLikelihood = 0.0;

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc)
				logLikelihood += calculate_log_likelihood4Parent(d);
			else
				logLikelihood += calculate_log_likelihood4Child(d);
		}

		return logLikelihood;
	}

	protected double calculate_log_likelihood4Parent(_Doc d) {
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();

		double topicSum = Utils.sumOfArray(d.m_sstat);
		double alphaSum = number_of_topics * d_alpha;

		for (int j = 0; j < fv.length; j++) {
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {

				double wordPerTopicLikelihood = wordByTopicProb(k, wid)
						* topicInDocProb(k, d) / (topicSum + alphaSum);
				wordLogLikelihood += wordPerTopicLikelihood;

			}
			if (wordLogLikelihood < 1e-10) {
				wordLogLikelihood += 1e-10;
				System.out.println("small log likelihood per word");
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);

			docLogLikelihood += value * wordLogLikelihood;
		}

		return docLogLikelihood;
	}

	protected double calculate_log_likelihood4Child(_Doc d) {
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();

		double topicSum = Utils.sumOfArray(d.m_sstat);
		double alphaSum = number_of_topics * d_alpha;

		for (int j = 0; j < fv.length; j++) {
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {

				double wordPerTopicLikelihood = wordByTopicProb(k, wid)
						* topicInDocProb(k, d) / (topicSum + alphaSum);
				wordLogLikelihood += wordPerTopicLikelihood;

			}
			if (wordLogLikelihood < 1e-10) {
				wordLogLikelihood += 1e-10;
				System.out.println("small log likelihood per word");
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);

			docLogLikelihood += value * wordLogLikelihood;
		}

		return docLogLikelihood;
	}

	public void printTopWords(int k, String betaFile) {

		double loglikelihood = calculate_log_likelihood();
		System.out.format("Final Log Likelihood %.3f\t", loglikelihood);

		String filePrefix = betaFile.replace("topWords.txt", "");
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}

		Utils.L1Normalization(m_sstat);

		try {
			System.out.println("beta file");
			PrintWriter betaOut = new PrintWriter(new File(betaFile));
			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							topic_term_probabilty[i][j]));

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

			betaOut.flush();
			betaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}
}
