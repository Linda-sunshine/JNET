package topicmodels.LDA;

import structures._Corpus;
import structures._Doc;
import structures._Word;
import topicmodels.multithreads.TopicModelWorker;
import utils.Utils;

public class LDA_Gibbs_test extends LDA_Gibbs {
	public LDA_Gibbs_test(int number_of_iteration, double converge,
			double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

	}

	public double inference(_Doc d) {

		initTest(d);

		double logLikelihood = 0;
		logLikelihood = inferenceDoc(d);

		return logLikelihood;

	}

	public double Evaluation() {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, log2 = Math.log(2.0), sumLikelihood = 0;
		double totalWords = 0.0;
		if (m_multithread) {
			multithread_inference();
			System.out.println("In thread");
			for (TopicModelWorker worker : m_workers) {
				sumLikelihood += worker.getLogLikelihood();
				perplexity += worker.getPerplexity();
			}
		} else {

			System.out.println("In Normal");
			for (_Doc d : m_testSet) {
				loglikelihood = inference(d);
				sumLikelihood += loglikelihood;
				perplexity += loglikelihood;
				totalWords += d.getDocTestLength();
				// perplexity += Math.pow(2.0,
				// -loglikelihood/d.getTotalDocLength() / log2);
			}

		}
		// perplexity /= m_testSet.size();
		perplexity /= totalWords;
		perplexity = Math.exp(-perplexity);
		sumLikelihood /= m_testSet.size();

		System.out.format(
				"Test set perplexity is %.3f and log-likelihood is %.3f\n",
				perplexity, sumLikelihood);

		return perplexity;
	}

	protected void initTest(_Doc d) {

		int testLength = (int) (m_testWord4PerplexityProportion * d
				.getTotalDocLength());
		d.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
	}

	protected double inferenceDoc(_Doc d) {
		double likelihood = 0;

		int i = 0;
		do {
			calculate_E_step(d);

			if (i < m_burnIn && i % m_lag == 0)
				collectStats(d);
		} while (++i < number_of_iteration);

		estThetaInDoc(d);

		likelihood = cal_logLikelihood_partial(d);

		return likelihood;
	}

	protected double cal_logLikelihood_partial(_Doc d) {
		double docLogLikelihood = 0;

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

	protected double calculate_log_likelihood(_Doc d) {
		int wid = 0;

		double docLogLikelihood = 0;
		
		double docTopicSum = Utils.sumOfArray(d.m_sstat);
		
		for (_Word w : d.getWords()) {
			wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				wordLogLikelihood += wordByTopicProb(k, wid)
						* topicInDocProb(k, d)
						/ (docTopicSum + number_of_topics * d_alpha);
			}
			wordLogLikelihood = Math.log(wordLogLikelihood);

			docLogLikelihood += wordLogLikelihood;
		}

		return docLogLikelihood;
	}
	
	protected double calculate_log_likelihood() {
		double logLikelihood = 0;
		for (_Doc d : m_trainSet) {
			logLikelihood += calculate_log_likelihood(d);
		}

		return logLikelihood;
	}
}
