package topicmodels.correspondenceModels;

import structures._Corpus;

public class ACCTM_CZ_test extends ACCTM_CZ {
	public ACCTM_CZ_test(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, gamma);
	}

	public String toString() {
		return String
				.format("ACCTM_CZ [k:%d, alpha:%.2f, beta:%.2f, training proportion:%.2f, Gibbs Sampling]",
						number_of_topics, d_alpha, d_beta,
						m_testWord4PerplexityProportion);
	}

	public void printTopWords(int k, String betaFile) {

	}
}
