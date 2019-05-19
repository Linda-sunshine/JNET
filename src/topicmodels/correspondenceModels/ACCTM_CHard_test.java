package topicmodels.correspondenceModels;

import structures._Corpus;

public class ACCTM_CHard_test extends ACCTM_CHard {
	public ACCTM_CHard_test(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, gamma);
	}

	public void printTopWords(int k, String betaFile) {

	}

}
