package topicmodels.correspondenceModels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc4DCM;
import structures._RankItem;
import structures._Word;
import utils.Utils;

public class DCMLDA4AC_test extends DCMLDA4AC {
	public DCMLDA4AC_test(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, int number_of_topics, double alpha, double  burnIn, int lag, double ksi, double tau, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, ksi, tau, newtonIter, newtonConverge);
	}

	public String toString() {
		return String.format(
				"DCMLDA4AC_test[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta);
	}
	
	public void printTopWords(int k, String betaFile) {
		double logLikelihood = calculate_log_likelihood();
		System.out.format("final log likelihood %.3f\t", logLikelihood);

		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(k, filePrefix);

		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		printTopWordsDistribution(k, betaFile);
	}

	public void debugOutput(int topK, String filePrefix) {

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");

		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory\t" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}

		if (!childTopicFolder.exists()) {
			System.out.println("creating directory\t" + childTopicFolder);
			childTopicFolder.mkdir();
		}

		File parentWordTopicDistributionFolder = new File(filePrefix
				+ "wordTopicDistribution");
		if (!parentWordTopicDistributionFolder.exists()) {
			System.out.println("creating word topic distribution folder\t"
					+ parentWordTopicDistributionFolder);
			parentWordTopicDistributionFolder.mkdir();
		}

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc4DCM) {
				printParentTopicAssignment(d, parentTopicFolder);
				printWordTopicDistribution(d,
						parentWordTopicDistributionFolder, topK);
			} else {
				printChildTopicAssignment(d, childTopicFolder);
			}
		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";

		printParameter(parentParameterFile, childParameterFile, m_trainSet);

		String betaFile = filePrefix + "/topBetas.txt";
		printTopBeta(topK, betaFile);

	}

	protected void printTopWordsDistribution(int topK, String topWordFile) {
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}

		Utils.L1Normalization(m_sstat);

		try {
			System.out.println("top word file");
			PrintWriter betaOut = new PrintWriter(new File(topWordFile));
			for (int i = 0; i < m_topic_word_prob.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						topK);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_topic_word_prob[i][j]));

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

	public void printTopBeta(int k, String topWordPath) {

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));

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

	protected void printParentTopicAssignment(_Doc d, File topicFolder) {
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));

			for (_Word w : d.getWords()) {
				int index = w.getIndex();
				int topic = w.getTopic();

				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}

			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	protected void printChildTopicAssignment(_Doc d, File topicFolder) {
		String topicAssignmentFile = d.getName() + ".txt";

		try {
			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));
			for (_Word w : d.getWords()) {
				int wid = w.getIndex();
				int tid = w.getTopic();

				String featureName = m_corpus.getFeature(wid);
				pw.print(featureName + ":" + tid + "\t");
			}
			pw.flush();
			pw.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	protected void printParameter(String parentParameterFile,
			String childParameterFile, ArrayList<_Doc> docList) {
		System.out.println("printing parameter");

		try {
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);

			PrintWriter parentParaOut = new PrintWriter(new File(
					parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(
					childParameterFile));

			for (_Doc d : docList) {
				parentParaOut.print(d.getName() + "\t");
				parentParaOut.print("topicProportion\t");
				for (int k = 0; k < number_of_topics; k++) {
					parentParaOut.print(d.m_topics[k] + "\t");
				}

				parentParaOut.println();
			}

			parentParaOut.flush();
			parentParaOut.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void printWordTopicDistribution(_Doc d,
			File wordTopicDistributionFolder, int k) {
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;

		String wordTopicDistributionFile = pDoc.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(
					wordTopicDistributionFolder, wordTopicDistributionFile));

			for (int i = 0; i < number_of_topics; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int v = 0; v < vocabulary_size; v++) {
					String featureName = m_corpus.getFeature(v);
					double wordProb = pDoc.m_wordTopic_prob[i][v];
					_RankItem ri = new _RankItem(featureName, wordProb);
					fVector.add(ri);
				}

				pw.format("Topic %d(%.5f):\t", i, d.m_topics[i]);
				for (_RankItem it : fVector)
					pw.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				pw.write("\n");
			}

			pw.flush();
			pw.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
