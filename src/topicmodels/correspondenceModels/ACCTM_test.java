package topicmodels.correspondenceModels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

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


public class ACCTM_test extends ACCTM {
	public ACCTM_test(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);
	}
	
	public String toString() {
		return String.format("ACCTM [k:%d, alpha:%.2f, beta:%.2f, training proportion:%.2f, Gibbs Sampling]",
						number_of_topics, d_alpha, d_beta,
						m_testWord4PerplexityProportion);
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
			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						topK);
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

	public void debugOutput(int topK, String filePrefix) {

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
		}

		File parentPhiFolder = new File(filePrefix + "parentPhi");
		File childPhiFolder = new File(filePrefix + "childPhi");
		if (!parentPhiFolder.exists()) {
			System.out.println("creating directory" + parentPhiFolder);
			parentPhiFolder.mkdir();
		}
		if (!childPhiFolder.exists()) {
			System.out.println("creating directory" + childPhiFolder);
			childPhiFolder.mkdir();
		}

		File childXFolder = new File(filePrefix + "xValue");
		if (!childXFolder.exists()) {
			System.out.println("creating x Value directory" + childXFolder);
			childXFolder.mkdir();
		}

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc) {
				printParentTopicAssignment((_ParentDoc) d, parentTopicFolder);
				printParentPhi((_ParentDoc) d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment(d, childTopicFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile, m_trainSet);

		String similarityFile = filePrefix + "topicSimilarity.txt";

		printEntropy(filePrefix);

		printTopKChild4Stn(filePrefix, topK);

		String childMuFile = filePrefix + "childMu.txt";
		printMu(childMuFile);
	}

	protected void printParentTopicAssignment(_Doc d, File topicFolder) {
		// System.out.println("printing topic assignment parent documents");
		_ParentDoc pDoc = (_ParentDoc) d;
		String topicAssignmentFile = pDoc.getName() + ".txt";
		try {

			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));

			for (_Stn stnObj : pDoc.getSentences()) {
				pw.print(stnObj.getIndex() + "\t");
				for (_Word w : stnObj.getWords()) {
					int index = w.getIndex();
					int topic = w.getTopic();
					String featureName = m_corpus.getFeature(index);
					// System.out.println("test\t"+featureName+"\tdocName\t"+d.getName());
					pw.print(featureName + ":" + topic + "\t");
				}
				pw.println();
			}

			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	protected void printChildTopicAssignment(_Doc d, File topicFolder) {
		// System.out.println("printing topic assignment parent documents");

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
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	protected void printParentPhi(_Doc d, File phiFolder) {
		_ParentDoc pDoc = (_ParentDoc) d;
		String parentPhiFileName = pDoc.getName() + ".txt";
		_SparseFeature[] fv = pDoc.getSparse();

		try {
			PrintWriter parentPW = new PrintWriter(new File(phiFolder,
					parentPhiFileName));

			for (int n = 0; n < fv.length; n++) {
				int index = fv[n].getIndex();
				String featureName = m_corpus.getFeature(index);
				parentPW.print(featureName + ":\t");
				for (int k = 0; k < number_of_topics; k++)
					parentPW.print(pDoc.m_phi[n][k] + "\t");
				parentPW.println();
			}
			parentPW.flush();
			parentPW.close();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	protected void printEntropy(String filePrefix) {
		String entropyFile = filePrefix + "entropy.txt";
		boolean logScale = true;

		try {
			PrintWriter entropyPW = new PrintWriter(new File(entropyFile));

			for (_Doc d : m_trainSet) {
				double entropyValue = 0.0;
				entropyValue = Utils.entropy(d.m_topics, logScale);
				entropyPW.print(d.getName() + "\t" + entropyValue);
				entropyPW.println();
			}
			entropyPW.flush();
			entropyPW.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	protected void printTopKChild4Stn(String filePrefix, int topK) {
		String topKChild4StnFile = filePrefix + "topChild4Stn.txt";
		try {
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));

			// m_LM.generateReferenceModel();

			for (_Doc d : m_trainSet) {
				if (d instanceof _ParentDoc) {
					_ParentDoc pDoc = (_ParentDoc) d;

					pw.println(pDoc.getName() + "\t" + pDoc.getSenetenceSize());

					for (_Stn stnObj : pDoc.getSentences()) {
						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(
								stnObj, pDoc);

						int i = 0;
						pw.print((stnObj.getIndex() + 1) + "\t");

						for (String childDocName : likelihoodMap.keySet()) {
							// if(i==topK)
							// break;
							pw.print(childDocName);
							pw.print(":" + likelihoodMap.get(childDocName));
							pw.print("\t");

							i++;
						}
						pw.println();

					}
				}
			}
			pw.flush();
			pw.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj,
			_ParentDoc pDoc) {

		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		for (_ChildDoc cDoc : pDoc.m_childDocs) {

			double stnLogLikelihood = 0;
			for (_Word w : stnObj.getWords()) {
				int wid = w.getIndex();

				double wordLogLikelihood = 0;

				for (int k = 0; k < number_of_topics; k++) {

					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
							* childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}

				stnLogLikelihood += Math.log(wordLogLikelihood);
			}

			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}

		return childLikelihoodMap;
	}

	public void printParameter(String parentParameterFile,
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
				if (d instanceof _ParentDoc) {
					parentParaOut.print(d.getName() + "\t");
					parentParaOut.print("topicProportion\t");
					for (int k = 0; k < number_of_topics; k++) {
						parentParaOut.print(d.m_topics[k] + "\t");
					}

					for (_Stn stnObj : d.getSentences()) {
						parentParaOut.print("sentence"
								+ (stnObj.getIndex() + 1) + "\t");
						for (int k = 0; k < number_of_topics; k++) {
							parentParaOut.print(stnObj.m_topics[k] + "\t");
						}
					}

					parentParaOut.println();

					for (_ChildDoc cDoc : ((_ParentDoc) d).m_childDocs) {

						childParaOut.print(d.getName() + "\t");

						childParaOut.print(cDoc.getName() + "\t");

						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(cDoc.m_topics + "\t");
						}

						childParaOut.println();
					}
				}
			}

			parentParaOut.flush();
			parentParaOut.close();

			childParaOut.flush();
			childParaOut.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	protected void printMu(String childMuFile) {
		System.out.println("print mu");
		try {
			PrintWriter muPW = new PrintWriter(new File(childMuFile));

			for (_Doc d : m_trainSet) {
				if (d instanceof _ChildDoc) {
					muPW.println(d.getName() + "\t" + ((_ChildDoc) d).getMu());
				}

			}
			muPW.flush();
			muPW.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
