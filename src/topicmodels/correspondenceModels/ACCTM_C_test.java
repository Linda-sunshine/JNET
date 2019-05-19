package topicmodels.correspondenceModels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_C_test extends ACCTM_C {
	public ACCTM_C_test(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, gamma);
		
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

		File childLocalWordTopicFolder = new File(filePrefix
				+ "childLocalTopic");

		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
		}
		if (!childLocalWordTopicFolder.exists()) {
			System.out
					.println("creating directory" + childLocalWordTopicFolder);
			childLocalWordTopicFolder.mkdir();
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
				printParentTopicAssignment(d, parentTopicFolder);
				printParentPhi(d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment(d, childTopicFolder);
				printChildLocalWordTopicDistribution((_ChildDoc4BaseWithPhi) d,
						childLocalWordTopicFolder);
				printXValue(d, childXFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile, m_trainSet);

		String xProportionFile = filePrefix + "childXProportion.txt";
		printXProportion(xProportionFile, m_trainSet);

		String similarityFile = filePrefix + "topicSimilarity.txt";

		printEntropy(filePrefix);

		int topKStn = 10;
		int topKChild = 10;
		printTopKChild4Stn(filePrefix, topKChild);


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

	
	protected void generateLanguageModel() {
		double totalWord = 0;

		for (_Doc d : m_corpus.getCollection()) {
			if (d instanceof _ParentDoc)
				continue;
			_SparseFeature[] fv = d.getSparse();
			for (int i = 0; i < fv.length; i++) {
				int wid = fv[i].getIndex();
				double val = fv[i].getValue();

				totalWord += val;
				if (m_wordSstat.containsKey(wid)) {
					double oldVal = m_wordSstat.get(wid);
					m_wordSstat.put(wid, oldVal + val);
				} else {
					m_wordSstat.put(wid, val);
				}
			}
		}

		for (int wid : m_wordSstat.keySet()) {
			double val = m_wordSstat.get(wid);
			double prob = val / totalWord;
			m_wordSstat.put(wid, prob);
		}
	}
	
	protected HashMap<Integer, Double> rankStn4ChildBySim(_ParentDoc pDoc,
			_ChildDoc cDoc) {
		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();

		for (_Stn stnObj : pDoc.getSentences()) {
			double stnKL = Utils.klDivergence(cDoc.m_xTopics[0],
					stnObj.m_topics);

			stnSimMap.put(stnObj.getIndex() + 1, -stnKL);
		}
		
		return stnSimMap;
	}

	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj,
			_ParentDoc pDoc) {
		double gammaLen = Utils.sumOfArray(m_gamma);
		
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			double cDocTopicSum = Utils.sumOfArray(cDoc.m_xSstat);

			double stnLogLikelihood = 0;
			for (_Word w : stnObj.getWords()) {
				int wid = w.getIndex();
				
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
//					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
//							* childTopicInDocProb(k, cDoc)
//							* childXInDocProb(0, cDoc)
//							/ (gammaLen + cDocTopicSum);
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
							* childTopicInDoc(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected void printChildLocalWordTopicDistribution(
			_ChildDoc4BaseWithPhi d,
			File childLocalTopicDistriFolder) {

		String childLocalTopicDistriFile = d.getName() + ".txt";
		try {
			PrintWriter childOut = new PrintWriter(new File(
					childLocalTopicDistriFolder, childLocalTopicDistriFile));

			for (int wid = 0; wid < this.vocabulary_size; wid++) {
				String featureName = m_corpus.getFeature(wid);
				double wordTopicProb = d.m_xTopics[1][wid];
				if (wordTopicProb > 0.001)
					childOut.format("%s:%.3f\t", featureName, wordTopicProb);
			}
			childOut.flush();
			childOut.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void printXProportion(String xProportionFile, ArrayList<_Doc> docList) {
		System.out.println("x proportion for parent doc");
		try {
			PrintWriter pw = new PrintWriter(new File(xProportionFile));
			for (_Doc d : docList) {
				if (d instanceof _ParentDoc) {
					for (_ChildDoc doc : ((_ParentDoc) d).m_childDocs) {
						_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) doc;
						pw.print(d.getName() + "\t");
						pw.print(cDoc.getName() + "\t");
						pw.print(cDoc.m_xProportion[0] + "\t");
						pw.print(cDoc.m_xProportion[1]);
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
							childParaOut.print(cDoc.m_xTopics[0][k] + "\t");
						}

						childParaOut.print("xProportion\t");
						for (int x = 0; x < m_gamma.length; x++) {
							childParaOut.print(cDoc.m_xProportion[x] + "\t");
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

	protected void printXValue(_Doc d, File childXFolder) {
		String XValueFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(childXFolder, XValueFile));

			for (_Word w : d.getWords()) {
				int index = w.getIndex();
				int x = w.getX();
				double xProb = w.getXProb();
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + x + ":" + xProb + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
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

}
