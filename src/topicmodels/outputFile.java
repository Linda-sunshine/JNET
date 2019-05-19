package topicmodels;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.TreeMap;

import Analyzer.ParentChildAnalyzer;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;

public class outputFile {
	public static void outputFiles(String filePrefix, _Corpus c) {
		try {
			String selectedSentencesinParentFile = filePrefix
					+ "/selected_Stn.txt";
			String selectedCommentsFile = filePrefix + "/selected_Comments.txt";

			String sctmFormatParentFile = filePrefix + "/abagf.AT.txt";
			String sctmFormatChildFile = filePrefix + "/cbagf.AT.txt";
			String sctmWordFile = filePrefix + "/words.AT.txt";
			
			String stnLengthFile = filePrefix + "/selected_StnLength.txt";
			
			String shortStnFile = filePrefix + "/selected_ShortStn.txt";
			String longStnFile = filePrefix + "/selected_LongStn.txt";
			
			if (c.getFeatureSize() != 0) {
				PrintWriter wordPW = new PrintWriter(new File(sctmWordFile));
				for (int i = 0; i < c.getFeatureSize(); i++) {
					String wordName = c.getFeature(i);
					wordPW.println(wordName);
				}
				wordPW.flush();
				wordPW.close();
			}
			
			PrintWriter stnLengthPW = new PrintWriter(new File(stnLengthFile));
			
			PrintWriter shortParentPW = new PrintWriter(new File(shortStnFile));
			PrintWriter longParentPW = new PrintWriter(new File(longStnFile));
			
			PrintWriter parentPW = new PrintWriter(new File(
					selectedSentencesinParentFile));
			PrintWriter childPW = new PrintWriter(
					new File(selectedCommentsFile));

			PrintWriter sctmParentPW = new PrintWriter(new File(
					sctmFormatParentFile));
			PrintWriter sctmChildPW = new PrintWriter(new File(
					sctmFormatChildFile));

			int totoalParentNum = 0;
			TreeMap<Integer, _ParentDoc> parentMap = new TreeMap<Integer, _ParentDoc>();

			int totalStnNum = 0;
			
			ArrayList<_Doc> m_trainSet = c.getCollection();
			ArrayList<Integer> parentNameList = new ArrayList<Integer>();
			for (_Doc d : m_trainSet) {
				if (d instanceof _ParentDoc) {
					
					// HashMap<Integer, _Stn> stnMap = ((_ParentDoc)
					// d).m_sentenceMap;
					totoalParentNum += 1;
					
					String parentName = d.getName();
					parentMap.put(Integer.parseInt(parentName), (_ParentDoc) d);

					parentNameList.add(Integer.parseInt(parentName));
				}
			}
			
			ArrayList<Double> parentDocLenList = new ArrayList<Double>();
			ArrayList<Double> childDocLenList = new ArrayList<Double>();
			
			double parentDocLenSum = 0;
			double childDocLenSum = 0;

			for (int parentID : parentMap.keySet()) {
				_ParentDoc parentObj = parentMap.get(parentID);
				
				double parentDocLen = parentObj.getTotalDocLength();
				parentDocLenSum += parentDocLen;
				parentDocLenList.add(parentDocLen);
				
				for(_ChildDoc cDoc: parentObj.m_childDocs){
					double childDocLen = cDoc.getTotalDocLength();
					childDocLenList.add(childDocLen);
					childDocLenSum += childDocLen;
				}
				
				_Stn[] sentenceArray = parentObj.getSentences();
				int selectedStn = 0;
				for (int i = 0; i < sentenceArray.length; i++) {
					_Stn stnObj = sentenceArray[i];
					if(stnObj==null)
						continue;
					
					selectedStn += 1;
					
					
					stnLengthPW.println(stnObj.getLength());
					// if(stnObj==null)
					// continue;
					// selectedStn += 1;
				}
				
				totalStnNum += selectedStn;
				parentPW.print(parentID + "\t" + selectedStn + "\t");
				shortParentPW.print(parentID+"\t");
				longParentPW.print(parentID+"\t");
				for (int i = 0; i < sentenceArray.length; i++) {
					_Stn stnObj = sentenceArray[i];
					if (stnObj == null)
						continue;
					if(stnObj.getLength()<15)
						shortParentPW.print((stnObj.getIndex() + 1) + "\t");
					else
						longParentPW.print((stnObj.getIndex() + 1) + "\t");
					parentPW.print((stnObj.getIndex() + 1) + "\t");
				}
				parentPW.println();
				longParentPW.println();
				shortParentPW.println();
			
			}
			
			System.out.println("longest child\t"+Collections.max(childDocLenList));
			System.out.println("shortest child\t"+Collections.min(childDocLenList));
			
			System.out.println("parent doc len\t"+parentDocLenSum/parentDocLenList.size());
			System.out.println("child doc len\t"+childDocLenSum/childDocLenList.size());
			
			parentPW.flush();
			parentPW.close();
				
			stnLengthPW.flush();
			stnLengthPW.close();
			
			shortParentPW.flush();
			shortParentPW.close();
			
			longParentPW.flush();
			longParentPW.close();
			
			sctmParentPW.println(totoalParentNum);
			sctmChildPW.println(totoalParentNum);
			
			System.out.println("stnNum"+totalStnNum);
			
			for (int parentID: parentMap.keySet()) {
				_ParentDoc d = parentMap.get(parentID);
				// HashMap<Integer, _Stn> stnMap = ((_ParentDoc)
				// d).m_sentenceMap;
				
				_Stn[] sentenceArray = (d).getSentences();
				int selectedStn = 0;

				for (int i = 0; i < sentenceArray.length; i++) {
					_Stn stnObj = sentenceArray[i];
					if (stnObj == null)
						continue;
					selectedStn += 1;
				}

				sctmParentPW.println(selectedStn);
				for (int i = 0; i < sentenceArray.length; i++) {
					_Stn stnObj = sentenceArray[i];
					if (stnObj == null)
						continue;

					_SparseFeature[] sv = stnObj.getFv();
					sctmParentPW.print((int) stnObj.getLength() + "\t");
					for (int j = 0; j < sv.length; j++) {
						int index = sv[j].getIndex();
						double value = sv[j].getValue();
						
						for (int v = 0; v < value; v++)
							sctmParentPW.print(index + "\t");
					}
					sctmParentPW.println();
				}

				ArrayList<_ChildDoc> childDocs = ((_ParentDoc) d).m_childDocs;
				sctmChildPW.println(childDocs.size());

				String parentName = d.getName();
				TreeMap<Integer, _ChildDoc> childMap = new TreeMap<Integer, _ChildDoc>();
				for (_ChildDoc cDoc : childDocs) {
					String childName = cDoc.getName();
					int childID = Integer.parseInt(childName.replace(parentName
							+ "_", ""));
					childMap.put(childID, cDoc);
				}
				
				childPW.print(parentName + "\t");
				for (int t : childMap.keySet()) {
					_ChildDoc cDoc = childMap.get(t);
					sctmChildPW.print((int) cDoc.getTotalDocLength() + "\t");

					childPW.print(cDoc.getName() + "\t");
//					System.out.println(cDoc.getName() + "\t");
					
					_SparseFeature[] fv = cDoc.getSparse();
					for (int j = 0; j < fv.length; j++) {
						int index = fv[j].getIndex();
						double value = fv[j].getValue();
						for (int v = 0; v < value; v++) {
							sctmChildPW.print(index + "\t");
						}
					}
					sctmChildPW.println();

					
				}
				childPW.println();

			}

			sctmParentPW.flush();
			sctmParentPW.close();

			sctmChildPW.flush();
			sctmChildPW.close();
			
			childPW.flush();
			childPW.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void statisticDocLen(_Corpus c){
		
		ArrayList<Double> childDocLenList = new ArrayList<Double>();
		double childDocLenSum = 0;
		
		ArrayList<_Doc> m_trainSet = c.getCollection();
		for (_Doc d : m_trainSet) {
			double childDocLen = d.getTotalDocLength();
			childDocLenList.add(childDocLen);
			childDocLenSum += childDocLen;
		}
				
			
		System.out.println("longest child\t"+Collections.max(childDocLenList));
		System.out.println("shortest child\t"+Collections.min(childDocLenList));
		
//		System.out.println("parent doc len\t"+parentDocLenSum/parentDocLenList.size());
		System.out.println("child doc len\t"+childDocLenSum/childDocLenList.size());
	
	} 

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each document should have at least 2 sentences
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "wsdm"; // 2topic, pLSA, HTMM, LRHTMM, Tensor,
									// LDA_Gibbs, LDA_Variational, HTSM, LRHTSM,
									// ParentChild_Gibbs
	
		String category = "tablet";
		int number_of_topics = 20;
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
		boolean setRandomFold = true; // false means no shuffling and true means shuffling
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
		double converge = -1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 20, number_of_iteration = 50, crossV = 10;
		int gibbs_iteration = 2000, gibbs_lag = 50;
		gibbs_iteration = 10;
		gibbs_lag = 2;
		double burnIn = 0.4;
		boolean display = true, sentence = false;
		
		// most popular items under each category from Amazon
		// needed for docSummary
		String tabletProductList[] = {"B008GFRDL0"};
		String cameraProductList[] = {"B005IHAIMA"};
		String phoneProductList[] = {"B00COYOAYW"};
		String tvProductList[] = {"B0074FGLUM"};
		
		/*****The parameters used in loading files.*****/
		String amazonFolder = "./data/amazon/tablet/topicmodel";
		String newEggFolder = "./data/NewEgg";
		String articleType = "Tech";
		articleType = "Yahoo";
//		articleType = "Gadgets";
//		articleType = "APP";
		String articleFolder = String.format("./data/ParentChildTopicModel/%sArticles", articleType);
		String commentFolder = String.format("./data/ParentChildTopicModel/%sComments", articleType);
		
		articleFolder = String.format("../../Code/Data/TextMiningProject/APPDescriptions");
		commentFolder = String.format("../../Code/Data/TextMiningProject/APPReviews");
		
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		String posModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
			sentence = true;
		}
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_%s.txt", Ngram, articleType);
		//String fvFile = String.format("./data/Features/fv_%dgram_topicmodel.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
	
		String aspectList = "./data/Model/aspect_"+ category + ".txt";
		String aspectSentiList = "./data/Model/aspect_sentiment_"+ category + ".txt";
		
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		File rootFolder = new File("./data/results");
		if(!rootFolder.exists()){
			System.out.println("creating root directory"+rootFolder);
			rootFolder.mkdir();
		}
		
		Calendar today = Calendar.getInstance();
		String filePrefix = String.format("./data/results/%s-%s-%s%s-%s-%s",
				1 + today.get(Calendar.MONTH),
				today.get(Calendar.DAY_OF_MONTH),
				today.get(Calendar.HOUR_OF_DAY), today.get(Calendar.MINUTE),
				topicmodel, articleType);
		
		File resultFolder = new File(filePrefix);
		if (!resultFolder.exists()) {
			System.out.println("creating directory" + resultFolder);
			resultFolder.mkdir();
		}
		
		String infoFilePath = filePrefix + "/Information.txt";
		////store top k words distribution over topic
		String topWordPath = filePrefix + "/topWords.txt";
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.0; // Used in feature selection, the starting point of the features.
		double endProb = 0.95; // Used in feature selection, the ending point of the features.
		int DFthreshold = 3; // Filter the features with DFs smaller than this threshold.

		System.out.println("Performing feature selection, wait...");
		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadParentDirectory(articleFolder, suffix);
//		analyzer.LoadChildDirectory(commentFolder, suffix);
		analyzer.LoadDirectory(commentFolder, suffix);
		
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);	
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.		
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
//		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel, category, 2);
		
		/***** parent child topic model *****/
//		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
//		analyzer.LoadParentDirectory(TechArticlesFolder, suffix);
//		analyzer.LoadChildDirectory(TechCommentsFolder, suffix);
//		analyzer.LoadDirectory(TechArticlesFolder, suffix);
//		analyzer.LoadDirectory(TechCommentsFolder, suffix);

//		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		statisticDocLen(c);
//		outputFiles(filePrefix, c);

	}
}