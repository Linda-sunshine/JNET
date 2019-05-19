package topicmodels;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import Analyzer.ParentChildAnalyzer;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class languageModelBaseLine{
	public HashMap<Integer, Double> m_wordSstat; //TTF
	double m_allWordFrequency; //total TTF
	protected _Corpus m_corpus;
	public double m_smoothingMu;
	double m_allWordFrequencyWithXVal;
	
	public languageModelBaseLine(_Corpus c, double mu){
		m_corpus = c;
		m_wordSstat = new HashMap<Integer, Double>();
		m_allWordFrequency = 0;
		m_smoothingMu = mu;
		m_allWordFrequencyWithXVal = 0;
	}
	
	public void generateReferenceModel(){
		m_allWordFrequency = 0;
		for(_Doc d: m_corpus.getCollection()){
			_SparseFeature[] fv = d.getSparse();
			
			for(int i=0; i<fv.length; i++){
				int wid = fv[i].getIndex();
				double val = fv[i].getValue();
				
				m_allWordFrequency += val;
				if(m_wordSstat.containsKey(wid)){
					double oldVal = m_wordSstat.get(wid);
					m_wordSstat.put(wid, oldVal+val);
				}else{
					m_wordSstat.put(wid, val);
				}
				
			}
		}
		
		for(int wid:m_wordSstat.keySet()){
			double val = m_wordSstat.get(wid);
			double prob = val/m_allWordFrequency;
			m_wordSstat.put(wid, prob);
		}
	}

	protected void generateReferenceModelWithXVal(){
		m_allWordFrequencyWithXVal = 0;
		for(_Doc d: m_corpus.getCollection()){
			if(d instanceof _ParentDoc){
			
				for(_SparseFeature fv:d.getSparse()){
					int wid = fv.getIndex();
					double val = fv.getValue();
					
					m_allWordFrequencyWithXVal += val;
					if(m_wordSstat.containsKey(wid)){
						double oldVal = m_wordSstat.get(wid);
						m_wordSstat.put(wid, oldVal+val);
					}else{
						m_wordSstat.put(wid, val);
					}
				}
			}else{
				double docLenWithXVal = 0;
				
				for(_Word w:d.getWords()){
//					double xProportion = w.getXProb();
					int wid = w.getIndex();
					double val = 0;
//					double val = 1-xProportion;

					
					if (((_ChildDoc) d).m_wordXStat.containsKey(wid)) {
						val = ((_ChildDoc) d).m_wordXStat.get(wid);
					}
					
					docLenWithXVal += val;
					
					m_allWordFrequencyWithXVal += val;
					if(m_wordSstat.containsKey(wid)){
						double oldVal = m_wordSstat.get(wid);
						m_wordSstat.put(wid, oldVal+val);
					}else{
						m_wordSstat.put(wid, val);
					}
				}
				
				((_ChildDoc) d).setChildDocLenWithXVal(docLenWithXVal);
			}
				
		}
		
		for(int wid:m_wordSstat.keySet()){
			double val = m_wordSstat.get(wid);
			double prob = val/m_allWordFrequencyWithXVal;
			m_wordSstat.put(wid, prob);
		}
	}
	
	public double getReferenceProb(int wid) {
		return m_wordSstat.get(wid);
	}
	
	protected void printTopChild4Stn(String filePrefix){
		String topChild4StnFile = filePrefix + "/topChild4Stn.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topChild4StnFile));
			
			for(_Doc d:m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){

//						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
						HashMap<String, Double> likelihoodMap = rankChild4StnByLanguageModel(stnObj, pDoc);

				
//						int i=0;
						pw.print((stnObj.getIndex()+1)+"\t");
						
						for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
//							if(i==topK)
//								break;
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
//							i++;
						}
						pw.println();		
				
					}
				}
			}
			pw.flush();
			pw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected HashMap<String, Double> rankChild4StnByLanguageModel(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		
		double smoothingMu = 1000;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			int cDocLen = cDoc.getTotalDocLength();
			_SparseFeature[] fv = cDoc.getSparse();
			
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord:sv){
				double featureLikelihood = 0;
				
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				double docVal = 0;
				if(featureIndex!=-1){
					docVal = fv[featureIndex].getValue();
				}
				
				double smoothingProb = (1-alphaDoc)*docVal/(cDocLen);
				
				smoothingProb += alphaDoc*getReferenceProb(wid);
				featureLikelihood = Math.log(smoothingProb);
				stnLogLikelihood += stnVal*featureLikelihood;
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();

		for(_ChildDoc cDoc:pDoc.m_childDocs){
			int cDocLen = cDoc.getTotalDocLength();
			_SparseFeature[] fv = cDoc.getSparse();
			
			double stnLogLikelihood = 0;
			double alphaDoc = m_smoothingMu/(m_smoothingMu+cDocLen);

			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord: sv){
				double featureLikelihood = 0;
				
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				if(featureIndex==-1)
					continue;
				
				double docVal = fv[featureIndex].getValue();
				double smoothingProb = docVal/(m_smoothingMu+cDocLen);
				smoothingProb += m_smoothingMu*m_wordSstat.get(wid)/(m_smoothingMu+cDocLen);
				featureLikelihood = Math.log(smoothingProb/(alphaDoc*m_wordSstat.get(wid)));
				stnLogLikelihood += stnVal*featureLikelihood;
			}
			stnLogLikelihood += stnObj.getLength()*Math.log(alphaDoc);	
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;

	}

	protected List<Map.Entry<String, Double>> sortHashMap4String(HashMap<String, Double> stnLikelihoodMap, boolean descendOrder){
		List<Map.Entry<String, Double>> sortList = new ArrayList<Map.Entry<String, Double>>(stnLikelihoodMap.entrySet());
		
		if(descendOrder == true){
			Collections.sort(sortList, new Comparator<Map.Entry<String, Double>>() {
				public int compare(Entry<String, Double> e1, Entry<String, Double> e2){
					return e2.getValue().compareTo(e1.getValue());
				}
			});
		}else{
			Collections.sort(sortList, new Comparator<Map.Entry<String, Double>>() {
				public int compare(Entry<String, Double> e1, Entry<String, Double> e2){
					return e2.getValue().compareTo(e1.getValue());
				}
			});
		}
		
		return sortList;

	}
	
	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		// The way of calculating the feature value, which can also be "TFIDF",
		// "BM25"
		String featureValue = "BM25";
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each document should have at least 2 sentences
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "languageModel"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM, ParentChild_Gibbs
	
		String category = "tablet";
		int number_of_topics = 20;
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
		boolean setRandomFold = false; // false means no shuffling and true means shuffling
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
		double converge = 1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 20, number_of_iteration = 50, crossV = 1;
		int gibbs_iteration = 2000, gibbs_lag = 50;
		gibbs_iteration = 4;
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
//		articleType = "GadgetsArticles";
		String articleFolder = String.format("./data/ParentChildTopicModel/%sArticles", articleType);
		String commentFolder = String.format("./data/ParentChildTopicModel/%sComments", articleType);
		
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
		String filePrefix = String.format("./data/results/%s-%s-%s%s-%s", today.get(Calendar.MONTH), today.get(Calendar.DAY_OF_MONTH), 
						today.get(Calendar.HOUR_OF_DAY), today.get(Calendar.MINUTE), topicmodel);
		
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
		double startProb = 0.5; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 30; // Filter the features with DFs smaller than this threshold.

//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.		
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
//		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel, category, 2);
		
		/***** parent child topic model *****/
		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);

		analyzer.LoadParentDirectory(articleFolder, suffix);
		analyzer.LoadChildDirectory(commentFolder, suffix);
		

//		analyzer.LoadNewEggDirectory(newEggFolder, suffix); //Load all the documents as the data set.
//		analyzer.LoadDirectory(amazonFolder, suffix);				
		
//		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

		double mu = 800;
		languageModelBaseLine lm = new languageModelBaseLine(c, mu);
		lm.generateReferenceModel();
		lm.printTopChild4Stn(filePrefix);
		
		
//		bm25Corr.rankChild4Stn(c, TopChild4StnFile);
//		bm25Corr.rankStn4Child(c, TopStn4ChildFile);
//		bm25Corr.rankChild4Parent(c, TopChild4ParentFile);
//		bm25Corr.discoverSpecificComments(c, similarityFile);
//		String DFFile = filePrefix+"/df.txt";
//		bm25Corr.outputDF(c, DFFile);
	}

}
