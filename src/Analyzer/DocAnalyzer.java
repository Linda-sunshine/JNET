package Analyzer;

import json.JSONArray;
import json.JSONObject;
import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import structures.*;
import utils.Utils;

import java.io.*;
import java.text.Normalizer;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * 
 * @author Lin Gong
 * Specialized analyzer for text documents
 */

public class DocAnalyzer extends Analyzer {
	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected SentenceDetectorME m_stnDetector;
	protected POSTaggerME m_tagger;
	Set<String> m_stopwords;

	protected SimpleDateFormat m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project;
	protected int m_stnSizeThreshold = 2;//minimal size of sentences
	boolean m_newCV = false;// flag to indicate whether we are using the old cv or new one. 

	//shall we have it here???
	protected HashMap<String, Integer> m_posTaggingFeatureNameIndex;//Added by Lin
	protected SentiWordNet m_sentiWordNet;
	
	//Constructor with TokenModel, ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		m_stnDetector = null; // indicating we don't need sentence splitting
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	//Constructor with TokenModel, ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, boolean b) 
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		m_stnDetector = null; // indicating we don't need sentence splitting
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_Ngram = Ngram;
		m_newCV = b;// added by Lin for using different cv.
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	//TokenModel + stnModel.
	public DocAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(classNo, threshold);
		
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();// we will only handle English text documents
		
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	//TokenModel + stnModel + posModel.
	public DocAnalyzer(String tokenModel, String stnModel, String posModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		
		if (posModel!=null)
			m_tagger = new POSTaggerME(new POSModelLoader().load(new File(posModel)));
		else
			m_tagger = null;
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}

	public void setReleaseContent(boolean release) {
		m_releaseContent = release;
	}
	
	public void setMinimumNumberOfSentences(int number){
		m_stnSizeThreshold = number;
	}
	
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				line = SnowballStemming(Normalize(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}	
	
	//Tokenizing input text string
	protected String[] Tokenizer(String source){
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	
	//Normalize.
	protected String Normalize(String token){
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		
		if (Utils.isNumber(token))
			return "NUM";
		else
			return token;
	}
	
	//Snowball Stemmer.
	protected String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return token;
	}
	
	protected boolean isLegit(String token) {
		return !token.isEmpty() 
			&& !m_stopwords.contains(token)
			&& token.length()>1
			&& token.length()<20;
	}
	
	//check if it is a sentence's boundary
	protected boolean isBoundary(String token) {
		return token.isEmpty();//is this a good checking condition?
	}
	
	// added by Lin, the same function with different parameters.
	protected double sentiWordScore(String[] tokens, String[] posTags) {
		double senScore = 0.0;
		double tmp;
		String word, tag;

		for(int i=0; i<tokens.length;i++){
			word = SnowballStemming(Normalize(tokens[i]));
			tag = posTags[i];
			if(tag.equalsIgnoreCase("NN") || tag.equalsIgnoreCase("NNS") || tag.equalsIgnoreCase("NNP") || tag.equalsIgnoreCase("NNPS"))
				tag = "n";
			else if(tag.equalsIgnoreCase("JJ") || tag.equalsIgnoreCase("JJR") || tag.equalsIgnoreCase("JJS"))
				tag = "a";
			else if(tag.equalsIgnoreCase("VB") || tag.equalsIgnoreCase("VBD") || tag.equalsIgnoreCase("VBG"))
				tag = "v";
			else if(tag.equalsIgnoreCase("RB") || tag.equalsIgnoreCase("RBR") || tag.equalsIgnoreCase("RBS"))
				tag = "r";
			
			tmp = m_sentiWordNet.extract(word, tag);
			if(tmp!=-2) // word found in SentiWordNet
				senScore+=tmp;
		}
		return senScore/tokens.length;//This is average, we may have different ways of calculation.
	}
		
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected TokenizeResult TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		TokenizeResult result = new TokenizeResult(tokens);
		
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		LinkedList<String> Ngrams = new LinkedList<String>();
		int tokenLength = tokens.length, N = m_Ngram;			
		
		for(int i=0; i<tokenLength; i++) {
			String token = tokens[i];
			boolean legit = isLegit(token);
			if (legit) 
				Ngrams.add(token);//unigram
			else
				result.incStopwords();
			
			//N to 2 grams
			if (!isBoundary(token)) {
				for(int j=i-1; j>=Math.max(0, i-N+1); j--) {	
					if (isBoundary(tokens[j]))
						break;//touch the boundary
					
					token = tokens[j] + "-" + token;
					legit &= isLegit(tokens[j]);
					if (legit)//at least one of them is legitimate
						Ngrams.add(token);
				}
			}
		}
		
		result.setTokens(Ngrams.toArray(new String[Ngrams.size()]));
		return result;
	}

	//Load a full text review document and analyze it.
	//We will assume the document content is only about the text 
	@Override
	public void LoadDoc(String filename) {
		if (filename.toLowerCase().endsWith(".json"))
			LoadJsonDoc(filename);
		else
			LoadTxtDoc(filename);
	}
	
	protected void LoadTxtDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				buffer.append(line);
			}
			reader.close();
			
			int yLabel = filename.contains("pos") ? 1:0;
			
			//Collect the number of documents in one class as its document id.
			_Doc doc = new _Doc(m_corpus.getSize(), buffer.toString(), yLabel);

			if(this.m_stnDetector!=null)
				AnalyzeDocWithStnSplit(doc);
			else
				AnalyzeDoc(doc);			
			
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	protected JSONObject LoadJSON(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			return new JSONObject(buffer.toString());
		} catch (Exception e) {
			System.out.print('X');
			return null;
		}
	}
	
//	//Load a document and analyze it.
//	protected void LoadJsonDoc(String filename) {
//		_Product prod = null;
//		JSONArray jarray = null;
//		
//		try {
//			JSONObject json = LoadJSON(filename);
//			prod = new _Product(json.getJSONObject("ProductInfo"));
//			jarray = json.getJSONArray("Reviews");
//		} catch (Exception e) {
//			System.err.print('X');//fail to parse a json document
//			return;
//		}	
//		
//		for(int i=0; i<jarray.length(); i++) {
//			try {
//				_Post post = new _Post(jarray.getJSONObject(i));
//				if (post.isValid(m_dateFormatter)) {
//					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
//					String content;
//					
//					//append document title into document content
//					if (Utils.endWithPunct(post.getTitle()))
//						content = post.getTitle() + " " + post.getContent();
//					else
//						content = post.getTitle() + ". " + post.getContent();
//					
//					//int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp
//					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), prod.getID(), post.getTitle(), content, post.getLabel()-1, timeStamp);
//					if(this.m_stnDetector!=null)
//						AnalyzeDocWithStnSplit(review);
//					else
//						AnalyzeDoc(review);
//				}
//			} catch (ParseException e) {
//				System.out.print('T');
//			} catch (JSONException e) {
//				System.out.print('P');
//			}
//		}
//	}
	
	//Load a document and analyze it.
	protected void LoadJsonDoc(String filename) {
		JSONArray jarray = null;
		
		try {
			JSONObject json = LoadJSON(filename);
			jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				_Post post = new _Post(jarray.getJSONObject(i));
				int ylabel = post.getLabel() < 4 ? 0: 1;
				_Doc review = new _Doc(m_corpus.getSize(), post.getContent(), ylabel);
				AnalyzeDoc(review);
			}
		} catch (Exception e) {
			System.err.print('X');//fail to parse a json document
			return;
		}	
		
	}
	
	//convert the input token sequence into a sparse vector (docWordMap cannot be changed)
	protected HashMap<Integer, Double> constructSpVct(String[] tokens, int y, HashMap<Integer, Double> docWordMap) {
		int index = 0;
		double value = 0;
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		
		for (String token : tokens) {//tokens could come from a sentence or a document
			// CV is not loaded, take all the tokens as features.
			if (!m_isCVLoaded) {
				if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						if (docWordMap==null || !docWordMap.containsKey(index)) {
							if(m_featureStat.containsKey(token))
								m_featureStat.get(token).addOneDF(y);
						}
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					if(m_featureStat.containsKey(token))
						m_featureStat.get(token).addOneDF(y);
				}
				if(m_featureStat.containsKey(token))
					m_featureStat.get(token).addOneTTF(y);
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					if (!m_isCVStatLoaded && (docWordMap==null || !docWordMap.containsKey(index)))
						m_featureStat.get(token).addOneDF(y);
				}
				
				if (!m_isCVStatLoaded)
					m_featureStat.get(token).addOneTTF(y);
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		return spVct;
	}
	
	//Added by Lin for constructing pos tagging vectors.
	public HashMap<Integer, Double> constructPOSSpVct(String[] tokens, String[] tags){
		int posIndex = 0;
		double posValue = 0;
		HashMap<Integer, Double> posTaggingVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.	

		for(int i = 0; i < tokens.length; i++){
			if (isLegit(tokens[i])){
				//If the word is adj/adv, construct the sparse vector.
				if(tags[i].equals("RB") || tags[i].equals("RBR") || tags[i].equals("RBS") 
					|| tags[i].equals("JJ") || tags[i].equals("JJR") || tags[i].equals("JJS")) {
					if(m_posTaggingFeatureNameIndex.containsKey(tokens[i])){
						posIndex = m_posTaggingFeatureNameIndex.get(tokens[i]);
						if(posTaggingVct.containsKey(posIndex)){
							posValue = posTaggingVct.get(posIndex) + 1;
							posTaggingVct.put(posIndex, posValue);
						} else
							posTaggingVct.put(posIndex, 1.0);
					} else {
						posIndex = m_posTaggingFeatureNameIndex.size();
						m_posTaggingFeatureNameIndex.put(tokens[i], posIndex);
						posTaggingVct.put(posIndex, 1.0);
					}
				}
			}
		}
		return posTaggingVct;
	}
	
	/*Analyze a document and add the analyzed document back to corpus.*/
//	protected boolean AnalyzeDoc(_Doc doc) {
//		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
//		String[] tokens = result.getTokens();
//		int y = doc.getYLabel();
//		
//		// Construct the sparse vector.
//		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
//		if (spVct.size()>m_lengthThreshold) {
//			doc.createSpVct(spVct);
//			doc.setStopwordProportion(result.getStopwordProportion());
//			
//			m_corpus.addDoc(doc);
//			m_classMemberNo[y]++;
//			if (m_releaseContent)
//				doc.clearSource();
//			return true;
//		} else {
//			/****Roll back here!!******/
//			rollBack(spVct, y);
//			return false;
//		}
//	}
	/*Analyze a document and add the analyzed document back to corpus.*/
	protected boolean AnalyzeDoc(_Doc doc) {
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();
		
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		doc.createSpVct(spVct);
		m_corpus.addDoc(doc);
		m_classMemberNo[y]++;

		return true;
	}
	
	protected boolean AnalyzeDocByStn(_Doc doc, String[] sentences) {
		TokenizeResult result;
		int y = doc.getYLabel(), index = 0;		
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // sparse sentence feature vectors 
		double stopwordCnt = 0, rawCnt = 0;
		
		for(String sentence : sentences) {
			result = TokenizerNormalizeStemmer(sentence);// Three-step analysis.
			HashMap<Integer, Double> sentence_vector = constructSpVct(result.getTokens(), y, spVct);// construct bag-of-word vector based on normalized tokens	

			if (sentence_vector.size()>2) {//avoid empty sentence	
				String[] posTags;
				if(m_tagger==null)
					posTags = null;
				else
					posTags = m_tagger.tag(result.getRawTokens());
				
				stnList.add(new _Stn(index, Utils.createSpVct(sentence_vector), result.getRawTokens(), posTags, sentence));
				Utils.mergeVectors(sentence_vector, spVct);
				
				stopwordCnt += result.getStopwordCnt();
				rawCnt += result.getRawCnt();
			}
			index ++;
		} // End For loop for sentence	
	
		//the document should be long enough
		if (spVct.size()>=m_lengthThreshold && stnList.size()>=m_stnSizeThreshold) { 
			doc.createSpVct(spVct);		
			doc.setStopwordProportion(stopwordCnt/rawCnt);
			doc.setSentences(stnList);
			
			m_corpus.addDoc(doc);
			m_classMemberNo[y] ++;
			
			if (m_releaseContent)
				doc.clearSource();
			return true;
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
			return false;
		}
	}

	// adding sentence splitting function, modified for HTMM
	protected boolean AnalyzeDocWithStnSplit(_Doc doc) {
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());
		return AnalyzeDocByStn(doc, sentences);		
	}
}	

