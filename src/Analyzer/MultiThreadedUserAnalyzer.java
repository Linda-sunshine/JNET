package Analyzer;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import structures.TokenizeResult;
import structures._Doc;
import structures._Review;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

/**
 * @author Mohammad Al Boni
 * Multi-threaded extension of UserAnalyzer
 */

public class MultiThreadedUserAnalyzer extends UserAnalyzer {

	protected int m_numberOfCores;
	protected Tokenizer[] m_tokenizerPool;
	protected SnowballStemmer[] m_stemmerPool;
	protected Object m_allocReviewLock = null;
	protected Object m_corpusLock = null;
	protected Object m_rollbackLock = null;
	protected Object m_featureStatLock = null;
	protected Object m_featureNameIndexLock = null;
	protected Object m_featureNamesLock = null;

	protected String m_suffix = null;//filter by suffix
	protected boolean m_allocateFlag = true;

	protected HashMap<String, Integer> m_userIDIndex;

	public MultiThreadedUserAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
		
		m_numberOfCores = numberOfCores;
		
		// since DocAnalyzer already contains a tokenizer, then we can user it and define a pool with length of m_numberOfCores - 1
		m_tokenizerPool = new Tokenizer[m_numberOfCores-1]; 
		m_stemmerPool = new SnowballStemmer[m_numberOfCores-1];
		for(int i=0;i<m_numberOfCores-1;++i){
			m_tokenizerPool[i] = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
			m_stemmerPool[i] = new englishStemmer();
		}
		
		m_allocReviewLock = new Object();// lock when collecting review statistics
		m_corpusLock = new Object(); // lock when collecting class statistics 
		m_rollbackLock = new Object(); // lock when revising corpus statistics
		m_featureStatLock = new Object();
		m_featureNameIndexLock = new Object();
		m_featureNamesLock = new Object();
	}

	// Decide whether we will allocate reviews or not
	public void setAllocateReviewFlag(boolean b){
		m_allocateFlag = b;
	}

	@Override
	public void loadUserDir(String folder){
		if(folder == null || folder.isEmpty())
			return;

		File dir = new File(folder);
		final File[] files=dir.listFiles();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<m_numberOfCores;++i){
			threads.add(  (new Thread() {
				int core;
				@Override
				public void run() {
					try {
						for (int j = 0; j + core <files.length; j += m_numberOfCores) {
							File f = files[j+core];
							if(f.isFile() 
								&& (m_suffix==null || f.getAbsolutePath().endsWith(m_suffix))){//load the user								
								loadUser(f.getAbsolutePath(), core);
							}
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core ) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			
			threads.get(i).start();
		}
		for(int i=0;i<m_numberOfCores;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
		
		// process sub-directories
		int count=0;
		for(File f:files) {
			if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
			else if (m_suffix==null || f.getAbsolutePath().endsWith(m_suffix))
				count++;

		}
		if (count>0)
			System.out.format("%d users/%d docs are loaded from %s...\n", count, m_corpus.getSize(), folder);
	}
		
	// Load one file as a user here. 
	protected void loadUser(String filename, int core){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.				
			
			// Skip the first line since it is user name.
			reader.readLine();
			int postId, parentId, score;
			String source;
			ArrayList<_Review> reviews = new ArrayList<_Review>();

			String category = "", productID = "";
			_Review review;
			int ylabel, index = 0;
			long timestamp;
			while((line = reader.readLine()) != null){
				productID = line.trim();
				source = reader.readLine(); // review content
				category = reader.readLine().trim();
				ylabel = Integer.valueOf(reader.readLine()); // ylabel
				timestamp = Long.valueOf(reader.readLine());

				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
						review = new _Review(-1, source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review, core)) { //Create the sparse vector for the review.
						reviews.add(review);
						review.setID(index++);
					}
				}
			}
			if(reviews.size() > 1){//at least one for adaptation and one for testing
				synchronized (m_allocReviewLock) {
					if(m_allocateFlag)
						allocateReviews(reviews);
					m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
					m_corpus.addDocs(reviews);
				}
			} else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
				review = reviews.get(0);
				synchronized (m_rollbackLock) {
					rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
				}
			}

			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//Tokenizing input text string
	private String[] Tokenizer(String source, int core){
		String[] tokens = getTokenizer(core).tokenize(source);
		return tokens;
	}
	
	//Snowball Stemmer.
	private String SnowballStemming(String token, int core){
		SnowballStemmer stemmer = getStemmer(core);
		stemmer.setCurrent(token);
		if(stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected TokenizeResult TokenizerNormalizeStemmer(String source, int core){
		String[] tokens = Tokenizer(source, core); //Original tokens.
		TokenizeResult result = new TokenizeResult(tokens);

		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]),core);
		
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
					legit |= isLegit(tokens[j]);
					if (legit)//at least one of them is legitimate
						Ngrams.add(token);
				}
			}
		}

		result.setTokens(Ngrams.toArray(new String[Ngrams.size()]));
		return result;
	}
	
	/*Analyze a document and add the analyzed document back to corpus.*/
	protected boolean AnalyzeDoc(_Doc doc, int core) {
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource(),core);// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();

		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		
		if (spVct.size()>m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());
			synchronized (m_corpusLock) {
				m_classMemberNo[y]++;
			}
			if (m_releaseContent)
				doc.clearSource();
			
			return true;
		} else {
			/****Roll back here!!******/
			synchronized (m_rollbackLock) {
				rollBack(spVct, y);
			}
			return false;
		}
	}

	public void constructUserIDIndex(){
		m_userIDIndex = new HashMap<String, Integer>();
		for(int i=0; i<m_users.size(); i++)
			m_userIDIndex.put(m_users.get(i).getUserID(), i);
	}

	//convert the input token sequence into a sparse vector (docWordMap cannot be changed)
	// Since multiple threads access the featureStat, we need lock for this variable.
	@Override
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
							if(m_featureStat.containsKey(0)){
								synchronized(m_featureStatLock){
									m_featureStat.get(token).addOneDF(y);
								}
							}
						}
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					if(m_featureStat.containsKey(token)){
						synchronized(m_featureStatLock){
							m_featureStat.get(token).addOneDF(y);
						}
					}
				}
				if(m_featureStat.containsKey(token)){
					synchronized(m_featureStatLock){
						m_featureStat.get(token).addOneTTF(y);
					}
				}
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					if (!m_isCVStatLoaded && (docWordMap==null || !docWordMap.containsKey(index))){
						synchronized(m_featureStatLock){
							m_featureStat.get(token).addOneDF(y);
						}
					}
				}
				
				if (!m_isCVStatLoaded){
					synchronized(m_featureStatLock){
						m_featureStat.get(token).addOneTTF(y);
					}
				}
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		return spVct;
	}
	
	// return a tokenizer using the core number
	private Tokenizer getTokenizer(int index){
		if(index==m_numberOfCores-1)
			return m_tokenizer;
		else
			return m_tokenizerPool[index];
	}
	
	// return a stemmer using the core number
	private SnowballStemmer getStemmer(int index){
		if(index==m_numberOfCores-1)
			return m_stemmer;
		else
			return m_stemmerPool[index];
	}

}
