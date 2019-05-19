package Analyzer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.*;
import structures._Doc.rType;
import utils.Utils;

import java.io.*;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public class MultiThreadedLMAnalyzer extends MultiThreadedLinkPredAnalyzer {
	// We don't record the feature stats.
	ArrayList<String> m_lmFeatureNames;
	HashMap<String, Integer> m_lmFeatureNameIndex;
	boolean m_isLMCVLoaded = false;
	
	public MultiThreadedLMAnalyzer(String tokenModel, int classNo, String providedCV, String lmFvFile,
			int Ngram, int threshold, int numberOfCores, boolean b)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
		m_lmFeatureNames = new ArrayList<String>();
		m_lmFeatureNameIndex = new HashMap<String, Integer>();
		loadLMFeatures(lmFvFile);
	}
	
	// Added by Lin. Load the features for language models from a file and store them in the m_LMFeatureNames.
	public boolean loadLMFeatures(String filename){
		//If no lm features provided, we will use the same features as logistic model.
		if (filename==null || filename.isEmpty()){
			m_lmFeatureNameIndex = m_featureNameIndex;
			m_lmFeatureNames = m_featureNames;
			System.out.println("Language models share the same features with classification models!");
			m_isLMCVLoaded = true;
			return true;
		}
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#"))//comments
					continue;	
				else{
					m_lmFeatureNameIndex.put(line, m_lmFeatureNames.size()); // set the index of the new feature.
					m_lmFeatureNames.add(line); // Add the new feature.
				}
			}
			reader.close();
			System.out.format("%d features for language model are loaded from %s...\n", m_lmFeatureNames.size(), filename);
			m_isLMCVLoaded = true;
			return true;
		
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
	
	@Override
	// Analyze the sparse features and language model features at the same time.
	protected boolean AnalyzeDoc(_Doc doc, int core) {
		
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource(),core);// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();

		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		// Construct the sparse vector for the language models.
		HashMap<Integer, Double> lmSpVct = constructLMSpVct(tokens);
		
		if (spVct.size()>m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.createLMSpVct(lmSpVct);
			doc.setStopwordProportion(result.getStopwordProportion());
			synchronized (m_corpusLock) {
				m_corpus.addDoc(doc);
				m_classMemberNo[y]++;
			}
			if (m_releaseContent)
				doc.clearSource();
			
			return true;
		} else {
			/****Roll back here!!******/
			synchronized (m_rollbackLock) {
				rollBack(spVct, y);// no need to roll back lm features.
			}
			return false;
		}
	}

	//Added by Lin for constructing language model vectors and we don't record stats for features used in LM.
	public HashMap<Integer, Double> constructLMSpVct(String[] tokens){
		int lmIndex = 0;
		double lmValue = 0;
		HashMap<Integer, Double> lmVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.	

		// We assume we always have the features loaded beforehand.
		for(int i = 0; i < tokens.length; i++){
			if (isLegit(tokens[i])){
				if(m_lmFeatureNameIndex.containsKey(tokens[i])){
					lmIndex = m_lmFeatureNameIndex.get(tokens[i]);
					if(lmVct.containsKey(lmIndex)){
						lmValue = lmVct.get(lmIndex) + 1;
						lmVct.put(lmIndex, lmValue);
					} else
						lmVct.put(lmIndex, 1.0);
				}
			}
		}
		return lmVct;
	}
	
	// Estimate a global language model.
	// We traverse all review documents instead of using the global TF 
	public double[] estimateGlobalLM(){
		double[] lm = new double[getLMFeatureSize()];
		double sum = 0;
		for(_User u: m_users){
			for(_Review r: u.getReviews()){
				for(_SparseFeature fv: r.getLMSparse()){
					lm[fv.getIndex()] += fv.getValue();
					sum += fv.getValue();
				}
			}
		}
		for(int i=0; i<lm.length; i++){
				lm[i] /= sum;
				if(lm[i] == 0)
					lm[i] = 0.0001;
		}
		return lm;
	}
	
	public int getLMFeatureSize(){
		return m_lmFeatureNames.size();
	}
	
	public void getStat(){
		ArrayList<Integer> medians = new ArrayList<Integer>();
		double pos = 0, total = 0;
		for(_User u: m_users){
			medians.add(u.getReviewSize());
			for(_Review r: u.getReviews()){
				if(r.getYLabel() == 1)
					pos++;
				total++;
			}
		}
		Collections.sort(medians);
		double median = 0;
		if(medians.size() % 2 == 0)
			median = (medians.get(medians.size()/2)+medians.get(medians.size()/2-1))/2;
		else 
			median = medians.get(medians.size()/2);
		System.out.println("median: " + median);
		System.out.println("pos: " + pos);
		System.out.println("total: " + total);
		System.out.println("pos ratio: " + pos/total);

	}
	
	// added by Lin, selected k users for separate testing.
	public void seperateUsers(int k){
		int count = 0;
		while(count < k){
			for(_Review r: m_users.get(count++).getReviews())
				r.setType(rType.SEPARATE);
		}
	}
	
	public void saveUsers(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(_User u: m_users)
				writer.write(u.getUserID()+"\n");
			writer.close();
			System.out.println(m_users.size() + " users are saved!");
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public ArrayList<String> getLMFeatures(){
		return m_lmFeatureNames;
	}
	
	//Load a document and analyze it.
	@Override
	public void LoadJsonDoc(String filename) {
		_Product prod = null;
		JSONArray jarray = null;
		
		try {
			JSONObject json = LoadJSON(filename);
			prod = new _Product(json.getJSONObject("asin"));
			jarray = json.getJSONArray("related");
		} catch (Exception e) {
			System.err.print('X');//fail to parse a json document
			return;
		}	
		
		for(int i=0; i<jarray.length(); i++) {
			try {
				_Post post = new _Post(jarray.getJSONObject(i));
				if (post.isValid(m_dateFormatter)) {
					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
					String content;
					
					//append document title into document content
					if (Utils.endWithPunct(post.getTitle()))
						content = post.getTitle() + " " + post.getContent();
					else
						content = post.getTitle() + ". " + post.getContent();
					
					//int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp
					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), prod.getID(), post.getTitle(), content, post.getLabel()-1, timeStamp);
					if(this.m_stnDetector!=null)
						AnalyzeDocWithStnSplit(review);
					else
						AnalyzeDoc(review);
				}
			} catch (ParseException e) {
				System.out.print('T');
			} catch (JSONException e) {
				System.out.print('P');
			}
		}
	}
	
	// Check if two users have the co-purchase 
	protected boolean hasCoPurchase(_User ui, _User uj){
		int count = 0;
		HashSet<String> item_i = new HashSet<String>();
		for(_Review r: ui.getTrainReviews()){
			item_i.add(r.getItemID());
		}
		for(_Review r: uj.getTrainReviews()){
			if(item_i.contains(r.getItemID())){
				count++;
				if(count == 1)
					return true;
			}
		}
		return false;
	}
	
	// Check if two users have the co-purchase 
	protected boolean hasCoPurchaseInTest(_User ui, _User uj){
		int count = 0;
		HashSet<String> item_i = new HashSet<String>();
		for(_Review r: ui.getTestReviews()){
			item_i.add(r.getItemID());
		}
		for(_Review r: uj.getTestReviews()){
			if(item_i.contains(r.getItemID())){
				count++;
				if(count == 1)
					return true;
			}
		}
		return false;
	}
	
	HashMap<String, Integer> m_validUserMap = new HashMap<String, Integer>();
	public void findtrainFriends(String filename){
		_User ui, uj;
		// Detect all co-purchase.
		for(int i=0; i<m_users.size(); i++){
			ui = m_users.get(i);
			for(int j=i+1; j<m_users.size(); j++){
				uj = m_users.get(j);
				if(hasCoPurchase(ui, uj)){
					ui.addAmazonFriend(uj.getUserID());
					uj.addAmazonFriend(ui.getUserID());
				}
			}
		}
		try{
			double avg = 0, count = 0;
			PrintWriter writer = new PrintWriter(new File(filename));
			for(_User u: m_users){
				if(u.getAmazonFriends().size() == 0)
					continue;
				m_validUserMap.put(u.getUserID(), m_validUserMap.size());
				count++;
				avg += u.getAmazonFriends().size();
				writer.write(u.getUserID()+"\t");
				for(String frd: u.getAmazonFriends())
					writer.write(frd+"\t");
				writer.write("\n");
			}
			System.out.format("[Info]%.1f users have friends and avg friend size: %.2f\n", count, avg/count);
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
	
	public void findTestFriends(String filename){
		_User ui, uj;
		// Detect all co-purchase.
		for(int i=0; i<m_users.size(); i++){
			ui = m_users.get(i);
			if(!m_validUserMap.containsKey(ui.getUserID()))
				continue;
			for(int j=i+1; j<m_users.size(); j++){
				uj = m_users.get(j);
				if(!m_validUserMap.containsKey(uj.getUserID()))
					continue;
				if(hasCoPurchaseInTest(ui, uj)){
					ui.addAmazonTestFriend(uj.getUserID());
					uj.addAmazonTestFriend(ui.getUserID());
				}
			}
		}
		try{
			double avg = 0, count = 0;
			PrintWriter writer = new PrintWriter(new File(filename));
			for(_User u: m_users){
				if(u.getAmazonTestFriends().size() == 0)
					continue;
				
				count++;
				avg += u.getAmazonFriends().size();
				writer.write(u.getUserID()+"\t");
				for(String frd: u.getAmazonFriends())
					writer.write(frd+"\t");
				writer.write("\n");
			}
			System.out.format("[Info]%.1f users have friends in test set and avg test friend size: %.2f\n", count, avg/count);
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
	
	// get the current user size, used in iso mmb as we load training users first
	public int getCurrentUserSize(){
		return m_users.size();
	}

}