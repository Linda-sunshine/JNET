package Analyzer;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * 
 * @author Hongning Wang
 * Basic functionalities for analyzing documents
 * This should only include the most abstractive operations
 */

public abstract class Analyzer {
	
	protected _Corpus m_corpus;
	protected int m_classNo; //This variable is just used to init stat for every feature. How to generalize it?
	protected int[] m_classMemberNo; //Store the number of members in a class.
	protected int m_Ngram; 
	protected int m_TotalDF = -1; // why do we need this in Analyzer class???
	
	//structures for managing features 
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	
	/* Indicate if we will allow new features. After loading the CV file, the flag is set to true, 
	 * which means no new features will be created when analyzing documents.*/
	protected boolean m_isCVLoaded = false;
	protected boolean m_isCVStatLoaded = false; // indicate if we will collect corpus-level feature statistics
		
	//conditions for filtering loaded documents 
	protected int m_lengthThreshold = 5;//minimal length of indexed document	
	
	//if we have to store the original content of documents
	protected boolean m_releaseContent = true;//by default we will not store it to save memory
	
	public Analyzer(int classNo, int minDocLength) {
		m_corpus = new _Corpus();
		
		m_classNo = classNo;
		m_classMemberNo = new int[classNo];
		
		m_featureNames = new ArrayList<String>();
		m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		m_featureStat = new HashMap<String, _stat>();
		
		m_lengthThreshold = minDocLength;
	}	
	
	public void reset() {
		Arrays.fill(m_classMemberNo, 0);
		m_featureNames.clear();
		m_featureNameIndex.clear();
		m_featureStat.clear();
		m_corpus.reset();
	}
	
	public HashMap<String, Integer> getFeatureMap() {
		return m_featureNameIndex;
	}
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	protected boolean LoadCV(String filename) {
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#")){//comments
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}						
				} else 
					expandVocabulary(line);
			}
			reader.close();
			
			System.out.format("%d feature words loaded from %s...\n", m_featureNames.size(), filename);
			m_isCVLoaded = true;
			
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
	
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix) throws IOException {
		if (folder==null || folder.isEmpty())
			return;
		
		int current = m_corpus.getSize();
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				LoadDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		System.out.format("Loading %d reviews from %s\n", m_corpus.getSize()-current, folder);
	}
	
	abstract public void LoadDoc(String filename);
	
	//Add one more token to the current vocabulary.
	protected void expandVocabulary(String token) {
		m_featureNameIndex.put(token, m_featureNames.size()); // set the index of the new feature.
		m_featureNames.add(token); // Add the new feature.
		m_featureStat.put(token, new _stat(m_classNo));
	}
		
	//Return corpus without parameter and feature selection.
	//why do we need such a function????
	public _Corpus returnCorpus(String finalLocation) throws FileNotFoundException {
		SaveCVStat(finalLocation);
		
		int sum = 0;
		for(int c:m_classMemberNo) {
			System.out.print(c + " ");
			sum += c;
		}
		System.out.println(", Total: " + sum);
		
		return getCorpus();
	}
	
	public _Corpus getCorpus() {
		//store the feature names into corpus
		m_corpus.setFeatures(m_featureNames);
		m_corpus.setFeatureStat(m_featureStat);
		m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		m_corpus.setContent(!m_releaseContent);
		return m_corpus;
	}
	
	void rollBack(HashMap<Integer, Double> spVct, int y){
		if (!m_isCVLoaded) {
			for(int index: spVct.keySet()) {
				String token="";
				if(m_featureNames.contains(index)) {	
					token = m_featureNames.get(index);
					_stat stat = m_featureStat.get(token);

					if(Utils.sumOfArray(stat.getDF())==1){//If the feature is the first time to show in feature set.
						m_featureNameIndex.remove(index);
						m_featureStat.remove(token);
						m_featureNames.remove(index);
					} else {//If the feature is not the first time to show in feature set.
						stat.minusOneDF(y);
						stat.minusNTTF(y, spVct.get(index));
					}
				}
			}
		} else{//If CV is loaded and CV's statistics are loaded from file, no need to change it
			if (m_isCVStatLoaded)
				return;
			
			// otherwise, we can minus the DF and TTF directly.
			for(int index: spVct.keySet()){
				String token = m_featureNames.get(index);
				_stat stat = m_featureStat.get(token);
				stat.minusOneDF(y);
				stat.minusNTTF(y, spVct.get(index));
			}
		}
	}
	
	//Give the option, which would be used as the method to calculate feature value and returned corpus, calculate the feature values.
	public void setFeatureValues(String fValue, int norm) {
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
		int N = m_isCVStatLoaded ? m_TotalDF : docs.size();

		if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					sf.setTF(sf.getValue());
					
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					double TFIDF = TF * IDF;
					sf.setValue(TFIDF);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("TFIDF-sublinear")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
//					sf.setTF(sf.getValue());
					
					double TF = 1 + Math.log10(sf.getValue());// sublinear TF
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = 1 + Math.log10(N / DF);
					double TFIDF = TF * IDF;
					sf.setValue(TFIDF);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("BM25")) {
			double k1 = 1.5; // [1.2, 2]
			double b = 0.75; // (0, 1000]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg, avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
//					sf.setTF(sf.getValue());
					
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N - DF + 0.5) / (DF + 0.5));
					double BM25 = IDF * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
					sf.setValue(BM25);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("PLN")) {
			double s = 0.5; // [0, 1]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg, avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
//					sf.setTF(sf.getValue());
					
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n)) * IDF;
					sf.setValue(PLN);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else {
			System.out.println("No feature value is set, keep the raw count of every feature in setFeatureValues().");
			//the original feature is raw TF
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
//					sf.setTF(sf.getValue());
					
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		}

		//rank the documents by product and time in all the cases
		//Collections.sort(m_corpus.getCollection());
		if (norm == 1){
			for(_Doc d:docs)			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d:docs)			
				Utils.L2Normalization(d.getSparse());
		} else
			System.out.println("No normalizaiton is adopted here or wrong parameters in setFeatureValues()!");
		
		System.out.format("Text feature generated for %d documents...\n", m_corpus.getSize());
	}

	//Select the features and store them in a file.
	public void featureSelection(String location, String featureSelection, double startProb, double endProb, int maxDF, int minDF) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(startProb, endProb, maxDF, minDF);

		System.out.println("*******************************************************************");
		if (featureSelection.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		
		m_featureNames = selector.getSelectedFeatures();		
		SaveCV(location, featureSelection, startProb, endProb, maxDF, minDF); // Save all the features and probabilities we get after analyzing.
		System.out.println(m_featureNames.size() + " features are selected!");
		
		// need some redesign of the current awkward procedure for feature selection and feature vector construction!!!!
		//clear memory for next step feature construction
//		reset();
//		LoadCV(location);//load the selected features
	}
	
	// Added by Lin for feature selection.
	//Select the features and store them in a file.
	public void featureSelection(String location, String featureSelection, int maxDF, int minDF, int topK) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(0, 1, maxDF, minDF);

		System.out.println("*******************************************************************");
		if (featureSelection.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		
		ArrayList<String> features = selector.getSelectedFeatures();	
		if(topK > features.size())
			m_featureNames = features;
		else{
			m_featureNames.clear();
			for(int i=features.size()-1; i>features.size()-1-topK; i--){
				m_featureNames.add(features.get(i));
			}
		}
		
		SaveCV(location, featureSelection, 0, 1, maxDF, minDF); // Save all the features and probabilities we get after analyzing.
		System.out.println(m_featureNames.size() + " features are selected!");
	}
	
	// Added by Lin for feature selection.
	//Select the features and store them in a file.
	public void featureSelection(String location, String fs1, String fs2, int maxDF, int minDF, int topK) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(0, 1, maxDF, minDF);

		String featureSelection1 = fs1;
		String featureSelection2 = fs2;
		
		System.out.println("*******************************************************************");
		if (featureSelection1.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection1.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection1.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection1.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		ArrayList<String> features1 = selector.getSelectedFeatures();	
		String cur;
		int end = 0;
		for(int i=0; i<features1.size()/2; i++){
			end = features1.size()-1-i;
			cur = features1.get(i);
			features1.set(i, features1.get(end));
			features1.set(end, cur);
		}
		
		if (featureSelection2.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection2.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection2.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection2.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		
		ArrayList<String> features2 = selector.getSelectedFeatures();	
		for(int i=0; i<features2.size()/2; i++){
			end = features2.size()-1-i;
			cur = features2.get(i);
			features2.set(i, features1.get(end));
			features2.set(end, cur);
		}
		// Take the union of the two sets of features.
		HashSet<String> selectedFeatures = new HashSet<String>();
		for(int i=0; i<features1.size() || i<features2.size(); i++){
			if(i < features1.size() && i < features2.size()){
				selectedFeatures.add(features1.get(i));
				selectedFeatures.add(features2.get(i));
			} else if(i >= features1.size() && i < features2.size())
				selectedFeatures.add(features2.get(i));
			else
				selectedFeatures.add(features1.get(i));
		}
		ArrayList<String> features = new ArrayList<String>();
		for(String s: selectedFeatures)
			features.add(s);
		if(topK > selectedFeatures.size())
			m_featureNames = features;
		else{
			m_featureNames.clear();
			for(int i=selectedFeatures.size()-1; i>selectedFeatures.size()-1-topK; i--){
				m_featureNames.add(features.get(i));
			}
		}
		
		SaveCV(location, fs1+"_"+fs2, 0, 1, maxDF, minDF); // Save all the features and probabilities we get after analyzing.
		System.out.println(m_featureNames.size() + " features are selected!");
	}

	//Save all the features and feature stat into a file.
	protected void SaveCV(String featureLocation, String featureSelection, double startProb, double endProb, int maxDF, int minDF) throws FileNotFoundException {
		if (featureLocation==null || featureLocation.isEmpty())
			return;
		
		System.out.format("Saving controlled vocabulary to %s...\n", featureLocation);
		PrintWriter writer = new PrintWriter(new File(featureLocation));
		//print out the configurations as comments
		writer.format("#NGram:%d\n", m_Ngram);
		writer.format("#Selection:%s\n", featureSelection);
		writer.format("#Start:%f\n", startProb);
		writer.format("#End:%f\n", endProb);
		writer.format("#DF_MaxCut:%d\n", maxDF);
		writer.format("#DF_MinCut:%d\n", minDF);
		
		//print out the features
		for (int i = 0; i < m_featureNames.size(); i++)
			writer.println(m_featureNames.get(i));
		writer.close();
	}
	
	//Save all the features and feature stat into a file.
	public void SaveCVStat(String fvStatFile) {
		if (fvStatFile==null || fvStatFile.isEmpty())
			return;
		
		ArrayList<Double> DFList = new ArrayList<Double>();
		double totalDF = 0;
		
		ArrayList<Double> TTFList = new ArrayList<Double>();
		double totalTTF = 0;
		
		try {
			PrintWriter writer = new PrintWriter(new File(fvStatFile));
		
			for(int i = 0; i < m_featureNames.size(); i++){
				writer.print(m_featureNames.get(i));
				_stat temp = m_featureStat.get(m_featureNames.get(i));
				for(int j = 0; j < temp.getDF().length; j++){
					if(temp.getDF()[j]>0){
						DFList.add((double)temp.getDF()[j]);
						totalDF += temp.getDF()[j];
						
					}
					
					writer.print("\t" + temp.getDF()[j]);
				}
				for(int j = 0; j < temp.getTTF().length; j++){
					if(temp.getTTF()[j]>0){
						TTFList.add((double)temp.getTTF()[j]);
						totalTTF += temp.getTTF()[j];
					}
						
					writer.print("\t" + temp.getTTF()[j]);
				}
				writer.println();
			}
			writer.close();
			
			//print out some basic statistics of the corpus
			double maxDF = Collections.max(DFList);
			double avgDF = totalDF/m_featureNames.size();
			System.out.println("maxDF\t"+maxDF+"\t avgDF \t"+avgDF+"\t totalDF\t"+totalDF);
			
			double maxTTF = Collections.max(TTFList);
			double avgTTF = totalTTF/m_featureNames.size();
			System.out.println("maxTTF\t"+maxTTF+"avgTTF\t"+avgTTF+"\t totalTTF \t"+totalTTF);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	//Return the number of features.
	public int getFeatureSize(){
		return m_featureNames.size();
	}
	
	public HashMap<String, _stat> getFeatureStat(){
		return m_featureStat;
	}
	
	public ArrayList<String> getFeatures(){
		return m_featureNames;
	}
}
