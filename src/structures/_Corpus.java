/**
 * 
 */
package structures;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import utils.Utils;

/**
 * @author lingong
 * General structure of corpus of a set of documents
 */
public class _Corpus {
	static final int ReviewSizeCut = 3;
	
	ArrayList<_Doc> m_collection; //All the documents in the corpus.
	ArrayList<String> m_features; //ArrayList for feature names
	public HashMap<String, _stat> m_featureStat; //statistics about the features
	boolean m_withContent = false; // by default all documents' content has been released
		
	public void setFeatureStat(HashMap<String, _stat> featureStat) {
		this.m_featureStat = featureStat;
	}

	// m_mask is used to do shuffling and its size is the total number of all the documents in the corpus.
	int[] m_mask; 
			
	//Constructor.
	public _Corpus() {
		this.m_collection = new ArrayList<_Doc>();
 	}
	
	public void reset() {
		m_collection.clear();
	}
	
	public void setContent(boolean content) {
		m_withContent = content;
	}
	
	public boolean hasContent() {
		return m_withContent;
	}
	
	public void setFeatures(ArrayList<String> features) {
		m_features = features;
	}
	
	public String getFeature(int i) {
		return m_features.get(i);
	}
	
	public int getFeatureSize() {
		return m_features.size();
	}
	
	public int getClassSize() {
		HashSet<Integer> labelSet = new HashSet<Integer>();
		for(_Doc d:m_collection)
			labelSet.add(d.getYLabel());
		return labelSet.size();
	}
	
	//Initialize the m_mask, the default value is false.
	public void setMasks() {
		this.m_mask = new int[this.m_collection.size()];
	}
	
	//Get all the documents of the corpus.
	public ArrayList<_Doc> getCollection(){
		return this.m_collection;
	}
	
	//Get the corpus's size, which is the total number of documents.
	public int getSize(){
		return m_collection.size();
	}
	
	public int getLargestSentenceSize()
	{
		int max = 0;
		for(_Doc d:m_collection) {
			int length = d.getSenetenceSize();
			if(length > max)
				max = length;
		}
		
		return max;
	}
	
	/*
	 rand.nextInt(k) will always generates a number between 0 ~ (k-1).
	 Access the documents with the masks can help us split the whole whole 
	 corpus into k folders. The function is used in cross validation.
	*/
	public void shuffle(int k) {
		Random rand = new Random();
		for(int i=0; i< m_mask.length; i++) {
			this.m_mask[i] = rand.nextInt(k);
		}
	}
	
	//Add a new doc to the corpus.
	public void addDoc(_Doc doc){
		m_collection.add(doc);
	}
	
	//Add a set of docs to the corpus.
	public void addDocs(ArrayList<_Review> docs){
		m_collection.addAll(docs);
	}
	
	//Get the mask array of the corpus.
	public int[] getMasks(){
		return this.m_mask;
	}
	
	public void mapLabels(int threshold) {
		int y;
		for(_Doc d:m_collection) {
			y = d.getYLabel();
			if (y<threshold)
				d.setYLabel(0);
			else
				d.setYLabel(1);
		}
	}
	
	//save documents to file as sparse feature vectors
	public void save2File(String filename) {
		if (filename==null || filename.isEmpty()) {
			System.out.println("Please specify the file name to save the vectors!");
			return;
		}
		
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
			for(_Doc doc:m_collection) {
				writer.write(String.format("%d", doc.getYLabel()));
				for(_SparseFeature fv:doc.getSparse())
					writer.write(String.format(" %d:%f", fv.getIndex()+1, fv.getValue()));//index starts from 1
				writer.write(String.format(" #%s-%s\n", doc.m_itemID, doc.m_name));//product ID and review ID
			}
			writer.close();
			
			System.out.format("%d feature vectors saved to %s\n", m_collection.size(), filename);
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	// added by Md. Mustafizur Rahman for Topic Modelling
	public double[] getBackgroundProb() {
		double back_ground_probabilty [] = new double [m_features.size()];
		
		for(int i = 0; i<m_features.size();i++) {
			String featureName = m_features.get(i);
			_stat stat =  m_featureStat.get(featureName);
			back_ground_probabilty[i] = Utils.sumOfArray(stat.getTTF());
			
			if (back_ground_probabilty[i] < 0)
				System.err.println("Encounter negative count for word" + featureName);
		}
		
		double sum = Utils.sumOfArray(back_ground_probabilty) + back_ground_probabilty.length;//add one smoothing
		for(int i = 0; i<m_features.size();i++)
			back_ground_probabilty[i] = (1.0 + back_ground_probabilty[i]) / sum;
		return back_ground_probabilty;
	}
}
