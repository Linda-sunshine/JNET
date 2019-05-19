/**
 * 
 */
package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import utils.Utils;

/**
 * @author lingong
 * General structure to present a document for DM/ML/IR
 */
public class _Doc extends _DocBase implements Comparable<_Doc> {
	public enum rType {
		TRAIN, // for training the global model
		ADAPTATION, // for training the personalized model
		TEST, // for testing
		SEPARATE // added by Lin for sanity check.
	}

	rType m_type; // specification of this doc

	String m_itemID; // ID of the product being commented
	String m_title; //The short title of the document	
	
	int m_sourceType = 1; // source is 1 for Amazon (unlabeled) and 2 for newEgg (labeled)
	
	// added by Lin
	private _SparseFeature[] m_x_posVct; // sparse vector projected by pos tagging 
	private double[] m_x_aspVct; // aspect vector	
		
	double m_stopwordProportion = 0;
	double m_avgIDF = 0;
	double m_sentiScore = 0; //Sentiment score from sentiwordnet
	
	public void setSourceType(int sourceName) {
		m_sourceType = sourceName;
	}
	
	public int getSourceType() {
		return m_sourceType;
	}
	
	public double getAvgIDF() {
		return m_avgIDF;
	}

	public void setAvgIDF(double avgIDF) {
		this.m_avgIDF = avgIDF;
	}

	public double getStopwordProportion() {
		return m_stopwordProportion;
	}

	public void setStopwordProportion(double stopwordProportion) {
		this.m_stopwordProportion = stopwordProportion;
	}

	public void setSentiScore(double s){
		this.m_sentiScore = s;
	}
	
	public double getSentiScore(){
		return this.m_sentiScore;
	}

	//We only need one representation between dense vector and sparse vector: V-dimensional vector.
	private _SparseFeature[] m_x_projection; // selected features for similarity computation (NOTE: will use different indexing system!!)	
	
	static public final int stn_fv_size = 4; // bias, cosine, length_ratio, position
	static public final int stn_senti_fv_size = 6; // bias, cosine, sentiWordNetScore, prior_positive_negative_count, POS tag divergency
	
	_Stn[] m_sentences;
	
	//p(z|d) for topic models in general
	public double[] m_topics;
	//sufficient statistics for estimating p(z|d)
	public double[] m_sstat;//i.e., \gamma in variational inference p(\theta|\gamma)
	
	// structure only used by Gibbs sampling to speed up the sampling process
	_Word[] m_words; 
	
	_Word[] m_testWords;
	
	int m_testLength;
	
	// structure only used by variational inference
	public double[][] m_phi; // p(z|w, \phi)	
	Random m_rand;
	
	//Constructor.
	public _Doc (int ID, String source, int ylabel){
		this.m_ID = ID;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		m_topics = null;
		m_sstat = null;
		m_words = null;
		m_sentences = null;
		m_type = rType.TRAIN; // by default, every doc is used for training the model
	}

//	public _Doc(int ID, String source, int ylabel, int parentId, String userID, long timestamp){
//		this.m_ID = ID;
//		this.m_source = source;
//		this.m_y_label = ylabel;
//		m_topics = null;
//		m_sstat = null;
//		m_words = null;
//		m_sentences = null;
//		m_type = rType.TRAIN; // by default, every doc is used for training the model
//	}
//
	public _Doc (int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp){
		this.m_ID = ID;
		this.m_name = name;
		this.m_itemID = prodID;
		this.m_title = title;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		this.m_timeStamp = timeStamp;
		m_topics = null;
		m_sstat = null;
		m_words = null;
		m_sentences = null;
		m_type = rType.TRAIN; // by default, every review is used for training the model
	}

	public rType getType() {
		return m_type;
	}

	public void setType(rType type) {
		m_type = type;
	}

	public void setItemID(String itemID) {
		m_itemID = itemID;
	}
	
	public String getItemID() {
		return m_itemID;
	}
		
	public String getTitle(){
		return m_title;
	}
	
	public void setTitle(String title){
		m_title = title;
	}
	
	public _SparseFeature[] getProjectedFv() {
		return this.m_x_projection;
	}
	
	public _Word[] getWords() {
		return m_words;
	}
	
	public _Word[] getTestWords(){
		return m_testWords;
	}
	
	public int getDocTestLength(){
		return m_testLength;
	}
	
	public int getDocInferLength(){
		return this.m_words.length;
	}
		
	public _SparseFeature[] m_x_sparse_infer;
	public void createSparseVct4Infer(){
		HashMap<Integer, Double> inferVct = new HashMap<Integer, Double>();
		
		for(_Word w:m_testWords){
			int wIndex = w.getIndex();
			int featureIndex = Utils.indexOf(m_x_sparse, wIndex);
			double featureVal = m_x_sparse[featureIndex].getValue();
			inferVct.put(featureIndex,  featureVal--);
		}
		
		m_x_sparse_infer = Utils.createSpVct(inferVct);	
	}
	
	public _SparseFeature[] getSparseVct4Infer(){
		return m_x_sparse_infer;
	}
	
	//Create the sparse postagging vector for the document. 
	public void createPOSVct(HashMap<Integer, Double> posVct){
		m_x_posVct = Utils.createSpVct(posVct);
	}
	
	public _SparseFeature[] getPOSVct(){
		return m_x_posVct;
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public void setSentences(ArrayList<_Stn> stnList) {
		m_sentences = stnList.toArray(new _Stn[stnList.size()]);
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public int getSenetenceSize() {
		return this.m_sentences.length;
	}
	
	public _Stn[] getSentences() {
		return m_sentences;
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public _Stn getSentence(int index) {
		return this.m_sentences[index];
	}
	
	public boolean hasSegments() {
		return m_x_sparse[0].m_values != null;
	}
	
	//we will reset the topic vector
	public void setTopics(int k, double alpha) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];
		}
		Utils.randomize(m_sstat, alpha);
	}
	
	public double[] getTopics(){
		return m_topics;
	}
	
	//create necessary structure for variational inference
	public void setTopics4Variational(int k, double alpha) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];//used as p(z|w,\phi)
			m_phi = new double[m_x_sparse.length][k];
		}
		
		Arrays.fill(m_sstat, alpha);
		for(int n=0; n<m_x_sparse.length; n++) {
			Utils.randomize(m_phi[n], alpha);
			double v = m_x_sparse[n].getValue();
			for(int i=0; i<k; i++)
				m_sstat[i] += m_phi[n][i] * v;
		}
	}
	
	void createSpace(int k, double alpha) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];
		}

		Arrays.fill(m_topics, 0);
		Arrays.fill(m_sstat, alpha);
		
		//Warning: in topic modeling, we cannot normalize the feature vector and we should only use TF as feature value!
		int docSize = getTotalDocLength();
		if (m_words==null || m_words.length != docSize) {
			m_words = new _Word[docSize];
		} 
		
		if (m_rand==null)
			m_rand = new Random();
	}
	
	//create necessary structure to accelerate Gibbs sampling
	public void setTopics4Gibbs(int k, double alpha) {
		createSpace(k, alpha);
		
		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse) {
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++) {
				tid = m_rand.nextInt(k);
				m_words[wIndex] = new _Word(wid, tid);// randomly initializing the topics inside a document
				m_sstat[tid] ++; // collect the topic proportion
			
				wIndex ++;
			}
		}
	}
	
	public void createSpace4GibbsTest(int k, double alpha, int trainLength){
		if(m_topics==null||m_topics.length!=k){
			m_topics = new double[k];
			m_sstat = new double[k];
		}
		
		Arrays.fill(m_topics, 0);
		Arrays.fill(m_sstat, alpha);
		
		if(m_words==null||m_words.length!=trainLength){
			m_words = new _Word[trainLength];
		}
		
		if(m_rand==null)
			m_rand = new Random();
	}
	
	public void setTopics4GibbsTest(int k, double alpha, int testLength){
		int trainLength = m_totalLength-testLength;
		createSpace4GibbsTest(k, alpha, trainLength);
		m_testLength = testLength;
		m_testWords = new _Word[testLength];
		
		ArrayList<Integer> wordIndexs = new ArrayList<Integer>();
		while(wordIndexs.size()<testLength){
			int testIndex = m_rand.nextInt(m_totalLength);
			if(!wordIndexs.contains(testIndex)){
				wordIndexs.add(testIndex);
			}
		}
		
		int wIndex = 0, wTrainIndex=0, wTestIndex=0, tid, wid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				tid = m_rand.nextInt(k);
				if(wordIndexs.contains(wIndex)){
					m_testWords[wTestIndex] = new _Word(wid, tid);
					wTestIndex ++;
				}else{
					m_words[wTrainIndex] = new _Word(wid, tid);
					wTrainIndex ++;
					m_sstat[tid] ++; // collect the topic proportion

				}
				
				wIndex ++;
			}
			
		}
		
	}
	
	//permutation the order of words for Gibbs sampling
	public void permutation() {
		_Word t;
		int s;
		for(int i=m_words.length-1; i>1; i--) {
			s = m_rand.nextInt(i);
			
			//swap the word
			t = m_words[s];
			m_words[s] = m_words[i];
			m_words[i] = t;
		}
	}

	@Override
	public int compareTo(_Doc d) {
		int prodCompare = m_itemID.compareTo(d.m_itemID);
		if (prodCompare==0) {
			if(m_timeStamp == d.getTimeStamp())
				return 0;
			return m_timeStamp < d.getTimeStamp() ? -1 : 1;
		} else
			return prodCompare;
	}
	
	public boolean sameProduct(_Doc d) {
		if (m_itemID == null || d.m_itemID == null)
			return false;
		return m_itemID.equals(d.m_itemID);
	}
	
	@Override
	public String toString() {
		return String.format("ProdID: %s\tID: %s\t Rating: %d\n%s", m_itemID, m_name, m_y_label, m_source);
	}
	
	public void setProjectedFv(Map<Integer, Integer> filter) {
		m_x_projection = Utils.projectSpVct(m_x_sparse, filter);
//		if (m_x_projection!=null)
//			Utils.L2Normalization(m_x_projection);
	}
	
	public void setProjectedFv(double[] denseFv) {
		m_x_projection = Utils.createSpVct(denseFv);
	}

	public void setAspVct(double[] aspVct){
		m_x_aspVct = aspVct;
	}
	
	public double[] getAspVct(){
		return m_x_aspVct;
	}
	
	//Query features, added by Lin.
	int m_qDim = 0;
	int m_inlink = 0; //How many documents select this document as neighbor, added by Lin
	double m_avgBoW = 0;
	double m_avgTP = 0;
	double m_avgJcd = 0;
	int[] m_queryIndices;
	double[] m_x_queryValues;
	
	//Construct the query features.
	public void setQueryValues(){
		m_x_queryValues = new double[m_qDim];
	
		//feature[0]: avg BoW.
		m_x_queryValues[0] = m_avgBoW;
		
		//feature[1]: avg TP.
		m_x_queryValues[1] = m_avgTP;
		
		//feature[2]: avg Jaccard.
		m_x_queryValues[2] = m_avgJcd;
		
		//feature[3]: avg IDF.		
		m_x_queryValues[3] = m_avgIDF;
		
		//feature[4]: stopwordProportion
		m_x_queryValues[4] = m_stopwordProportion;
		
		//feature[5]: document length.
		m_x_queryValues[5] = getDocLength();
		
		//feature[6]: sentiment score
		m_x_queryValues[6] = getSentiScore();
		
		//feature[7]: inlink.
		m_x_queryValues[7] = getInlink();
		
	}
	
	public int getInlink(){
		return m_inlink;
	}
	
	public int[] getQueryIndices(){
		m_queryIndices = new int[m_qDim];
		for(int i=0; i<m_qDim; i++)
			m_queryIndices[i] = i;
		return m_queryIndices;
	}
	
	//Access the query features.
	public double[] getQueryValues(){
		return m_x_queryValues;
	}
	// adde by Lin for cluster usage.
	int m_clusterNo = 0;
	public void setQueryDim(int dim){
		m_qDim = dim;
	}
	
	public void setClusterNo(int c){
		m_clusterNo = c;
	}
	
	public int getClusterNo(){
		return m_clusterNo;
	}
	/////////////////////////////////////////////////////////
	//temporal structure for CLogisticRegressionWithDP
	/////////////////////////////////////////////////////////
	public int m_pCount = 0;
	public double m_prob = 0;
	
	public int m_pCount_g = 0;
	public double m_prob_g = 0;
	
	// Added by Lin for clustered svm.
	double m_predV = -1;

	public void setPredValue(double v) {
		m_predV = v;
	}

	public double getPredValue() {
		return m_predV;
	}
}