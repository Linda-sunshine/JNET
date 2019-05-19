/**
 * 
 */
package structures;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

/**
 * @author hongning
 * Sentence structure for text documents
 */
public class _Stn {

	//added by Renqin
	public double[] m_topics;
	public double[] m_topicSstat;
	public _Word [] m_words;
	public double[] m_xSstat;
	Random m_rand;
	int m_index; // position in the document
	public int m_xSize;
	
	_SparseFeature[] m_x_sparse; // bag of words for a sentence
	
	//structure for HTSM
	public double[] m_transitFv; // features for determine topic transition
	double m_transitStat; // posterior topic transit probability

	// tructure for LR-HTSM
	public double[] m_sentiTransitFv; // features for determine sentiment transition
	double m_sentiTransitStat; // posterior sentiment transit probability
	
	String[] m_rawTokens; // raw token sequence after tokenization
	String[] m_sentencePOSTag; // corresponding POS tags
	String m_rawSource; // original sentence string content
	
	//structure for topic assignment used in HTSM and LR-HTSM, one topic per sentence
	int m_topic; //topic/aspect assignment
	
	//attribute label for NewEgg data
	// default is -1 so that it can help to debug 
	// 0 is for pos, 1 is for neg and 2 for neutral or comment
	// use in FastRestritedHMM.java for sentiment to decide sentiment switch 
	int m_sentimentLabel = -1;
	int m_predictedSentimentLabel = -1;
	
	HashMap<Integer, Double> m_childSimMap;

	public _Stn(int index, _SparseFeature[] x, String[] rawTokens, String[] posTags, String rawSource) {
		m_index = index;
		m_x_sparse = x;
		m_rawTokens = rawTokens;
		m_sentencePOSTag = posTags;
		m_rawSource = rawSource;
		
		m_transitFv = new double[_Doc.stn_fv_size];
		m_sentiTransitFv = new double[_Doc.stn_senti_fv_size];
	}
	
	public _Stn(_SparseFeature[] x, String[] rawTokens, String[] posTags, String rawSource, int label) {
		m_x_sparse = x;
		m_rawTokens = rawTokens;
		m_sentencePOSTag = posTags;
		m_rawSource = rawSource;
		m_sentimentLabel = label;
		
		m_transitFv = new double[_Doc.stn_fv_size];
		m_sentiTransitFv = new double[_Doc.stn_senti_fv_size];
	}

	// added by Renqin
	//initial topic proportion
	public void setTopicsVct4ThreePhi(int k, int xSize) {

		m_topics = new double[k+1];
		m_topicSstat = new double[k+1];
		Arrays.fill(m_topics, 0);
		Arrays.fill(m_topicSstat, 0);
	
		int stnSize = (int)getLength();
		
		if(m_words==null||m_words.length!=stnSize){
			m_words = new _Word[stnSize];
		}
		
		if(m_rand==null)
			m_rand = new Random();
		
		m_xSize = xSize;
		
		if(m_xSstat==null){
			m_xSstat = new double[m_xSize];
		}
		
		int wIndex = 0, wid, tid, xid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				xid = m_rand.nextInt(m_xSize);
				tid = 0;
				if(xid==0){
					tid = m_rand.nextInt(k);
					m_topicSstat[tid] ++;
					m_xSstat[xid] ++;
				}else if(xid==1){
					tid = k;
					m_topicSstat[tid] ++;
					m_xSstat[xid] ++;
				}
				
				m_words[wIndex] = new _Word(wid, tid, xid);
			
				wIndex ++;
			}
		}
		
	}
	
	public void setTopicsVct(int k) {

		m_topics = new double[k];
		m_topicSstat = new double[k];
		Arrays.fill(m_topics, 0);
		Arrays.fill(m_topicSstat, 0);
		
		int stnSize = (int)getLength();
		
		if(m_words==null||m_words.length!=stnSize){
			m_words = new _Word[stnSize];
		}
		
		if(m_rand==null)
			m_rand = new Random();
		
		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			tid = 0;
			for(int j=0; j<fv.getValue(); j++){
				tid = m_rand.nextInt(k);
				m_topicSstat[tid] ++;
				m_words[wIndex] = new _Word(wid, tid);	
				wIndex ++;
			}
				
		}
		
	}
	
	public int getIndex() {
		return m_index;
	}

	public _SparseFeature[] getFv() {
		return m_x_sparse;
	}
	
	public void setStnPredSentiLabel(int label){
		m_predictedSentimentLabel = label;
	}
	
	public int getStnPredSentiLabel(){
		return m_predictedSentimentLabel;
	}
	
	public void setStnSentiLabel(int label){
		m_sentimentLabel = label;
	}
	
	public int getStnSentiLabel(){
		return m_sentimentLabel;
	}
	
	public String getRawSentence(){
		return m_rawSource;
	}
	
	public String[] getRawTokens() {
		return m_rawTokens;
	}
	
	public String[] getSentencePosTag(){
		return m_sentencePOSTag;
	}	
	
	public double[] getTransitFvs() {
		return m_transitFv;
	}
	
	public double[] getSentiTransitFvs() {
		return m_sentiTransitFv;
	}
	
	public double getTransitStat() {
		return m_transitStat;
	}
	
	public void setTransitStat(double t) {
		m_transitStat = t;
	}
	
	public double getSentiTransitStat() {
		return m_sentiTransitStat;
	}
	
	// this is not actually document length, given we might normalize the values in m_x_sparse
	public double getLength() {
		double length = 0;
		for(_SparseFeature f:m_x_sparse)
			length += f.getValue();
		return length;
	}
	
	public void setSentiTransitStat(double t) {
		m_sentiTransitStat = t;
	}
	
	public int getTopic() {
		return m_topic;
	}
	
	public void setTopic(int i) {
		m_topic = i;
	}
	
	
	//annotate by all the words
	public int AnnotateByKeyword(Set<Integer> keywords){
		int count = 0;
		for(_SparseFeature t:m_x_sparse){
			if (keywords.contains(t.getIndex()))
				count ++;
		}
		return count;
	}
	
	public void permuteStn(){
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

	public _Word[] getWords() {
		return m_words;
	}
	
}
