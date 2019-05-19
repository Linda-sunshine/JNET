/**
 * 
 */
package structures;

/**
 * @author Hongning Wang
 * structure used for Gibbs sampling in documents
 */
public class _Word {

	//structure used for LDA model
	int m_index; // global word index in the vocabulary
	int m_topic; // topic assignment
	
	//structure used for ParentChildTopicModel
	int m_x;
	double[] m_xProb;	
	double[] m_features;
	
	//structure used for ParentChildTopicModelwithProbit
	double m_xVal;
	int m_localIndex; // index in the sorted sparse vector of its original documents

	public _Word(int index){
		m_index = index;
		m_xProb = new double[2];
	}
	
	public _Word(int index, int topic) {
		m_index = index;
		m_topic = topic;
		m_xProb = new double[2];
	}

	public _Word(int index, int topic, int x) {
		m_index = index;
		m_topic = topic;
		m_x = x;
		m_xProb = new double[2];
	}
	
	public _Word(int index, int topic, int xid, int localIndex, double[] fVct){
		m_index = index;
		m_topic = topic;
		m_x = xid;
		m_localIndex = localIndex;
		m_xProb = new double[2];
	}
	
	public _Word(int index, int topic, double xVal, int localIndex, double[] fVct) {
		m_index = index;
		m_topic = topic;
		m_xVal = xVal;
		m_localIndex = localIndex;
		m_x = xVal>0?1:0;
		m_xProb = new double[2];
	}
	
	public int getIndex() {
		return m_index;
	}
	
	public void setTopic(int topic) {
		m_topic = topic;
	}
	
	public int getTopic() {
		return m_topic;
	}
	
	public void setX(int x) {
		m_x = x;
	}
	
	public int getX() {
		return m_x;
	}
	
	public void setXValue(double xVal) {
		m_xVal = xVal;
//		m_x = xVal>0?1:0;
	}
	
	public double getXValue() {
		return m_xVal;
	}
	
	public int getLocalIndex() {
		return m_localIndex;
	}
	
	public void collectXStats(){
		m_xProb[m_x] ++;
	}
	
	public double getXProb(){
		return m_xProb[1];
	}
	
	public void setFeatures(double[] features){
		int featureLen = features.length;
		m_features = new double[featureLen];
		for(int i=0; i<featureLen; i++){
			m_features[i] = features[i];
		}
	}
	
	public double[] getFeatures(){
		return m_features;
	}
}
