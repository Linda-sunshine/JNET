package structures;

import java.util.HashMap;

public class _Review extends _Doc {

	protected String m_userID;
	protected String m_category;

	// Added for the HDP algorithm.
	protected _HDPThetaStar m_hdpThetaStar;

	// key: hdpThetaStar, value: count
	protected HashMap<_HDPThetaStar, Integer> m_thetaCountMap = new HashMap<_HDPThetaStar, Integer>();

	protected double m_L4New = 0;

	// Assign each review a confidence, used in hdp model.
	protected double m_confidence = 1;

	// total count of words of the review represented in language model.
	double m_lmSum = -1;

	// mask used for cross validation
	protected int m_mask = -1;

	// this is designed for stackoverflow post
	// post id is its own id, while id is the index of the current review in the user.
	// if parent_id is -1, it is question, other wise it is answer (the corresponding question id)
	protected int m_postId = -1;
	protected int m_parentId = -1;

	//Constructor for route project.
	public _Review(int ID, String source, int ylabel){
		super(ID, source, ylabel);
	}

	// the originial ID is the global index of the document.
	// later, the meaning of ID is changed based on need.
	// in EUB, the ID is the local index of the document in the current user.
	public _Review(int ID, int postId, String source, int ylabel, int parentId, String userID, long timestamp){
		super(ID, source, ylabel);
		m_postId = postId;
		m_userID = userID;
		m_parentId = parentId;
		m_timeStamp = timestamp;
	}

	public _Review(int ID, String source, int ylabel, String userID, String productID, String category, long timeStamp){
		super(ID, source, ylabel);
		m_userID = userID;
		m_itemID = productID;
		m_category = category;
		m_timeStamp = timeStamp;
	}
	public int getPostId(){
		return m_postId;
	}

	public int getParentId(){
		return m_parentId;
	}
	//Compare the timestamp of two documents and sort them based on timestamps.
	@Override
	public int compareTo(_Doc d){
		if(m_timeStamp < d.m_timeStamp)
			return -1;
		else if(m_timeStamp == d.m_timeStamp)
			return 0;
		else 
			return 1;
	}

	//Access the userID of the review.
	public String getUserID(){
		return m_userID;
	}
	
	@Override
	public String toString(){
		return String.format("%s-%s-%s-%s", m_userID, m_itemID, m_category, m_type);
	}
	
	// Added by Lin for experimental purpose.
	public String getCategory(){
		return m_category;
	}

	public void setHDPThetaStar(_HDPThetaStar s){
		m_hdpThetaStar = s;
	}
	
	// Increase the current hdpThetaStar count.
	public void updateThetaCountMap(int c){
		if(!m_thetaCountMap.containsKey(m_hdpThetaStar)){
			m_thetaCountMap.put(m_hdpThetaStar, c);
		} else{
			int v = m_thetaCountMap.get(m_hdpThetaStar);
			m_thetaCountMap.put(m_hdpThetaStar, v+c);
		}
	}
	
	public HashMap<_HDPThetaStar, Integer> getThetaCountMap(){
		return m_thetaCountMap;
	}
	public void clearThetaCountMap(){
		m_thetaCountMap.clear();
	}
	public _HDPThetaStar getHDPThetaStar(){
		return m_hdpThetaStar;
	}
	
	// Added by Lin for HDP evaluation.
	double[] m_cluPosterior;
	
	public void setClusterPosterior(double[] posterior) {
		if (m_cluPosterior==null || m_cluPosterior.length != posterior.length)
			m_cluPosterior = new double[posterior.length];
		System.arraycopy(posterior, 0, m_cluPosterior, 0, posterior.length);
	}
	
	public double[] getCluPosterior(){
		return m_cluPosterior;
	}
	public void setL4NewCluster(double l){
		m_L4New = l;
	}
	
	public double getL4NewCluster(){
		return m_L4New;
	}

	public double getLMSum(){
		if(m_lmSum != -1)
			return m_lmSum;
		else{
			double sum = 0;
			for(_SparseFeature sf: m_lm_x_sparse)
				sum += sf.getValue();
			return sum;
		}
	}

	public void setConfidence(double conf){
		m_confidence = conf;
	}
	
	public double getConfidence(){
		return m_confidence;
	}

	public void setMask4CV(int k){
		m_mask = k;
	}

	public int getMask4CV(){
		return m_mask;
	}

}
