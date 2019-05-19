package structures;

import java.util.*;

import structures._Doc.rType;
import utils.Utils;

/***
 * @author lin
 * The data structure stores the information of a user used in CoLinAdapt.
 */

public class _User {
	
	protected String m_userID;

	//text reviews associated with this user
	protected ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	
	//profile for this user
	protected double[] m_lowDimProfile;
	protected _SparseFeature[] m_BoWProfile; //The BoW representation of a user.
	
	//personalized prediction model
	protected double[] m_pWeight;
	protected int m_classNo;
	protected int m_featureSize;
	protected ArrayList<Integer> m_category;
	protected double[] m_svmWeights;
	protected double m_sim; // Similarity of itself.

	// performance statistics
	protected _PerformanceStat m_perfStat;

	protected int m_cIndex = 0; // added by Lin, cluster number.
	protected double m_posRatio = 0;
	protected double m_avgIDF = 0;

	// added by Lin for friendship.
	protected String[] m_friends = null;
	protected String[] m_testFriends = null;
	protected String[] m_nonFriends = null;

	protected ArrayList<_Review> m_trainReviews;
	protected ArrayList<_Review> m_testReviews;
	protected int m_trainReviewSize = -1;
	protected int m_testReviewSize = -1;

	protected final HashMap<String, Integer> m_itemIDRating;
	protected String[] m_rankingItems;
	protected HashMap<String, _Review> m_testReviewMap;

	// load the candidates from file instead of constructing online
	protected ArrayList<String> m_candidates = new ArrayList<String>();
	
	// The function is used for finding friends from Amazon data set.
	protected ArrayList<String> m_amazonFriends = new ArrayList<String>();

	public _User(String userID){
		m_userID = userID;
		m_reviews = null;		
		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;
		m_itemIDRating = new HashMap<String, Integer>();
	}
	
	public _User(int cindex, int classNo){
		m_cIndex = cindex;
		m_classNo = classNo;
		m_reviews = new ArrayList<_Review>();		

		m_userID = null;
		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;
		
		m_perfStat = new _PerformanceStat(classNo);
		m_itemIDRating = new HashMap<String, Integer>();
		constructTrainTestReviews();
	}
	
	public _User(String userID, int classNo, ArrayList<_Review> reviews){
		m_userID = userID;
		m_classNo = classNo;

		m_reviews = reviews;

		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;

		m_perfStat = new _PerformanceStat(classNo);
		m_itemIDRating = new HashMap<String, Integer>();
		constructTrainTestReviews();
	}
	
	public void addOneItemIDRatingPair(String item, int r){
		if(!m_itemIDRating.containsKey(item))
			m_itemIDRating.put(item, r);
	}
	
	public void addOnePredResult(int predL, int trueL){
		m_perfStat.addOnePredResult(predL, trueL);
	}
	
	// construct the sparse vectors based on the feature used for sentiment model
//	public void constructLRSparseVector(){
//		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();
//		for(_Review r: m_trainReviews)
//			reviews.add(r.getSparse());
//
//		m_BoWProfile = Utils.MergeSpVcts(reviews);// this BoW representation is not normalized?!
//	}

	// construct the sparse vectors based on the feature used for sentiment model
	public void constructLRSparseVector(){
		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();
		for(_Review r: m_reviews)
			reviews.add(r.getSparse());

		m_BoWProfile = Utils.MergeSpVcts(reviews);// this BoW representation is not normalized?!
	}

	// construct the sparse vectors based on the feature used for language model
	public void constructLMSparseVector(){
		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();
		for(_Review r: m_trainReviews) 
			reviews.add(r.getLMSparse());
		
		m_BoWProfile = Utils.MergeSpVcts(reviews);// this BoW representation is not normalized?!
	}
	
	// build the profile for the user
	public void buildProfile(String model){
		if(model.equals("lm"))
			constructLMSparseVector();
		else
			constructLRSparseVector();
	}
	
	public void normalizeProfile(){
		double sum = 0;
		for(_SparseFeature fv: m_BoWProfile){
			sum += fv.getValue();
		}
		for(_SparseFeature fv: m_BoWProfile){
			double val = fv.getValue() / sum;
			fv.setValue(val);
		}
	}

	// added by Lin for accessing the index of user cluster.
	public int getClusterIndex() {
		return m_cIndex;
	}
	// added by Lin for setting the index of user cluster.
	public void setClusterIndex(int i) {
		m_cIndex = i;
	}

	// Get the user ID.
	public String getUserID(){
		return m_userID;
	}
	
	@Override
	public String toString() {
		return String.format("%s-R:%d", m_userID, getReviewSize());
	}
	
	public boolean hasAdaptationData() {
		for(_Review r:m_reviews) {
			if (r.getType() == rType.ADAPTATION) {
				return true;
			}
		}
		return false;
	}
	
	public void initModel(int featureSize) {
		m_pWeight = new double[featureSize];
	}
	
	public void setModel(double[] weight) {
		initModel(weight.length);
		System.arraycopy(weight, 0, m_pWeight, 0, weight.length);
		m_featureSize = weight.length;
	}

	public int getClassNo(){
		return m_classNo;
	}
	
	public double[] getPersonalizedModel() {
		return m_pWeight;
	}
	
	//Get the sparse vector of the user.
	public _SparseFeature[] getBoWProfile(){
		return m_BoWProfile;
	}
	
	public int getReviewSize() {
		return m_reviews==null?0:m_reviews.size();
	}
	
	public ArrayList<_Review> getReviews(){
		return m_reviews;
	}
	
	public double getBoWSim(_User u) {
		return Utils.cosine(m_BoWProfile, u.getBoWProfile());
	}
	
	public double getSVDSim(_User u) {
		return Utils.cosine(u.m_lowDimProfile, m_lowDimProfile);
	}
	
	public double[] getSVMWeights(){
		return m_svmWeights;
	}
	
	public double linearFunc(_SparseFeature[] fvs, int classid) {
		return Utils.dotProduct(m_pWeight, fvs, classid*m_featureSize);
	}
	
	public int predict(_Doc doc) {
		_SparseFeature[] fv = doc.getSparse();

		double maxScore = Utils.dotProduct(m_pWeight, fv, 0);
		if (m_classNo==2) {
			return maxScore>0?1:0;
		} else {//we will have k classes for multi-class classification
			double score;
			int pred = 0; 
		
			for(int i = 1; i < m_classNo; i++) {
				score = Utils.dotProduct(m_pWeight, fv, i * (m_featureSize + 1));
				if (score>maxScore) {
					maxScore = score;
					pred = i;
				}
			}
			return pred;
		}
	}

	public _PerformanceStat getPerfStat() {
		return m_perfStat;
	}
	
	// Added by Lin for lowDimProfile.
	public void setLowDimProfile(double[] ld){
		m_lowDimProfile = ld;
	}
	
	// Added by Lin to access the low dim profile.
	public double[] getLowDimProfile(){
		return m_lowDimProfile;
	}
	
	public double calculatePosRatio(){
		double count = 0;
		for(_Review r: m_reviews){
			if(r.getYLabel() == 1)
				count++;
		}
		return count/m_reviews.size();
	}

	// Set average IDF value.
	public void setAvgIDF(double v){
		m_avgIDF = v;
	}
	
	public void setSimilarity(double sim){
		m_sim = sim;
	}
	
	public void appendRvws(ArrayList<_Review> rs){
		for(_Review r: rs)
			m_reviews.add(r);
	}
	
	public void calcPosRatio(){
		double pos = 0;
		for(_Review r: m_reviews){
			if(r.getYLabel() == 1)
				pos++;
		}
		m_posRatio = pos / m_reviews.size();
	}
	
	public double getPosRatio(){
		return m_posRatio;
	}

	// Added by Lin for kmeans based on profile.
	public int[] getProfIndices() {
		int[] indices = new int[m_BoWProfile.length];
		for (int i = 0; i < m_BoWProfile.length; i++)
			indices[i] = m_BoWProfile[i].m_index;

		return indices;
	}
	public double[] getProfValues() {
		double[] values = new double[m_BoWProfile.length];
		for(int i=0; i<m_BoWProfile.length; i++) 
			values[i] = m_BoWProfile[i].m_value;
		
		return values;
	}

	public void setSVMWeights(double[] weights){
		m_svmWeights = new double[weights.length];
		m_svmWeights = Arrays.copyOf(weights, weights.length);
	}
	
	public void setFriends(String[] fs){
		m_friends = Arrays.copyOf(fs, fs.length);
	}
	
	public void setTestFriends(String[] fs){
		m_testFriends = Arrays.copyOf(fs, fs.length);
	}
	
	public void setNonFriends(String[] nonfs){
		m_nonFriends = Arrays.copyOf(nonfs, nonfs.length);
	}

	public String[] getFriends(){
		return m_friends;
	}

	public void removeOneFriend(String frd){
		ArrayList<String> frdList = new ArrayList<>();
		for(String f: m_friends){
			frdList.add(f);
		}
		if(!frdList.contains(frd)){
			System.out.println("The friend does not exist!!");
			return;
		}
		frdList.remove(frd);
		m_friends = new String[frdList.size()];
		m_friends = frdList.toArray(m_friends);
	}
	public int getFriendSize(){
		if(m_friends == null)
			return 0;
		else
			return m_friends.length;
	}
	
	public String[] getNonFriends(){
		return m_nonFriends;
	}
	
	public String[] getTestFriends(){
		return m_testFriends;
	}
	
	public int getTestFriendSize(){
		if(m_testFriends == null)
			return 0;
		else
			return m_testFriends.length;
	}
	
	public int getNonFriendSize(){
		if(m_nonFriends == null)
			return 0;
		else
			return m_nonFriends.length;
	}
	// check if a user is a friend of the current user
	public boolean hasFriend(String str){
		if(m_friends == null || m_friends.length == 0){
			return false;
		}
		for(String f: m_friends){
			if(str.equals(f))
				return true;
		}
		return false;
	}
	
	// check if a user is a friend of the current user
	public boolean hasNonFriend(String str){
		if(m_nonFriends == null || m_nonFriends.length == 0){
			return false;
		}
		for(String f: m_nonFriends){
			if(str.equals(f))
				return true;
		}
		return false;
	}
	
	public boolean hasTestFriend(String str){
		if(m_testFriends.length == 0){
			return false;
		}
		for(String f: m_testFriends){
			if(str.equals(f))
				return true;
		}
		return false;
	}
	public void addAmazonFriend(String s){
		m_amazonFriends.add(s);
	}
	
	public ArrayList<String> getAmazonFriends(){
		return m_amazonFriends;
	}
	
	// In order to construct the friends for testing
	ArrayList<String> m_amazonTestFriends = new ArrayList<String>();
	public void addAmazonTestFriend(String s){
		m_amazonTestFriends.add(s);
	}
	
	public ArrayList<String> getAmazonTestFriends(){
		return m_amazonTestFriends;
	}

	public void constructTrainTestReviews(){
		m_trainReviews = new ArrayList<>();
		m_testReviews = new ArrayList<>();
		m_testReviewMap = new HashMap<String, _Review>();
		for(_Review r: m_reviews){
			if(r.getType() == rType.ADAPTATION)
				m_trainReviews.add(r);
			else{
				m_testReviews.add(r);
				m_testReviewMap.put(r.getItemID(), r);
			}
		}
		m_trainReviewSize = m_trainReviews.size();
		m_testReviewSize = m_testReviews.size();
	}
	
	// set the test reveiws for the current user
	public void setTestReviews(ArrayList<_Review> reviews){
		m_testReviews = reviews;
		m_testReviewSize = m_testReviews.size();
		m_reviews.addAll(m_testReviews);
		m_testReviewMap = new HashMap<String, _Review>();
		
		for(_Review r: m_testReviews){
			m_testReviewMap.put(r.getItemID(), r);
		}
	}
	public ArrayList<_Review> getTrainReviews(){
		return m_trainReviews;
	}

	public ArrayList<_Review> getTestReviews(){
		return m_testReviews;
	}
	
	public _Review getTestReview(String item){
		return m_testReviewMap.get(item);
	}

	public int getTrainReviewSize(){
		return m_trainReviewSize;
	}
	
	public int getTestReviewSize(){
		return m_testReviewSize;
	}
	
	public int getItemRating(String item){
		// rating is 0 or 1, thus non-existing is -1
		if(m_itemIDRating.containsKey(item))
			return m_itemIDRating.get(item);
		else{
			return -1;
		}
	}
	// whether this user has rated this item in the testing set
	public boolean containsTestRvw(String item){
		return m_testReviewMap.containsKey(item);
	}

	public int getRankingItemSize(){
		if(m_rankingItems == null)
			return 0;
		else 
			return m_rankingItems.length;
	}
	
	public String[] getRankingItems(){
		return m_rankingItems;
	}
	
	public void setRankingItems(Set<String> items){
		if(items.size() == 0)
			m_rankingItems = null; 
		else{
			m_rankingItems = new String[items.size()];
			int index = 0;
			for(String item: items){
				m_rankingItems[index++] = item;
			}
		}
	}

	public void addOneCandidate(String item){
		m_candidates.add(item);
	}
	
	public void setRankingItems(HashMap<String, ArrayList<String>> itemMap){
		// check if it is a valid user or not
		int relevant = 0;
		ArrayList<String> validItems = new ArrayList<>();
		for(String item: m_candidates){
			if(containsTestRvw(item)){
				if(itemMap.containsKey(item)){
					relevant++;
					validItems.add(item);
				}
			} else{
				if(!itemMap.containsKey(item))
					System.out.println("[error] Bug in ranking candidates!");
				else{
					validItems.add(item);
				}
			}
		}
		// if the user has at least one relevant item
		if(relevant > 0){
			m_rankingItems = new String[validItems.size()];
			int index = 0;
			for(String item: validItems){
				m_rankingItems[index++] = item;
			}
		}
	}

	public _Review getReviewByID(int id){
		if(id >= m_reviews.size())
			System.err.println("[error] Index exceeds the array length!");
		return m_reviews.get(id);
	}
}
