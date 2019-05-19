package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import utils.Utils;

public class _ParentDoc extends _Doc {
	public ArrayList<_ChildDoc> m_childDocs4Dynamic;
	public ArrayList<_ChildDoc> m_childDocs;
	HashMap<Integer, Integer> m_word2Index;
	public double[] m_featureWeight;
	double[] m_wordDistribution;

	public _ParentDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);

		m_childDocs = new ArrayList<_ChildDoc>();
		
		setName(name);
		setTitle(title);
	}
	
	public void initFeatureWeight(int featureLen){
		m_featureWeight = new double[featureLen];
		Arrays.fill(m_featureWeight, 0);
	}
	
	public void addChildDoc(_ChildDoc cDoc){
		m_childDocs.add(cDoc);
	}
	
	public void addChildDoc4Dynamics(_ChildDoc cDoc){
		m_childDocs4Dynamic.add(cDoc);
	}
	
	@Override
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
		
		m_phi = new double[m_x_sparse.length][k];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++) 
			m_word2Index.put(m_x_sparse[i].m_index, i);
	}
	
	public void setTopics4GibbsTest(int k, double alpha, int testLength){
		super.setTopics4GibbsTest(k, alpha, testLength);
		
		m_phi = new double[m_x_sparse.length][k];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++) 
			m_word2Index.put(m_x_sparse[i].m_index, i);
	}
	
	public void collectTopicWordStat() {
		for (_Word w:m_words) {
			m_phi[m_word2Index.get(w.getIndex())][w.getTopic()]++;
		}
	}
	
	public void estStnTheta() {
		double[] theta;
		for(_Stn s:m_sentences) {
			theta = s.m_topics;
			for(_SparseFeature f:s.m_x_sparse) {
				for(int tid=0; tid<m_topics.length; tid++){
					int index = f.m_index;
					int term1 = m_word2Index.get(index);
					double temp = m_phi[term1][tid];
					theta[tid] += temp; 
				}
			}
			Utils.L1Normalization(theta);
		}
	}
	
	public void initWordDistribution(int vocSize, double smoothingParam) {
		m_wordDistribution = new double[vocSize];
		_SparseFeature[] sfs = getSparse();
		int uniqueWords = sfs.length;
		for(_SparseFeature sf:getSparse()){
			int wid = sf.getIndex();
			double val = sf.getValue();
			m_wordDistribution[wid] = (val*1.0)/(m_totalLength+smoothingParam*uniqueWords);
		}
		
		for(int wid=0; wid<vocSize; wid++)
			m_wordDistribution[wid] += smoothingParam/(m_totalLength+smoothingParam*uniqueWords);
	}
	
	public double getWordDistribution(int wid){
		return m_wordDistribution[wid];
	}
}
