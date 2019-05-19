package structures;

import java.util.HashMap;

import utils.Utils;

public class _ChildDoc extends _Doc {
	public double[][] m_xTopicSstat;//joint assignment of <x,z>: 0 from general, 1 from specific
	public double[] m_xSstat; // sufficient statistics for x

	public double[][] m_xTopics; // proportion of topics (0 from general, 1 from specific)
	public double[] m_xProportion; // proportion of x

	protected double m_docLenWithXVal;
	
	public _ParentDoc m_parentDoc;
	
	public double m_mu;//similarity between parent and child
	
	public HashMap<Integer, Integer> m_wordXStat;
	
	public _ChildDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);
		m_parentDoc = null;
		m_name = name;
		m_title = title;
		m_docLenWithXVal = 0;
		
	}
	
	public void setParentDoc(_ParentDoc pDoc){
		m_parentDoc = pDoc;
	}

	public void createXSpace(int k, int gammaSize) {
		m_xTopicSstat = new double[gammaSize][k];
		m_xTopics = new double[gammaSize][k];
		m_xSstat = new double[gammaSize];
		m_xProportion = new double[gammaSize];
	}
	
	public void setTopics4Gibbs_LDA(int k, double alpha) {
//		m_stnLikelihoodMap = new HashMap<Integer, Double>();
//		m_stnSimMap = new HashMap<Integer, Double>();
		super.setTopics4Gibbs(k, alpha);
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){		
		createSpace(k, alpha);
		m_wordXStat = new HashMap<Integer, Integer>();
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				tid = m_rand.nextInt(k);
				xid = m_rand.nextInt(gammaSize);
				m_words[wIndex] = new _Word(wid, tid, xid);

				m_xTopicSstat[xid][tid] ++;
				m_xSstat[xid] ++;
				
				if(m_wordXStat.containsKey(wid)){
					m_wordXStat.put(wid, m_wordXStat.get(wid)+1);	
				}else{
					m_wordXStat.put(wid, 1);
				}
				
				wIndex ++;
			}
		}
	}
	
	public void setMu(double mu){
		m_mu = mu;
	}
	
	public double getMu(){
		return m_mu;
	}
	
	public void estGlobalLocalTheta() {
		Utils.L1Normalization(m_xProportion);
		for(int x=0; x<m_xTopics.length; x++)
			Utils.L1Normalization(m_xTopics[x]);
		
		for(_Word w: m_words){
			Utils.L1Normalization(w.m_xProb);
		}
	}
	
	public void setChildDocLenWithXVal(double docLenWithXVal){
		m_docLenWithXVal = docLenWithXVal;
	}
	
	public void calChildDocLenWithXVal(){
		double docLenWithXVal = 0;
		for(_Word word:m_words){
			docLenWithXVal += 1-word.getXProb();
		}
		m_docLenWithXVal = docLenWithXVal;
	}
	
	public double getChildDocLenWithXVal(){
		return m_docLenWithXVal;
	}
	
}
