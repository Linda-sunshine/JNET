package structures;

import java.util.ArrayList;
import java.util.Collections;

import Classifier.supervised.liblinear.Feature;
import utils.Utils;

public class _Query {
	public ArrayList<_QUPair> m_docList;
	int m_pairSize;
	
	public _Query(){
		m_docList = new ArrayList<_QUPair>();
		m_pairSize = 0;
	}
	
	@Override
	public String toString() {
		return String.format("%d:%d", m_docList.size(), m_pairSize);
	}
	
	public void addQUPair(_QUPair pair){ m_docList.add(pair); }
	
	public int createRankingPairs() {
		_QUPair qui, quj;
		for(int i=0; i<m_docList.size(); i++) {
			qui = m_docList.get(i);
			for(int j=0; j<i; j++) {
				quj = m_docList.get(j);
				
				if (qui.m_y > quj.m_y) {
					qui.addWorseURL(quj);
					quj.addBetterURL(qui);
					m_pairSize ++;
				} else if (qui.m_y < quj.m_y) {
					qui.addBetterURL(quj);
					quj.addWorseURL(qui);
					m_pairSize ++;
				}
			}
		}
		return m_pairSize;
	}
	
	public void sortDocs() {
		Collections.sort(m_docList);//sort documents by predicted ranking score
	}
	
	public int getPairSize() {
		return m_pairSize;
	}
	
	public int getDocSize() {
		return m_docList.size();
	}
	
	public void extractPairs4RankSVM(ArrayList<Feature[]> fvs, ArrayList<Integer> labels) {
		boolean negSgn = (fvs.size()%2)==0;
		Feature[] fvct;
		for(_QUPair di:m_docList) {
			if (di.m_betterURLs==null)
				continue;
			
			if (negSgn) {
				for(_QUPair dj:di.m_betterURLs) {
					if ((fvct=di.getDiffFv(dj)) != null) {
						fvs.add(fvct);
						labels.add(-1);
					}
				}
			} else {
				for(_QUPair dj:di.m_betterURLs) {
					if ((fvct=dj.getDiffFv(di)) != null) {
						fvs.add(fvct);
						labels.add(1);
					}
				}
			}
		}
	}
	
	public void extractPairs4RankNet(ArrayList<double[]> fvs) {
		double[] diff;
		for(_QUPair di:m_docList) {
			if (di.m_worseURLs==null)
				continue;			
			
			for(_QUPair dj:di.m_worseURLs) {
				diff = Utils.diff(di.m_rankFv, dj.m_rankFv);
				if (diff!=null)
					fvs.add(diff);	
			}
		}
	}
}
