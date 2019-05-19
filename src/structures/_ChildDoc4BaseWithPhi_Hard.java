package structures;

import java.util.ArrayList;

import utils.Utils;

public class _ChildDoc4BaseWithPhi_Hard extends _ChildDoc4BaseWithPhi{
	public _ChildDoc4BaseWithPhi_Hard(int ID, String name, String title, String source, int ylabel) {
		super(ID, name, title, source, ylabel);
		
		// TODO Auto-generated constructor stub
	}
	
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		_SparseFeature[] parentFv = m_parentDoc.getSparse();
		
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		tid = 0;
		for(_SparseFeature fv:m_x_sparse){
			
			wid = fv.getIndex();
			
			for(int j=0; j<fv.getValue(); j++){

				if(Utils.indexOf(parentFv, wid)!=-1){
					xid = 0;
					tid = m_rand.nextInt(k);
					m_xTopicSstat[xid][tid] ++;
					m_xSstat[xid] ++;
				}else{
				
					xid = m_rand.nextInt(gammaSize);
					if(xid == 0){
						tid = m_rand.nextInt(k);
						m_xTopicSstat[xid][tid]++;
						m_xSstat[xid]++;
					}else if(xid==1){
						tid = k ;
						m_xTopicSstat[xid][wid]++;
						m_xSstat[xid]++;
						m_childWordSstat ++;
					}
				}
				
				m_words[wIndex] = new _Word(wid, tid, xid);
				
				wIndex ++;
			}
		}
	}

	public void setTopics4GibbsTest(int k, double alpha, int testLength){
		int trainLength = m_totalLength - testLength;
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
		
		_SparseFeature[] parentFv = m_parentDoc.getSparse();
		int wIndex = 0, wTrainIndex=0, wTestIndex=0, tid=0, xid=0,  wid=0, gammaSize=m_xSstat.length;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				if(wordIndexs.contains(wIndex)){
					xid = 0;
					tid = m_rand.nextInt(k);
					m_testWords[wTestIndex] = new _Word(wid, tid, xid);
					wTestIndex ++;
				}else{
					if(Utils.indexOf(parentFv, wid)!=-1){
						xid = 0;
						tid = m_rand.nextInt(k);
						m_xTopicSstat[xid][tid] ++;
						m_xSstat[xid] ++;
					}else{
						xid = m_rand.nextInt(gammaSize);
						if(xid == 0){
							tid = m_rand.nextInt(k);
							m_xTopicSstat[xid][tid]++;
							m_xSstat[xid]++;
						}else if(xid==1){
							tid = k ;
							m_xTopicSstat[xid][wid]++;
							m_xSstat[xid]++;
							m_childWordSstat ++;
						}
					}
					
					m_words[wTrainIndex] = new _Word(wid, tid, xid);
					wTrainIndex ++;
				}
				
				wIndex ++;
			}
		}
	}
	
}
