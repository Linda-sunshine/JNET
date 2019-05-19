package structures;

import java.util.ArrayList;
import java.util.Arrays;

public class _Doc4DCMLDA extends _Doc{
	
	public double[][] m_wordTopic_stat;
	public double[][] m_wordTopic_prob;
	
	public _Doc4DCMLDA(int ID, String name, String title, String source, int ylabel){
		super(ID, source, ylabel);
	
	}
	
	protected void setWordTopicStat(int k, int vocalSize){
		
		m_wordTopic_stat = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_stat[i], 0);
		
		m_wordTopic_prob = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_prob[i], 0);
	}
	
	public void setTopics4Gibbs(int k, double alpha, int vocalSize){
		createSpace(k, alpha);
		setWordTopicStat(k, vocalSize);
		
		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				tid = m_rand.nextInt(k);
				m_words[wIndex] = new _Word(wid, tid);
				m_sstat[tid] ++;
				
				m_wordTopic_stat[tid][wid] ++;
				wIndex ++;
			}
		}
	}
	
	public void setTopics4GibbsTest(int k, double alpha, int testLength, int vocalSize){
		int trainLength = m_totalLength - testLength;
		
		createSpace4GibbsTest(k, alpha, trainLength);
		setWordTopicStat(k, vocalSize);
		
		m_testLength = testLength;
		m_testWords = new _Word[testLength];
		
		ArrayList<Integer> wordIndexes = new ArrayList<Integer>();
		while(wordIndexes.size()< testLength){
			int testIndex = m_rand.nextInt(m_totalLength);
			if(!wordIndexes.contains(testIndex)){
				wordIndexes.add(testIndex);
			}
		}
		
		int wIndex = 0, wTrainIndex = 0, wTestIndex = 0, tid = 0, wid=0;
		for(_SparseFeature sf:m_x_sparse){
			wid = sf.getIndex();
			for(int j=0; j<sf.getValue(); j++){
				if(wordIndexes.contains(wIndex)){
					tid = m_rand.nextInt(k);
					m_testWords[wTestIndex] = new _Word(wid, tid);
					wTestIndex ++;
				}else{
					tid = m_rand.nextInt(k);
					m_words[wTrainIndex] = new _Word(wid, tid);
					wTrainIndex ++;
					m_wordTopic_stat[tid][wid] ++;
					m_sstat[tid] ++;
				}
				wIndex ++;
			}
		}
			
	}
}



