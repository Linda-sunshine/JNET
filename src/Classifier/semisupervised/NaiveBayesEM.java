/**
 * 
 */
package Classifier.semisupervised;

import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.NaiveBayes;

/**
 * @author hongning
 * Naive Bayes with EM training
 */
public class NaiveBayesEM extends NaiveBayes {
	
	//parameters to control EM iterations
	double m_converge = 1e-5;
	int m_maxIter = 50;
	
	public NaiveBayesEM(_Corpus c) {
		super(c);
	}

	public NaiveBayesEM(int classNo, int featureSize) {
		super(classNo, featureSize);
	}

	public NaiveBayesEM(_Corpus c, boolean presence, double deltaY,
			double deltaXY) {
		super(c, presence, deltaY, deltaXY);
	}
	
	public void setEMParam(int maxIter, double converge) {
		m_maxIter = maxIter;
		m_converge = converge;
	}

	@Override
	protected void init() {		
		for(_Doc doc: m_trainSet){
			if (doc.getSourceType()==1)
				doc.setTopics(m_classNo, m_deltaY);//create the storage space
		}
		
		MStep(m_trainSet, 0);
	}
	
	double EStep(Collection<_Doc> trainSet) {
		double likelihood = 0;
		for(_Doc doc: m_trainSet){
			score(doc, 0);//to compute p(x|y)p(y) and store it in m_cProbs
			
			if (doc.getSourceType()==1) {//unlabeled data
				double sumY = Utils.logSum(m_cProbs);
				for(int i=0; i<m_classNo; i++) {
					doc.m_sstat[i] = Math.exp(m_cProbs[i] - sumY); // p(y|x)
					likelihood += doc.m_sstat[i] * m_cProbs[i]; // p(x)
				}
			} else if (doc.getSourceType()==2) {//labeled data
				likelihood += m_cProbs[doc.getYLabel()]; //p(x, y=Y)
			}
		}
		
		return likelihood;
	}
	 
	void MStep(Collection<_Doc> trainSet, int iter) {
		super.init();
		
		for(_Doc doc: trainSet){
			if (doc.getSourceType()==2) {// labeled data
				int label = doc.getYLabel();
				m_pY[label] ++;
				for(_SparseFeature sf: doc.getSparse())
					m_Pxy[label][sf.getIndex()] += m_presence?1.0:sf.getValue();
			} else if (iter>0 && doc.getSourceType()==1) {// unlabeled data
				double[] label = doc.m_sstat;
				for(int i=0; i<m_classNo; i++) {
					m_pY[i] += label[i];
					for(_SparseFeature sf: doc.getSparse())
						m_Pxy[i][sf.getIndex()] += (m_presence?1.0:sf.getValue()) * label[i];
				}
			} 
		}
		
		//normalization
		double sumY = Math.log(Utils.sumOfArray(m_pY) + m_deltaY * m_classNo);
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY) - sumY;
			double sumX = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sumX;
		}
	}
	
	//EM-training on the data set.
	@Override
	public double train(Collection<_Doc> trainSet){
		init();
		
		double current = 0, last = -1.0, converge = 1.0;
		int iter = 1;
		
		do {
			current = EStep(trainSet);
			MStep(trainSet, iter);
			
			if (iter==1)
				converge = 1.0;
			else
				converge = (last-current)/last;
			
			last = current;
			iter ++;
		} while(iter<m_maxIter && converge>m_converge);
		
		System.out.format("NaiveBayes-EM converge to %.4f after %d iterations...\n", converge, iter);
		return last;
	}
}
