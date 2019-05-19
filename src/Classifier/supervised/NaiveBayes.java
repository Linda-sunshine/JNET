package Classifier.supervised;

import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.BaseClassifier;

public class NaiveBayes extends BaseClassifier {
	// both quantities in log space
	protected double[][] m_Pxy; // log p(X|Y)
	protected double[] m_pY;// log p(Y)
	protected boolean m_presence;
	protected double m_deltaY; // for smoothing p(Y) purpose;
	protected double m_deltaXY; // for smoothing p(X|Y) purpose;
	
	//Constructor.
	public NaiveBayes(_Corpus c){
		super(c);
		m_Pxy = new double [m_classNo][m_featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = false;
		m_deltaY = 0.1;
		m_deltaXY = 0.1;
	}
	
	//Constructor.
	public NaiveBayes(int classNo, int featureSize){
		super(classNo, featureSize);
		m_Pxy = new double [m_classNo][m_featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = false;
		m_deltaY = 0.1;
		m_deltaXY = 0.1;
	}
	
	//Constructor.
	public NaiveBayes(_Corpus c, boolean presence, double deltaY, double deltaXY){
		super(c);
		m_Pxy = new double [m_classNo][m_featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = presence;
		m_deltaY = deltaY;
		m_deltaXY = deltaXY;
	}
	
	@Override
	public String toString() {
		return String.format("Naive Bayes [C:%d, F:%d]", m_classNo, m_featureSize);
	}
	
	@Override
	protected void init() {
		for(int i=0; i<m_classNo; i++) {
			Arrays.fill(m_Pxy[i], 0);
			m_pY[i] = 0;
		}
	}
	
	//Train the data set.
	@Override
	public double train(Collection<_Doc> trainSet){
		init();
		
		for(_Doc doc: trainSet){
			int label = doc.getYLabel();
			m_pY[label] ++;
			for(_SparseFeature sf: doc.getSparse())
				m_Pxy[label][sf.getIndex()] += m_presence?1.0:sf.getValue();
		}
		
		//normalization
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY);//up to a constant since normalization of this is not important
			double sum = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sum;
		}
		return 0;//we should compute the log-likelihood
	}
		
	//Predict the label for one document.
	@Override
	public int predict(_Doc d){
		for(int i = 0; i < m_classNo; i++){
			m_cProbs[i] = m_pY[i];
			for(_SparseFeature f:d.getSparse())
				m_cProbs[i] += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue());
		}
		return Utils.argmax(m_cProbs);
	}
	
	@Override
	public double score(_Doc d, int label){
		for(int i = 0; i < m_classNo; i++){
			m_cProbs[i] = m_pY[i];
			for(_SparseFeature f:d.getSparse())
				m_cProbs[i] += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue()); // in log space
		}
		return m_cProbs[label] - Utils.logSum(m_cProbs);
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation) {
		
	}
	
	public void printTopFeatures(int topK) {
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK, true);

		for(int i=0; i<m_classNo; i++) {
			for(int n=0; n<m_featureSize; n++) 
				queue.add(new _RankItem(n, m_Pxy[i][n]));
			
			System.out.format("Class %d:\n", i);
			for(_RankItem item:queue)				
				System.out.println(m_corpus.getFeature(item.m_index));
		}
	}

	@Override
	protected void debug(_Doc d) {
		//to be implemented
	}
}
