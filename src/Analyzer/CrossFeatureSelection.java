package Analyzer;

import Classifier.BaseClassifier;
import Classifier.supervised.SVM;
import clustering.KMeansAlg4Vct;
import structures._Corpus;
import structures._Doc;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

/***
 * The class performs feature selection-cross.
 * @author lin
 *
 */
public class CrossFeatureSelection {
	int m_kFold;
	int m_kMeans;
	int m_classNo;
	int m_featureSize;
	double m_C = 1.0; //penalty of SVM.
	double[][] m_weights;
	ArrayList<_Doc> m_docs;
	
	int[] m_masks; //for shuffle
	ArrayList<ArrayList<_Doc>> m_trainSets;
	BaseClassifier m_classifier; 
	
	public CrossFeatureSelection(ArrayList<_Doc> docs, int classNo, int featureSize, int kFold, int kMeans){
		m_docs = docs;
		m_kFold = kFold;
		m_kMeans = kMeans;
		m_classNo = classNo;
		m_featureSize = featureSize;
		m_trainSets = new ArrayList<ArrayList<_Doc>>();
	}
	
	public CrossFeatureSelection(_Corpus c, int classNo, int featureSize, int kFold, int kMeans){
		m_docs = c.getCollection();
		m_kFold = kFold;
		m_kMeans = kMeans;
		m_classNo = classNo;
		m_featureSize = featureSize;
		m_trainSets = new ArrayList<ArrayList<_Doc>>();
	}
	
	public void init(){
		for(int i=0; i<m_kFold;i++)
			m_trainSets.add(new ArrayList<_Doc>());
		shuffle();
	}
	
	//Split the whole collection into k folds.
	public void splitCorpus(){
		init();
		int fold = 0;
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int j = 0; j < m_masks.length; j++) {
			fold = m_masks[j];
			m_trainSets.get(fold).add(m_docs.get(j));
		}
	}
	public void shuffle(){
		m_masks = new int[m_docs.size()];
		Random rand = new Random();
		for(int i=0; i< m_masks.length; i++) {
			this.m_masks[i] = rand.nextInt(m_kFold);
		}
	}
	// Train classifiers based on the splited training documents.
	public void train(){
		splitCorpus();
		m_weights = new double[m_kFold][];		
		for(int i=0; i < m_trainSets.size(); i++){
			m_classifier = new SVM(m_classNo, m_featureSize, m_C);
			m_classifier.train(m_trainSets.get(i));
			m_weights[i] = ((SVM) m_classifier).getWeights();
		}
		System.out.println(String.format("[Info]Finish training %d folds data!", m_kFold));
	}
	String m_filename;
	// Perform k-means on the features based on learned weights.
	public void kMeans(String filename){
		KMeansAlg4Vct kmeans = new KMeansAlg4Vct(m_weights, m_kMeans);
		kmeans.init();
		kmeans.train();
		System.out.print(String.format("%d clusters are generated.\n", kmeans.getClusterSize()));
		writeResults(filename, kmeans.getClusters(), kmeans.getClusterSize());
	}

	public void writeResults(String filename, int[] clusterNos, int size){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(int i=0; i<clusterNos.length-1; i++)
				writer.write(clusterNos[i] + ",");
			writer.write(clusterNos[clusterNos.length-1]+"\n");
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
