package Classifier.metricLearning;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.SolverType;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

public class LinearSVMMetricLearning extends GaussianFieldsByRandomWalk {
	
	public enum FeatureType {
		FT_diff,
		FT_cross
	}
	
	protected Model m_libModel;
	int m_bound;
	double m_L1C = 3.0; // SVM's trade-off for L1 feature selection
	double m_metricC = 1.0;// SVM's trade-off for metric learning
	
	HashMap<Integer, Integer> m_selectedFVs;
	boolean m_learningBased = true;
	FeatureType m_fvType = FeatureType.FT_diff; // has to be manually changed
	
	//Default constructor without any default parameters.
	public LinearSVMMetricLearning(_Corpus c, String classifier, double C, int bound){
		super(c, classifier, C);
		m_bound = bound;
	}
	
	public LinearSVMMetricLearning(_Corpus c, String classifier, double C, 
			double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean weightedAvg, 
			int bound) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta, weightedAvg);
		m_bound = bound;
	}
	
	public void setMetricLearningMethod(boolean opt) {
		m_learningBased = opt;
	}

	@Override
	public String toString() {
		return "LinearSVM-based Metric Learning for " + super.toString();
	}
	
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		double similarity;
		if (!m_learningBased) {
			_SparseFeature[] xi = di.getProjectedFv(), xj = dj.getProjectedFv(); 
			if (xi==null || xj==null)
				return super.getSimilarity(di, dj);//is this a good back-off?
			else
				similarity = Math.exp(Utils.dotProduct(xi, xj));
		} else {
			Feature[] fv = createLinearFeature(di, dj);
			if (fv == null)
				return super.getSimilarity(di, dj);//is this a good back-off?
			else
				similarity = Math.exp(Linear.predictValue(m_libModel, fv, 1));//to make sure this is positive
		}
		
		if (Double.isNaN(similarity)){
			System.out.println("similarity calculation hits NaN!");
			System.exit(-1);
		} else if (Double.isInfinite(similarity)){
			System.out.println("similarity calculation hits infinite!");
			System.exit(-1);
		}
		
		return similarity;			
	}
	
	@Override
	protected void init() {
		super.init();
		m_libModel = trainLibLinear(m_bound);
	}
	
	@Override
	protected void constructGraph(boolean createSparseGraph) {
		for(_Doc d:m_testSet)
			d.setProjectedFv(m_selectedFVs);
		
		super.constructGraph(createSparseGraph);
	}
	
	//using L1 SVM to select a subset of features
	void selFeatures(Collection<_Doc> trainSet, double C) {
		//use L1 regularization to reduce the feature size		
		m_libModel = SVM.libSVMTrain(trainSet, m_featureSize, SolverType.L1R_L2LOSS_SVC, C, -1);
		
		m_selectedFVs = new HashMap<Integer, Integer>();
		double[] w = m_libModel.getWeights();
		int cSize = m_classNo==2?1:m_classNo;//special treatment for binary classification
		for(int i=0; i<m_featureSize; i++) {
			for(int c=0; c<cSize; c++) {
				if (w[i*cSize+c]!=0) {//a non-zero feature
					m_selectedFVs.put(i, m_selectedFVs.size());
					break;
				}	
			}
		}
		System.out.format("Selecting %d non-zero features by L1 regularization...\n", m_selectedFVs.size());
		
		for(_Doc d:trainSet) 
			d.setProjectedFv(m_selectedFVs);
		
		if (m_debugOutput!=null) {
			try {
				for(int i=0; i<m_featureSize; i++) {
					if (m_selectedFVs.containsKey(i)) {
						m_debugWriter.write(String.format("%s(%.2f), ", m_corpus.getFeature(i), Utils.max(w, i*cSize, cSize)));
					}
				}
				m_debugWriter.write("\n");
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	//In this training process, we want to get the weight of all pairs of samples.
	protected Model trainLibLinear(int bound){
		//creating feature projection first (this is done by choosing important SVM features)
		selFeatures(m_trainSet, m_L1C);
		
		if (!m_learningBased)
			return null;
		
		int mustLink = 0, cannotLink = 0, label, PP=0, NN=0;
//		MyPriorityQueue<Double> maxSims = new MyPriorityQueue<Double>(1000, true), minSims = new MyPriorityQueue<Double>(1500, false);
		//In the problem, the size of feature size is m'*m'. (m' is the reduced feature space by L1-SVM)
		Feature[] fv;
		ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
		ArrayList<Integer> targetArray = new ArrayList<Integer>();
		
		for(int i = 0; i < m_trainSet.size(); i++){
			_Doc di = m_trainSet.get(i);
			
			for(int j = i+1; j < m_trainSet.size(); j++){
				_Doc dj = m_trainSet.get(j);
				
				if(di.getYLabel() == dj.getYLabel()) {//start from the extreme case?  && (d1.getYLabel()==0 || d1.getYLabel()==4)
					label = 1;
					
					if (di.getYLabel()==1)
						PP++;
					else
						NN++;
					
					if (PP>NN+1000)
						continue;
				} else if(Math.abs(di.getYLabel() - dj.getYLabel())>bound)
					label = -1;
				else
					continue;
				
//				double sim = super.getSimilarity(di, dj);
//				if ( (label==1 && !minSims.add(sim)) || (label==0 && !maxSims.add(sim)) )
//						continue;
//				else 
				if (label==1 && mustLink>cannotLink+2000 || label==-1 && mustLink+2000<cannotLink)
					continue;
				else if ((fv=createLinearFeature(di, dj))==null)
						continue;
				else {
					featureArray.add(fv);
					targetArray.add(label);
					
					if (label==1)
						mustLink ++;
					else
						cannotLink ++;
				}
			}
		}
		System.out.format("Generating %d must-links and %d cannot-links.\n", mustLink, cannotLink);
		
		int fSize = m_selectedFVs.size() * (1+m_selectedFVs.size())/2;		
		return SVM.libSVMTrain(featureArray, targetArray, fSize, SolverType.L2R_L1LOSS_SVC_DUAL, m_metricC, -1);
	}
	
	Feature[] createLinearFeature(_Doc d1, _Doc d2){ 
		if (m_fvType==FeatureType.FT_diff)
			return createLinearFeature_diff(d1, d2);
		else if (m_fvType==FeatureType.FT_cross)
			return createLinearFeature_cross(d1, d2);
		else
			return null;
	}
	
	//Calculate the new sample according to two documents.
	//Since cross-product will be symmetric, we don't need to store the whole matrix 
	Feature[] createLinearFeature_diff(_Doc d1, _Doc d2){
		_SparseFeature[] fv1=d1.getProjectedFv(), fv2=d2.getProjectedFv();
		if (fv1==null || fv2==null)
			return null;
		
		_SparseFeature[] diffVct = Utils.diffVector(fv1, fv2);
		
		Feature[] features = new Feature[diffVct.length*(diffVct.length+1)/2];
		int pi, pj, spIndex=0;
		double value = 0;
		for(int i = 0; i < diffVct.length; i++){
			pi = diffVct[i].getIndex();
			
			for(int j = 0; j < i; j++){
				pj = diffVct[j].getIndex();
				
				//Currently, we use one dimension array to represent V*V features 
				value = 2 * diffVct[i].getValue() * diffVct[j].getValue(); // this might be too small to count
				features[spIndex++] = new FeatureNode(getIndex(pi, pj), value);
			}
			value = diffVct[i].getValue() * diffVct[i].getValue(); // this might be too small to count
			features[spIndex++] = new FeatureNode(getIndex(pi, pi), value);
		}
		
		return features;
	}
	
	Feature[] createLinearFeature_cross(_Doc d1, _Doc d2){
		_SparseFeature[] fv1=d1.getProjectedFv(), fv2=d2.getProjectedFv();
		if (fv1==null || fv2==null)
			return null;
		
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>();
		int pi, pj, spIndex=0;
		double value = 0;
		for(int i = 0; i < fv1.length; i++){
			pi = fv1[i].getIndex();
			
			for(int j = 0; j < fv2.length; j++){
				pj = fv2[j].getIndex();
				spIndex = getIndex(pi, pj) - 1; // index issue will be taken care of in Utils.createLibLinearFV()
				value = fv1[i].getValue() * fv2[j].getValue(); // this might be too small to count
				
				//Currently, we use one dimension array to represent V*V features 
				
				if (spVct.containsKey(spIndex))
					value += spVct.get(spIndex);
				spVct.put(spIndex, value);
			}
		}
		
		return Utils.createLibLinearFV(spVct);
	}
	
	int getIndex(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return 1+i*(i+1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
}
