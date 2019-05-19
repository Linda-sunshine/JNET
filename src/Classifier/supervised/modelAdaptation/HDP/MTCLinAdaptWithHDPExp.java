package Classifier.supervised.modelAdaptation.HDP;

import Classifier.supervised.GlobalSVM;
import Classifier.supervised.SVM;
import structures.*;
import structures._Doc.rType;
import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class MTCLinAdaptWithHDPExp extends MTCLinAdaptWithHDP {
	double[] m_trainPerf;
	ArrayList<double[]> m_perfs = new ArrayList<double[]>();
	ArrayList<double[]> m_trainPerfs = new ArrayList<double[]>();

	boolean m_postCheck = false;
	
	GlobalSVM m_gsvm;
	public void trainGlobalSVM(){
		m_gsvm = new GlobalSVM(m_classNo, m_featureSize);
		m_gsvm.train(m_userList);
	}	
	
	public MTCLinAdaptWithHDPExp(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}
	public MTCLinAdaptWithHDPExp(int classNo, int featureSize, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPExp[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	public void setPosteriorSanityCheck(boolean b){
		m_postCheck = b;
	}
	
	public ArrayList<ArrayList<_Review>> collectClusterRvws(){
		HashMap<Integer, ArrayList<_Review>> clusters = new HashMap<Integer, ArrayList<_Review>>();
		_HDPThetaStar theta = null;
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct)m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				else{
					theta = r.getHDPThetaStar();
					if(clusters.containsKey(theta.getIndex()))
						clusters.get(theta.getIndex()).add(r);
					else 
						clusters.put(theta.getIndex(), new ArrayList<_Review>(Arrays.asList(r)));
				}
			}
		}
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(clusters.size());
		for(int index: clusters.keySet()){
			// sort the clusters of reviews in descending order.
			queue.add(new _RankItem(index, clusters.get(index).size()));
		}
		ArrayList<ArrayList<_Review>> sortedClusters = new ArrayList<ArrayList<_Review>>();
		for(_RankItem it: queue){
			sortedClusters.add(clusters.get(it.m_index));
		}
		
		System.out.println(String.format("Collect %d clusters of reviews for sanity check.\n", clusters.size()));
		return sortedClusters;
	}
	
	SVM m_svm;// added by Lin, svm for cross validation.
	public void CrossValidation(int kfold, int threshold){
		ArrayList<ArrayList<_Review>> sortedClusters = collectClusterRvws();
		ArrayList<double[]> prfs = new ArrayList<double[]>();
		ArrayList<Integer> sizes = new ArrayList<Integer>();
		
		// Initialize the svm for training purpose.
		m_svm = new SVM(m_classNo, m_featureSize, 1);
		for(ArrayList<_Review> cluster: sortedClusters){
			if(cluster.size() > threshold){
				sizes.add(cluster.size());
				prfs.add(CV4OneCluster(cluster, kfold));
			}
		}
		System.out.print("Size\tNeg:Precision\tRecall\t\tF1\t\tPos:Precision\tRecall\t\tF1\n");
		for(int i=0; i<prfs.size(); i++){
			double[] prf = prfs.get(i);
			System.out.print(String.format("%d\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\t%.4f+-%.4f\n", 
											sizes.get(i), prf[0], prf[6], prf[1], prf[7], prf[2], prf[8],
											prf[3], prf[9], prf[4], prf[10], prf[5], prf[11]));
		}
		System.out.println(sortedClusters.size() + " clusters in total!");
	} 
	
	public double[] CV4OneCluster(ArrayList<_Review> reviews, int kfold){
		Random r = new Random();
		int[] masks = new int[reviews.size()];
		// Assign the review fold index first.
		for(int i=0; i<reviews.size(); i++){
			masks[i] = r.nextInt(kfold);
		}
		ArrayList<_Doc> trainSet = new ArrayList<_Doc>();
		ArrayList<_Doc> testSet = new ArrayList<_Doc>();
		double[][] prfs = new double[kfold][6];
		double[] AvgVar = new double[12];
		for(int k=0; k<kfold; k++){
			trainSet.clear();
			testSet.clear();
			for(int j=0; j<reviews.size(); j++){
				if(masks[j] == k)
					testSet.add(reviews.get(j));
				else
					trainSet.add(reviews.get(j));
			}
			m_svm.train(trainSet);
			prfs[k] = test(testSet);
			// sum over all the folds 
			for(int j=0; j<6; j++)
				AvgVar[j] += prfs[k][j];
		}
		// prfs[k]: avg. calculate the average performance among different folds.
		for(int j=0; j<6; j++){
			AvgVar[j] /= (kfold*1.0);
			if(Double.isNaN(AvgVar[j]))
				System.out.println("NaN here!!");
		}

		// prfs[k+1]: var. calculate the variance among different folds.
		for(int j=0; j<6; j++){
			for(int k=0; k<kfold; k++){
				AvgVar[j+6] += (prfs[k][j] - AvgVar[j]) * (prfs[k][j] - AvgVar[j]);
			}
			AvgVar[j+6] = Math.sqrt(AvgVar[j+6]/(kfold*1.0));
			if(Double.isNaN(AvgVar[j+6]))
				System.out.println("NaN here!!");
		}
		return AvgVar;
	}
	
	public double[] test(ArrayList<_Doc> testSet){
		double[][] TPTable = new double[m_classNo][m_classNo];
		for(_Doc doc: testSet){
			int pred = m_svm.predict(doc), ans = doc.getYLabel();
			TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
		}
		
		double[] prf = new double[6];
		for (int i = 0; i < m_classNo; i++) {
			prf[3*i] = (double) TPTable[i][i] / (Utils.sumOfRow(TPTable, i) + 0.00001);// Precision of the class.
			prf[3*i + 1] = (double) TPTable[i][i] / (Utils.sumOfColumn(TPTable, i) + 0.00001);// Recall of the class.
			prf[3*i + 2] = 2 * prf[3 * i] * prf[3 * i + 1] / (prf[3 * i] + prf[3 * i + 1] + 0.00001);
		}
		return prf;
	}
	
	int m_threshold = 15;
	public void setThreshold(int t){
		m_threshold = t;
		System.out.println("[Info]Mix model threshold: "+ m_threshold);
	}
	
	public void printPerfs(){
		System.out.println("Test documents performance:");
		for(double[] perf: m_perfs){
			System.out.print(String.format("%.4f\t%.4f\n", perf[0], perf[1]));
		}
		System.out.println("Train documents performance:");
		for(double[] perf: m_trainPerfs){
			System.out.print(String.format("%.4f\t%.4f\n", perf[0], perf[1]));
		}
	}
	
	public void setSupModel(){
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
	}
}
