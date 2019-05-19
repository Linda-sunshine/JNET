package Classifier;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._PerformanceStat;
import utils.Utils;


public abstract class BaseClassifier {
	protected int m_classNo; //The total number of classes.
	protected int m_featureSize;
	protected _Corpus m_corpus;
	protected ArrayList<_Doc> m_trainSet; //All the documents used as the training set.
	protected ArrayList<_Doc> m_testSet; //All the documents used as the testing set.
	
	protected double[] m_cProbs;
	protected String[] m_features; //the detailed features
	//for cross-validation
	protected int[][] m_confusionMat, m_TPTable;//confusion matrix over all folds, prediction table in each fold
	protected ArrayList<double[][]> m_precisionsRecalls; //Use this array to represent the precisions and recalls.
	protected _PerformanceStat m_microStat; // this structure can replace the previous two arrays
	
	protected String m_debugOutput; // set up debug output (default: no debug output)
	protected BufferedWriter m_debugWriter; // debug output writer
	
	public double train() {
		return train(m_trainSet);
	}
	
	public abstract double train(Collection<_Doc> trainSet);
	public abstract int predict(_Doc doc);//predict the class label
	public abstract double score(_Doc d, int label);//output the prediction score
	protected abstract void init(); // to be called before training starts
	protected abstract void debug(_Doc d);
	
	public double test() {
		double acc = 0;
		for(_Doc doc: m_testSet){
			doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
			int pred = doc.getPredictLabel(), ans = doc.getYLabel();
			m_TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
			
			if (pred != ans) {
				if (m_debugOutput!=null && Math.random()<0.2)//try to reduce the output size
					debug(doc);
			} else {//also print out some correctly classified samples
				if (m_debugOutput!=null && Math.random()<0.02)
					debug(doc);
				acc ++;
			}
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		return acc /m_testSet.size();
	}
	
	public String getF1String() {
		double[][] PRarray = m_precisionsRecalls.get(m_precisionsRecalls.size()-1);
		StringBuffer buffer = new StringBuffer(128);
		for(int i=0; i<PRarray.length; i++) {
			double p = PRarray[i][0], r = PRarray[i][1];
			buffer.append(String.format("%d:%.3f ", i, 2*p*r/(p+r)));
		}
		return buffer.toString().trim();
	}
	
	// Constructor with given corpus.
	public BaseClassifier(_Corpus c) {
		m_classNo = c.getClassSize();
		m_featureSize = c.getFeatureSize();
		m_corpus = c;
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_microStat = new _PerformanceStat(m_classNo);
		m_debugOutput = null;
	}
	
	// Constructor with given dimensions
	public BaseClassifier(int classNo, int featureSize) {
		m_classNo = classNo;
		m_featureSize = featureSize;
		m_corpus = null;
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_microStat = new _PerformanceStat(m_classNo);
		m_debugOutput = null;
	}
	
	public void setDebugOutput(String filename) {
		if (filename==null || filename.isEmpty())
			return;
		
		File f = new File(filename);
		if(!f.isDirectory()) { 
			if (f.exists()) 
				f.delete();
			m_debugOutput = filename;
		} else {
			System.err.println("Please specify a correct path for debug output!");
		}	
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		try {
			if (m_debugOutput!=null){
				m_debugWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, false), "UTF-8"));
				m_debugWriter.write(this.toString() + "\n");
			}
			c.shuffle(k);
			int[] masks = c.getMasks();
			ArrayList<_Doc> docs = c.getCollection();
			//Use this loop to iterate all the ten folders, set the train set and test set.
			for (int i = 0; i < k; i++) {
				for (int j = 0; j < masks.length; j++) {
					//more for testing
					if( masks[j]==(i+1)%k || masks[j]==(i+2)%k ) // || masks[j]==(i+3)%k 
						m_trainSet.add(docs.get(j));
					else
						m_testSet.add(docs.get(j));
					
//					//more for training
//					if(masks[j]==i) 
//						m_testSet.add(docs.get(j));
//					else
//						m_trainSet.add(docs.get(j));
				}
				
				long start = System.currentTimeMillis();
				train();
				double accuracy = test();
				
				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
				m_trainSet.clear();
				m_testSet.clear();
			}
			calculateMeanVariance(m_precisionsRecalls);	
		
			if (m_debugOutput!=null)
				m_debugWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	abstract public void saveModel(String modelLocation);
	
	protected void calcMicroPerfStat() {
		m_microStat.calculatePRF();
		
		System.out.println("Micro confusion matrix:");
		for(int i=0; i<m_classNo; i++)
			System.out.print("\t" + i);
		System.out.println();
		
		for(int i=0; i<m_classNo; i++) {
			System.out.print(i);
			for(int j=0; j<m_classNo; j++) {
				System.out.print("\t" + m_microStat.getEntry(i, j));
			}
			System.out.println();
		}
		
		// micro average
		System.out.println("Micro F1:");
		for(int i=0; i<m_classNo; i++)
			System.out.format("Class %d: %.4f\t\t", i, m_microStat.getF1(i));
	}
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(int[][] tpTable) {
		double[][] PreRecOfOneFold = new double[m_classNo][2];
		
		for (int i = 0; i < m_classNo; i++) {
			PreRecOfOneFold[i][0] = (double) tpTable[i][i] / (Utils.sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			PreRecOfOneFold[i][1] = (double) tpTable[i][i] / (Utils.sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
		}
		
		for (int i = 0; i < m_classNo; i++) {			
			for(int j=0; j< m_classNo; j++) {
				m_confusionMat[i][j] += tpTable[i][j];
				tpTable[i][j] = 0; // clear the result in each fold
			}
		}
		return PreRecOfOneFold;
	}
	
	public void printConfusionMat() {
		for(int i=0; i<m_classNo; i++)
			System.out.format("\t%d", i);
		
		double total = 0, correct = 0;
		double[] columnSum = new double[m_classNo], prec = new double[m_classNo];
		System.out.println("\tP");
		for(int i=0; i<m_classNo; i++){
			System.out.format("%d", i);
			double sum = 0; // row sum
			for(int j=0; j<m_classNo; j++) {
				System.out.format("\t%d", m_confusionMat[i][j]);
				sum += m_confusionMat[i][j];
				columnSum[j] += m_confusionMat[i][j];
				total += m_confusionMat[i][j];
			}
			correct += m_confusionMat[i][i];
			prec[i] = m_confusionMat[i][i]/sum;
			System.out.format("\t%.4f\n", prec[i]);
		}
		
		System.out.print("R");
		for(int i=0; i<m_classNo; i++){
			columnSum[i] = m_confusionMat[i][i]/columnSum[i]; // recall
			System.out.format("\t%.4f", columnSum[i]);
		}
		System.out.format("\t%.4f", correct/total);
		
		System.out.print("\nF1");
		for(int i=0; i<m_classNo; i++)
			System.out.format("\t%.4f", 2.0 * columnSum[i] * prec[i] / (columnSum[i] + prec[i]));
		System.out.println();
	}
	
	//Calculate the mean and variance of precision and recall.
	public double[][] calculateMeanVariance(ArrayList<double[][]> prs){
		//Use the two-dimension array to represent the final result.
		double[][] metrix = new double[m_classNo][4]; 
			
		double precisionSum = 0.0;
		double precisionVarSum = 0.0;
		double recallSum = 0.0;
		double recallVarSum = 0.0;

		//i represents the class label, calculate the mean and variance of different classes.
		for(int i = 0; i < m_classNo; i++){
			precisionSum = 0;
			recallSum = 0;
			// Calculate the sum of precisions and recalls.
			for (int j = 0; j < prs.size(); j++) {
				precisionSum += prs.get(j)[i][0];
				recallSum += prs.get(j)[i][1];
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][0] = precisionSum/prs.size();
			metrix[i][1] = recallSum/prs.size();
		}

		// Calculate the sum of variances of precisions and recalls.
		for (int i = 0; i < m_classNo; i++) {
			precisionVarSum = 0.0;
			recallVarSum = 0.0;
			// Calculate the sum of precision variance and recall variance.
			for (int j = 0; j < prs.size(); j++) {
				precisionVarSum += (prs.get(j)[i][0] - metrix[i][0])*(prs.get(j)[i][0] - metrix[i][0]);
				recallVarSum += (prs.get(j)[i][1] - metrix[i][1])*(prs.get(j)[i][1] - metrix[i][1]);
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][2] = Math.sqrt(precisionVarSum/prs.size());
			metrix[i][3] = Math.sqrt(recallVarSum/prs.size());
		}
		
		// The final output of the computation.
		System.out.println("*************************************************");
		System.out.format("The final result of %s is as follows:\n", this.toString());
		System.out.println("The total number of classes is " + m_classNo);
		
		for(int i = 0; i < m_classNo; i++)
			System.out.format("Class %d:\tprecision(%.3f+/-%.3f)\trecall(%.3f+/-%.3f)\n", i, metrix[i][0], metrix[i][2], metrix[i][1], metrix[i][3]);
		
		printConfusionMat();
		return metrix;
	}
}
