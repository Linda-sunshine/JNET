package structures;

import java.util.Arrays;

import utils.Utils;

public class _PerformanceStat {
	
	public enum TestMode {
		TM_batch, // separate training and testing
		TM_online, // all instances are for testing
		TM_hybrid // adaptation data is also for testing beside the separated testing instances
	}
	
	int[][] m_confusionMat; // a k-by-k confusion matrix
	
	double[][] m_perfTable; // Store the performance for each new prediction result.

	//Constructor for batch mode.
	public _PerformanceStat(int[][] TPTable){
		m_confusionMat = TPTable;
		m_perfTable = new double[m_confusionMat.length][3];//column: classes; row: precision, recall, F1.
	}
	
	public _PerformanceStat(int classNo){
		m_confusionMat = new int[classNo][classNo];
		m_perfTable = new double[classNo][3];//column: 0-1 class; row: precision, recall, F1.
	}
	
	public void clear() {
		for(int i=0; i<m_confusionMat.length; i++)
			Arrays.fill(m_confusionMat[i], 0);
		
		for(int i=0; i<m_perfTable.length; i++)
			Arrays.fill(m_perfTable[i], 0);
	}
	
	public void addOnePredResult(int predL, int trueL){
		m_confusionMat[predL][trueL]++;
	}
	
	public void accumulateConfusionMat(_PerformanceStat stat) {
		for(int i=0; i<m_confusionMat.length; i++) {
			for(int j=0; j<m_confusionMat.length; j++) {
				m_confusionMat[i][j] += stat.m_confusionMat[i][j];
			}
		}
	}
	
	public double[][] getPerformanceTable() {
		return m_perfTable;
	}
	
	public void calculatePRF(){
		double PP, TP, sumP;
		for(int i=0; i<m_confusionMat.length; i++){
			// Precision.
			PP = Utils.sumOfRow(m_confusionMat, i);//predicted positives
			if (PP==0)
				m_perfTable[i][0] = 0;
			else
				m_perfTable[i][0] = m_confusionMat[i][i] / PP;
			
			// Recall.
			TP = Utils.sumOfColumn(m_confusionMat, i);//true positives
			if (TP==0)
				m_perfTable[i][1] = 0;
			else
				m_perfTable[i][1] = m_confusionMat[i][i] / TP;
			
			// F1
			sumP = m_perfTable[i][0] + m_perfTable[i][1];
			if ( sumP > 0)
				m_perfTable[i][2] = 2*m_perfTable[i][0]*m_perfTable[i][1]/sumP;
			else
				m_perfTable[i][2] = 0;	// precision=0, recall=0		
		}
	}
	
	public double getF1(int classId) {
		return m_perfTable[classId][2];
	}
	
	public int getEntry(int i, int j) {
		return m_confusionMat[i][j];
	}
	
	// How many reviews with the true class label specified.
	public int getTrueClassNo(int classId){
		return Utils.sumOfColumn(m_confusionMat, classId);
	}
	
	public double getAccuracy() {
		double acc = 0, total = 0;
		for(int i=0; i<m_confusionMat.length; i++) {
			acc += m_confusionMat[i][i];
			total += Utils.sumOfArray(m_confusionMat[i]);
		}
		
		if (total>1)
			return acc/total;
		else
			return 0;
	}
}
