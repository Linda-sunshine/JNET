package Analyzer;

import structures._RankItem;
import structures._stat;
import utils.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * 
 * @author Lin Gong
 * Implementation of several text feature selection algorithms.
 * Yang, Yiming, and Jan O. Pedersen. "A comparative study on feature selection in text categorization." ICML. Vol. 97. 1997.
 */
public class FeatureSelector {

	double m_startProb, m_endProb; // selecting features proportionally
	int m_maxDF, m_minDF; // upper and lower bounds for DF in feature selection (exclusive!)
	
	ArrayList<_RankItem> m_selectedFeatures;

	//Default setting of feature selection
	public FeatureSelector(){
		m_startProb = 0;
		m_endProb = 1;
		m_selectedFeatures = new ArrayList<_RankItem>();
	}
		
	//Given start and end of feature selection.
	public FeatureSelector(double startProb, double endProb, int maxDF, int minDF){
		if (startProb>endProb) {
			double t = startProb;
			startProb = endProb;
			endProb = t;
		}
		
		m_startProb = Math.max(0, startProb);
		m_endProb = Math.min(1.0, endProb);
		m_maxDF = maxDF<=0 ? Integer.MAX_VALUE : maxDF;
		m_minDF = minDF; 
		m_selectedFeatures = new ArrayList<_RankItem>();
	}
	
	//Return the selected features.
	public ArrayList<String> getSelectedFeatures(){
		ArrayList<String> features = new ArrayList<String>();
		Collections.sort(m_selectedFeatures);//ascending order

		int totalSize = m_selectedFeatures.size();
		System.out.format("[Info]With minDF: %d, total word size: %d \n", m_minDF, totalSize);
		System.out.format("Feature value min: %.5f, max: %.5f\n", m_selectedFeatures.get(0).m_value,
				m_selectedFeatures.get(m_selectedFeatures.size()-1).m_value);

		for(_RankItem it: m_selectedFeatures){
			System.out.println(it.m_name + '\t' + it.m_value);
		}

		int start = (int) (totalSize * m_startProb);
		int end = (int) (totalSize * m_endProb);
		for(int i=start; i<end; i++)
			features.add(m_selectedFeatures.get(i).m_name);
		
		return features;
	}
	
	//Feature Selection -- DF.
	public void DF(HashMap<String, _stat> featureStat){
		m_selectedFeatures.clear();
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			double sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if(sumDF > m_minDF && sumDF < m_maxDF)
				m_selectedFeatures.add(new _RankItem(f, sumDF));
		}
	}
		
	//Feature Selection -- IG.
	public void IG(HashMap<String, _stat> featureStat, int[] classMemberNo){
		m_selectedFeatures.clear();
		double classMemberSum = Utils.sumOfArray(classMemberNo);
		double[] PrCi = new double [classMemberNo.length];//I
		double[] PrCit = new double [classMemberNo.length];//II
		double[] PrCitNot = new double [classMemberNo.length];//III
			
		double Prt = 0, PrtNot = 0;
		double Gt = 0;//IG
		double PrCiSum = 0, PrCitSum = 0, PrCitNotSum = 0;
			
		//- $sigma$PrCi * log PrCi
		for(int i = 0; i < classMemberNo.length; i++) {
			PrCi[i] = classMemberNo[i] / classMemberSum;
			if(PrCi[i] != 0){
				PrCiSum -= PrCi[i] * Math.log(PrCi[i]);
			}
		}
		
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > m_minDF && sumDF < m_maxDF){
				_stat temp = featureStat.get(f);
				Prt = Utils.sumOfArray(temp.getDF()) / classMemberSum;
				PrtNot = 1 - Prt;
				PrCitSum = 0;
				PrCitNotSum = 0;
				for(int i = 0; i < classMemberNo.length; i++){
					PrCit[i] = ((double)temp.getDF()[i] / classMemberNo[i]) * PrCi[i] / Prt;
					PrCitNot[i] = ((double)(classMemberNo[i] - temp.getDF()[i]) / classMemberNo[i]) * PrCi[i] / PrtNot;
					if(PrCit[i] != 0){
						PrCitSum += PrCit[i] * Math.log(PrCit[i]);
					}
					if(PrCitNot[i] != 0){
						PrCitNotSum += PrCitNot[i] * Math.log(PrCitNot[i]);
					}
				}
				Gt = PrCiSum + Prt * PrCitSum + (1-Prt) * PrCitNotSum;
				m_selectedFeatures.add(new _RankItem(f, Gt));
			}
		}
	} 
		
	//Feature Selection -- MI.
	public void MI(HashMap<String, _stat> featureStat, int[] classMemberNo){
		m_selectedFeatures.clear();
		double[] PrCi = new double[classMemberNo.length];
		double[] ItCi = new double[classMemberNo.length];
		double N = Utils.sumOfArray(classMemberNo);
		double Iavg = 0;

		for (int i = 0; i < classMemberNo.length; i++)
			PrCi[i] = classMemberNo[i] / N;
		for (String f : featureStat.keySet()) {
			// Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > m_minDF && sumDF < m_maxDF) {
				Iavg = 0;
				for (int i = 0; i < classMemberNo.length; i++) {
					_stat temp = featureStat.get(f);
					double A = temp.getDF()[i];
					ItCi[i] = Math.log(A * N / classMemberNo[i]
							* Utils.sumOfArray(temp.getDF()));
					Iavg += ItCi[i] * PrCi[i];
				}
				
				m_selectedFeatures.add(new _RankItem(f, Iavg));
			}
		}
	}
		
	//Feature Selection -- CHI.
	public void CHI(HashMap<String, _stat> featureStat, int[] classMemberNo){
		m_selectedFeatures.clear();
		int classNo = classMemberNo.length;
		int N = Utils.sumOfArray(classMemberNo), sumDF;
		double[] X2tc = new double [classNo];
		double X2avg = 0;		
			
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			_stat temp = featureStat.get(f);
			sumDF = Utils.sumOfArray(temp.getDF());
			
			if (sumDF > m_minDF && sumDF < m_maxDF) {	
				X2avg = 0;				
				for(int i = 0; i < classNo; i++){
					X2tc[i] = Utils.ChiSquare(N, sumDF, temp.getDF()[i], classMemberNo[i]);
					X2avg += X2tc[i] * classMemberNo[i] / N;
				}
				//X2max = Utils.maxOfArrayValue(X2tc);
				m_selectedFeatures.add(new _RankItem(f, X2avg));
			}
		}
	}
}