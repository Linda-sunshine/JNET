package structures;

import java.util.ArrayList;
import java.util.Arrays;

public class _thetaStar implements Comparable<_thetaStar> {
	int m_index;
	int m_dim;
	int m_memSize;

	double[] m_beta;
	double m_proportion;
	
	double m_pCount, m_nCount; // number of positive and negative documents in this cluster
	ArrayList<double[]> m_betas = new ArrayList<double[]>();
	
	public _thetaStar(int dim){
		m_dim = dim;
		m_memSize = 0;
		m_beta = new double[m_dim];
	}
	
	public void addOneBeta(double[] b){
		m_betas.add(b);
	}
	
	public int getMemSize(){
		return m_memSize;
	}
	
	public void updateMemCount(int c){
		m_memSize += c;
	}
	
	public void setProportion(double p) {
		m_proportion = p;
	}
	
	public double getProportion() {
		return m_proportion;
	}
	
	public void setIndex(int i){
		m_index = i;
	}
	
	public int getIndex(){
		return m_index;
	}
	
	public double[] getModel() {
		return m_beta;
	}
	
	public void resetCount() {
		m_pCount = 0;
		m_nCount = 0;
	}
	
	public void incPosCount() {
		m_pCount++;
	}
	
	public void incNegCount() {
		m_nCount++;
	}
	
	public String showStat() {
		return String.format("%d(%.2f,%.1f)", m_memSize, m_pCount/(m_pCount+m_nCount), (m_pCount+m_nCount)/m_memSize);
	}
	@Override
	public int compareTo(_thetaStar o) {
		return o.getMemSize() - m_memSize;
	}
	
	public ArrayList<double[]> getAllModels(){
		return m_betas;
	}
	
	// pWeights is the model weights of sentiment model
	// while model is just for linear transformation parameters
	private double[] m_pWeights;
	public void setWeights(double[] ws){
		m_pWeights = ws;
	}
	public double[] getWeights(){
		return m_pWeights;
	}
	// reset the thetaStar
	public void reset(){
		m_index = -1;
		m_memSize = 0;
		m_proportion = 0;
		m_pCount = 0;
		m_nCount = 0;
		m_betas.clear();
		// keep the beta or not?
		Arrays.fill(m_beta, 0);

	}
}