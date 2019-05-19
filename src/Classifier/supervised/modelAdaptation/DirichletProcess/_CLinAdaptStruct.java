package Classifier.supervised.modelAdaptation.DirichletProcess;

import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import structures._User;

public class _CLinAdaptStruct extends _LinAdaptStruct {

	static double[] sharedA;//this stores shared transformation operation across all uesrs	
	int m_ttlUserSize, m_clusterSize; // total user size, for indexing user cluster's transformation matrix.
	public _CLinAdaptStruct(_User user, int dim, int id, int ttlUserSize, int clusterSize) {
		super(user, dim);
		m_id = id;
		m_ttlUserSize = ttlUserSize;
		m_clusterSize = clusterSize;
	}
	
	// Get the transformation matrix for the cluster.
	public double getClusterScaling(int cid, int gid){
		if (gid<0 || gid>m_dim || cid<0 || cid>m_clusterSize) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return sharedA[gid + m_dim*2*(m_ttlUserSize + cid)];
	}
	public double getClusterShifting(int cid, int gid){
		if (gid<0 || gid>m_dim || cid<0 || cid>m_clusterSize) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return sharedA[gid + m_dim + m_dim*2*(m_ttlUserSize + cid)];
	}
	
	// Get the transformation matrix for the global part.
	public double getGlobalScaling(int gid){
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return sharedA[gid + m_dim*2*(m_ttlUserSize + m_clusterSize)];
	}
	public double getGlobalShifting(int gid){
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return sharedA[gid + m_dim + m_dim*2*(m_ttlUserSize + m_clusterSize)];
	}
	
	// Get the transformation matrix for the inidividual part.
	public double getScaling(int gid){
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return sharedA[gid + m_id*m_dim*2];
	}
	public double getShifing(int gid){
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return sharedA[gid + m_dim + m_id*m_dim*2];
	}
}
