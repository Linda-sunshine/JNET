package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.LinkedList;

import structures.MyPriorityQueue;
import structures._RankItem;
import structures._User;

public class _CoLinAdaptDiffFvGroupsStruct extends _CoLinAdaptStruct {

	static double[] sharedB;//this stores shared transformation operation for another class
	protected double[] m_B; // transformation matrix for another class.
	protected int m_dimB; // number of feature groups for another class.
	double[] m_pWeightsB; // Weights for the other class.
	
	
	public _CoLinAdaptDiffFvGroupsStruct(_User user, int dim, int id, int topK, int dimB) {
		super(user, dim, id, topK);
		m_id = id;
		m_neighbors = new MyPriorityQueue<_RankItem>(topK);
		m_reverseNeighbors = new LinkedList<_RankItem>();
		m_dimB = dimB;
		m_B = new double[m_dimB*2];		
		for(int i=0; i < m_dimB; i++)
			m_B[i] = 1;//Scaling in the first dim dimensions. Initialize scaling to be 1 and shifting be 0.
	}
	
	// Merge the two arrays into one so that we can pass it to LBFGS for optimization.
	public static double[] getSharedAB(){
		double[] sharedAB = new double[sharedA.length + sharedB.length];
		System.arraycopy(sharedA, 0, sharedAB, 0, sharedA.length);
		System.arraycopy(sharedB, 0, sharedAB, sharedA.length, sharedB.length);
		return sharedAB;
	}
	
	public double getScalingB(int gid){
		if (gid < 0 || gid > m_dimB) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n",gid);
			return Double.NaN;
		}
		int offset = m_id * m_dimB * 2;
		return sharedB[offset + gid];
	}
	
	// Get the shifting parameter for class B.
	public double getShiftingB(int gid){
		if (gid < 0 || gid > m_dimB) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n",gid);
			return Double.NaN;
		}

		int offset = m_id * m_dimB * 2;
		return sharedB[offset + gid + m_dimB];
	}
}
