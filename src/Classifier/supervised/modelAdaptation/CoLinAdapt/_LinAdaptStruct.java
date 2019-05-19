package Classifier.supervised.modelAdaptation.CoLinAdapt;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._User;

public class _LinAdaptStruct extends _AdaptStruct {
	
	protected double[] m_A; // transformation matrix which is 2*(k+1) dimension.
	protected int m_dim; // number of feature groups
	
	public _LinAdaptStruct(_User user, int dim) {
		super(user);
		
		m_dim = dim;
		if (dim>0) {
			m_A = new double[dim*2];		
			for(int i=0; i < m_dim; i++)
				m_A[i] = 1;//Scaling in the first dim dimensions. Initialize scaling to be 1 and shifting be 0.
		}//otherwise we will not create the space
	}	

	@Override
	public double[] getUserModel() {
		return m_A;
	}
	
	//get the shifting operation for this group
	public double getShifting(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return m_A[m_dim+gid];
	}
	
	//get the shifting operation for this group
	public double getScaling(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the range of feature grouping!\n", gid);
			return Double.NaN;
		}
		return m_A[gid];
	}
}
