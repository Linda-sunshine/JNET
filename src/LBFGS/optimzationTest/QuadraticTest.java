package LBFGS.optimzationTest;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import LBFGS.Optimizable;

/**
 * Testing cases for numerical optimization problems
 * For both unconstrained problems:
 * 	Mor¨¦, J.J., Garbow, B.S. and Hillstrom, K.E., Testing Unconstrained Optimization Software, 
 * 	ACM Trans. Math. Software 7 (1981), 17-41.
 * And box-constrained problems:
 * 	Gay, D.M., A trust-region approach to linearly constrained optimization, pp. 72-105 
 * 	in: Numerical Analysis (Griffiths, D.F., ed.), Lecture Notes in Mathematics 1066, Springer, Berlin 1984.
 * website: http://www.mat.univie.ac.at/~neum/glopt/bounds.html
 * @author hongning
 *
 */

public abstract class QuadraticTest implements Optimizable {

	double[] m_x;
	double[] m_lbound, m_ubound; // lower bound and upper bound
	double[] m_g, m_diag;
	int m_neval;
	
	@Override
	public double[] getParameters() {
		return m_x;
	}

	@Override
	public void setParameters(double[] x) {
		if (x.length != m_x.length)
			return;
		System.arraycopy(x, 0, m_x, 0, m_x.length);
	}
	
	@Override
	public int getNumParameters() {
		return m_x.length;
	}
	
	public double calcFunc() {
		return calcFunc(m_x);
	}

	public void reset() {
		m_neval = 0;
	}
	
	void init() {
		reset();
		
		m_g = new double[m_x.length];
		m_diag = new double[m_x.length];
	}

	public double byLBFGS() {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue = 0;
		int fSize = m_x.length;
		
		init();
		
		try{
			do {
				fValue = calcFuncGradient(m_g);
				LBFGS.lbfgs(fSize, 6, m_x, fValue, m_g, false, m_diag, iprint, 1e-6, 1e-10, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
		return fValue;
	}

	@Override
	public void calcDiagnoal(double[] x) {
		m_neval ++;
	}

	@Override
	public int getNumEval() {
		return m_neval;
	}
	
	@Override
	public void projection(double[] x) {
		if (m_lbound!=null) {//check lower bound
			for(int i=0; i<x.length; i++) {
				if (x[i]<m_lbound[i])
					x[i]=m_lbound[i];
			}
		}
		
		if (m_ubound!=null) {//check lower bound
			for(int i=0; i<x.length; i++) {
				if (x[i]>m_ubound[i])
					x[i]=m_ubound[i];
			}
		}
	}
	
	public String getConstraints() {
		if (m_ubound==null && m_lbound==null)
			return null;
		
		StringBuffer buffer = new StringBuffer(128);
		for(int i=0; i<m_x.length; i++) {
			buffer.append(String.format("[%.2f, %.2f]", m_lbound!=null?m_lbound[i]:Double.NEGATIVE_INFINITY, m_ubound!=null?m_ubound[i]:Double.POSITIVE_INFINITY));
			if (i<m_x.length-1)
				buffer.append(", ");
		}
		return buffer.toString();
	}
}
