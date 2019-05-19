package LBFGS.optimzationTest;

import LBFGS.Optimizable;

public class Problem3 extends QuadraticTest implements Optimizable {

	public Problem3() {
		m_x = new double[2];
		m_lbound = new double[]{0,1};
		m_ubound = new double[]{1,9};
	}
	
	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f1 = 2*(10000*m_x[0]*m_x[1] - 1), f2 = 2*(Math.exp(-m_x[0]) + Math.exp(-m_x[1]) - 1.0001);
		
		g[0] = 10000 * m_x[1] * f1 - Math.exp(-m_x[0]) * f2;
		g[1] = 10000 * m_x[0] * f1 - Math.exp(-m_x[1]) * f2;
		return f1*f1 + f2*f2;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f1 = 10000*x[0]*x[1] - 1, f2 = Math.exp(-x[0]) + Math.exp(-x[1]) - 1.0001;
		return f1*f1 + f2*f2;
	}
	
	@Override
	public void reset() {
		super.reset();
		
		m_x[0] = 0;
		m_x[1] = 1;
	}

	@Override
	public String toString() {
		return "Problem3";
	}
}
