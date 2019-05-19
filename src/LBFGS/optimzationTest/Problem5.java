package LBFGS.optimzationTest;

import java.util.Arrays;

import LBFGS.Optimizable;

public class Problem5 extends QuadraticTest implements Optimizable {
	double[] m_Y = new double[]{1.5, 2.25, 2.625};
	
	public Problem5() {
		m_x = new double[2];
		m_lbound = new double[]{0.6,0.5};
		m_ubound = new double[]{10,100};
	}
	
	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f, sum = 0;
		Arrays.fill(g, 0);
		for(int i=0; i<m_Y.length; i++) {
			f = m_Y[i] - m_x[0] * (1-Math.pow(m_x[1], i+1));
			
			g[0] += 2 * f * (Math.pow(m_x[1], i+1) - 1);
			g[1] += 2 * f * m_x[0] * (i+1) * Math.pow(m_x[1], i);
			
			sum += f*f;
		}
		return sum;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f, sum = 0;
		for(int i=0; i<m_Y.length; i++){
			f = m_Y[i] - x[0] * (1-Math.pow(x[1], i+1));
			sum += f*f;
		}
		return sum;
	}
	
	@Override
	public void reset() {
		super.reset();
		
		m_x[0] = 1;
		m_x[1] = 1;
	}
	
	@Override
	public String toString() {
		return "Problem5";
	}
}
