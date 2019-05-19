package LBFGS.optimzationTest;

import java.util.Arrays;

import LBFGS.Optimizable;

public class Problem6 extends QuadraticTest implements Optimizable {

	int m = 10;
	
	public Problem6() {
		m_x = new double[2];
	}
	
	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f, sum = 0;
		Arrays.fill(g, 0);
		for(int i=1; i<=m; i++){
			f = 2*(i+1) - Math.exp(i*m_x[0]) - Math.exp(i*m_x[1]);
			g[0] -= 2*f * i*Math.exp(i*m_x[0]);
			g[1] -= 2*f * i*Math.exp(i*m_x[1]);
			
			sum += f*f;
		}
		return sum;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f, sum = 0;
		for(int i=1; i<=m; i++){
			f = 2*(i+1) - Math.exp(i*x[0]) - Math.exp(i*x[1]);
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
		return "Problem6";
	}
}
