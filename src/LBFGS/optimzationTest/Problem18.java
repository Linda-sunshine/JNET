package LBFGS.optimzationTest;

import java.util.Arrays;

public class Problem18 extends QuadraticTest {
	int m = 13;
	
	public Problem18() {
		m_x = new double[6];
		m_lbound = new double[]{0,0,0,1,0,0};
		m_ubound = new double[]{2,8,1,7,5,5};
	}

	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f, sum = 0, t, y;
		Arrays.fill(g, 0);
		for(int i=0; i<m; i++) {
			t = 0.1 * (1+i);
			y = Math.exp(-t) - 5*Math.exp(-10*t) + 3*Math.exp(-4*t);
			f = m_x[2]*Math.exp(-t*m_x[0]) 
				- m_x[3]*Math.exp(-t*m_x[1])
				+ m_x[5]*Math.exp(-t*m_x[4])
				- y;
					
			sum += f*f; 
			
			g[0] += 2*f * (-t*m_x[2]*Math.exp(-t*m_x[0]));
			g[1] -= 2*f * (-t*m_x[3]*Math.exp(-t*m_x[1]));
			g[2] += 2*f * Math.exp(-t*m_x[0]);
			g[3] -= 2*f * Math.exp(-t*m_x[1]);
			g[4] += 2*f * (-t*m_x[5]*Math.exp(-t*m_x[4]));
			g[5] += 2*f * Math.exp(-t*m_x[4]);
		}
		
		return sum;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f, sum = 0, t, y;
		for(int i=0; i<m; i++) {
			t = 0.1 * (1+i);
			y = Math.exp(-t) - 5*Math.exp(-10*t) + 3*Math.exp(-4*t);
			f = x[2]*Math.exp(-t*x[0]) 
				- x[3]*Math.exp(-t*x[1])
				+ x[5]*Math.exp(-t*x[4])
				- y;
					
			sum += f*f; 
		}
		
		return sum;
	}

	@Override
	public void reset() {
		super.reset();
		
		Arrays.fill(m_x, 1.0);
		m_x[1] = 2.0;
	}

	@Override
	public String toString() {
		return "Problem18";
	}
}
