package LBFGS.optimzationTest;

public class Problem2 extends QuadraticTest {

	public Problem2() {
		m_x = new double[2];
	}

	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f1 = -13 + m_x[0] + ((5-m_x[1])*m_x[1]-2)*m_x[1];
		double f2 = -29 + m_x[0] + ((1+m_x[1])*m_x[1]-14)*m_x[1];
		
		g[0] = 2*f1 + 2*f2;
		g[1] = 2*f1 * (10*m_x[1]-3*m_x[1]*m_x[1]-2) + 2*f2 * (3*m_x[1]*m_x[1]+2*m_x[1]-14);
		
		return f1*f1 + f2*f2;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f1 = -13 + x[0] + ((5-x[1])*x[1]-2)*x[1];
		double f2 = -29 + x[0] + ((1+x[1])*x[1]-14)*x[1];
		
		return f1*f1 + f2*f2;
	}

	@Override
	public void projection(double[] x) {
		
	}

	@Override
	public void reset() {
		super.reset();
		
		m_x[0] = 0.5;
		m_x[1] = -2;
	}

	@Override
	public String toString() {
		return "Problem2";
	}
}
