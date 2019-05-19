package LBFGS.optimzationTest;

public class Problem1 extends QuadraticTest {

	public Problem1() {
		m_x = new double[2];
	}

	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f1 = 10*(m_x[1] - m_x[0]*m_x[0]), f2 = 1-m_x[0];
		g[0] = 2*f1*(-20*m_x[0]) - 2*f2;
		g[1] = 20*f1;
		
		return f1*f1 + f2*f2;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f1 = 10*(x[1] - x[0]*x[0]), f2 = 1-x[0];
		return f1*f1 + f2*f2;
	}

	@Override
	public void projection(double[] x) {
		
	}

	@Override
	public void reset() {
		super.reset();
		
		m_x[0] = -1.20;
		m_x[1] = 1;
	}

	@Override
	public String toString() {
		return "Problem1";
	}
}
