package LBFGS.optimzationTest;

public class Problem14 extends QuadraticTest {

	public Problem14() {
		m_x = new double[4];
		m_lbound = new double[]{-100, -100, -100, -100};
		m_ubound = new double[]{0, 10, 100, 100};
	}

	@Override
	public double calcFuncGradient(double[] g) {
		m_neval ++;
		
		double f1 = 10*(m_x[1] - m_x[0]*m_x[0]);
		double f2 = 1 - m_x[0];
		double f3 = Math.sqrt(90.0) * (m_x[3]-m_x[2]*m_x[2]);
		double f4 = 1 - m_x[2];
		double f5 = Math.sqrt(10.0) * (m_x[1]+m_x[3]-2);
		double f6 = 1.0/Math.sqrt(10.0)*(m_x[1]-m_x[3]);
		
		g[0] = 2*f1*(-20*m_x[0]) - 2*f2;
		g[1] = 2*f1*10 + 2*f5*Math.sqrt(10.0) + 2*f6/Math.sqrt(10.0);
		g[2] = 2*f3*Math.sqrt(90.0)*(-2*m_x[2]) - 2*f4;
		g[3] = 2*f3*Math.sqrt(90.0) + 2*f5*Math.sqrt(10.0) - 2*f6/Math.sqrt(10.0);
		
		return f1*f1 + f2*f2 + f3*f3 + f4*f4 + f5*f5 + f6*f6;
	}

	@Override
	public double calcFunc(double[] x) {
		m_neval ++;
		
		double f1 = 10*(x[1] - x[0]*x[0]);
		double f2 = 1.0 - x[0];
		double f3 = Math.sqrt(90.0) * (x[3]-x[2]*x[2]);
		double f4 = 1.0 - x[2];
		double f5 = Math.sqrt(10.0) * (x[1]+x[3]-2);
		double f6 = 1.0/Math.sqrt(10.0)*(x[1]-x[3]);
		
		return f1*f1 + f2*f2 + f3*f3 + f4*f4 + f5*f5 + f6*f6;
	}

	@Override
	public void reset() {
		super.reset();
		
		m_x[0] = -3;
		m_x[1] = -1;
		m_x[2] = -3;
		m_x[3] = -1;
	}

	@Override
	public String toString() {
		return "Problem14";
	}
}
