/**
 * 
 */
package LBFGS.linesearch;

import LBFGS.Optimizable;

/**
 * @author hongning
 * base class for line search algorithms
 */
abstract public class LineSearch {

	Optimizable m_func;
	double m_istp; // current step size
	double m_ftol; // sufficient descent for function value
	double m_gtol; // sufficient reduction of gradient on search direction
	int m_maxStep;
	
	final static double INC = 1.5, DEC = 0.2;
	final static double MAX_STEP = 1e32, MIN_STEP = 1e-32;
	
	public LineSearch(Optimizable func, double initStep, double ftol, double gtol, int maxStep) {
		m_func = func;
		m_istp = initStep;
		m_maxStep = maxStep;
		
		if (ftol>=gtol || ftol>=1 || gtol<=0) {
			System.err.print("[Error]Inappropriate parameter setting! It should be 0<ftol<gtol<1!\n [Warning]Using the default setting for ftol and gtol!");
			m_ftol = 1e-4;
			m_gtol = 5e-4;
		} else {
			m_ftol = ftol;
			m_gtol = gtol;
		}
	}
	
	public void setInitStep(double step) {
		m_istp = step;
	}

	public abstract double linesearch(double fx, double[] x, double[] xp, double[] g, double[] sd);
}
