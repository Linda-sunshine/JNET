/**
 * 
 */
package LBFGS.linesearch;

import LBFGS.Optimizable;
import utils.Utils;

/**
 * @author hongning
 *
 */
public class LineSearchBacktracking extends LineSearch {
	
	public enum Converge {
		CT_Armijo,
		CT_Wolfe,
		CT_Wolfe_Strong
	}
	
	Converge m_cond; // converge condition

	public LineSearchBacktracking(Optimizable func, double initStp, double ftol, double gtol, int maxStep) {
		super(func, initStp, ftol, gtol, maxStep);
		m_cond = Converge.CT_Wolfe_Strong; // simplest condition
	}

	@Override
	public double linesearch(double fx, double[] x, double[] xp, double[] g, double[] sd) {
		
		double dg_init = Utils.dotProduct(sd, g), t = m_ftol * dg_init, value = 0, dg;
		double width = 1.0, stp = m_istp;
		int count = 0;
		
		if (dg_init>0)
			return fx; // incorrect search direction

		while (++count<m_maxStep && stp<MAX_STEP && stp>MIN_STEP) {
			//step along the search direction
			for(int i=0; i<x.length; i++)
				x[i] = xp[i] + stp * sd[i];
			m_func.projection(x);
			
			//compute function value and gradient at this new point
			value = m_func.calcFuncGradient(g);
			if (value > fx+stp*t)
				width = DEC;
			else if (m_cond == Converge.CT_Armijo)
				break; // already satisfying Armijo condition (sufficient function value descent)
			else { 
				dg = Utils.dotProduct(g, sd); 
				if (dg < m_gtol * dg_init)
					width = INC;
				else { 
					if (m_cond == Converge.CT_Wolfe)
						break; // already satisfying Wolfe condition (sufficient gradient reduction on search direction)
					else if (dg > -m_gtol * dg_init)
						width = DEC;
					else
						break; // satisfying strong Wolfe condition (sufficient gradient reduction on search direction)
				}
			}
			
			//update step size
			stp *= width;
		}
		
		return value;
	}

}
