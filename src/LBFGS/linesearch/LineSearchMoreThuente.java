/**
 * 
 */
package LBFGS.linesearch;

import LBFGS.Optimizable;
import utils.Utils;

/**
 * @author hongning
 * Line searching algorithm by More and Thuente
 * J. More, David J. Thuente: original Fortran version,
 *   as part of Minpack project. Argonne Nat'l Laboratory, June 1983.
 *   Robert Dodier: Java translation, August 1997.
 */
public class LineSearchMoreThuente extends LineSearch {
	double m_xtol;
	boolean m_bracket;
	
	public LineSearchMoreThuente(Optimizable func, double initStep,
			double ftol, double gtol, double xtol, int maxStep) {
		super(func, initStep, ftol, gtol, maxStep);
		m_xtol = xtol;
	}

	@Override
	public double linesearch(double finit, double[] x, double[] xp, double[] g, double[] sd) {
		double dg_init = Utils.dotProduct(sd, g), t = m_ftol * dg_init, value = 0, dg;
		if (dg_init>=0)
			return finit; // incorrect search direction
		
		double width = MAX_STEP - MIN_STEP, pwidth = 2*width;
		double[] stp = {m_istp};
		boolean stage = true, intervalUpdate = true;
		double stmin, stmax;
		m_bracket = false;
		
	    /*
        The variables stx, fx, dgx contain the values of the step,
        function, and directional derivative at the best step.
        The variables sty, fy, dgy contain the value of the step,
        function, and derivative at the other endpoint of
        the interval of uncertainty.
        The variables stp, f, dg contain the values of the step,
        function, and derivative at the current step.
	    */
		
		double[] stx = {0}, sty = {0}, fx = {finit}, fy = {finit}, dgx = {dg_init}, dgy = {dg_init};
		double[] fxm = {0}, dgxm = {0}, fym = {0}, dgym = {0}; double fm, dgm;
		
		int count = 0;
		while (++count<m_maxStep) {
			//Set the minimum and maximum steps corresponding to the present interval of uncertainty.
	        if (m_bracket) {
	            stmin = Math.min(stx[0], sty[0]);
	            stmax = Math.max(stx[0], sty[0]);
	        } else {
	            stmin = stx[0];
	            stmax = stp[0] + 4.0 * (stp[0] - stx[0]);
	        }
	        
	        if (stp[0]>MAX_STEP)
	        	stp[0] = MAX_STEP;
	        if (stp[0]<MIN_STEP)
	        	stp[0] = MIN_STEP;
	        
	        // If an unusual termination is to occur then let stp be the lowest point obtained so far.
	        if ( (m_bracket && (stp[0] <= stmin || stp[0] >= stmax))
	        		|| m_maxStep <= count + 1
	        		|| intervalUpdate == false
	        		|| (m_bracket && stmax - stmin <= m_xtol * stmax) ) {
	            stp[0] = stx[0];
	        }
	        
	        //step along the search direction
	        for(int i=0; i<x.length; i++)
				x[i] = xp[i] + stp[0] * sd[i];
	        m_func.projection(x);
	        
	        value = m_func.calcFuncGradient(g);
	        dg = Utils.dotProduct(g, sd); 
	        if (value <= finit+stp[0]*t && Math.abs(dg) <= m_gtol * (-dg_init)) {
	            // The sufficient decrease condition and the directional derivative condition hold.
	            return value;
	        }
	        
	        //In the first stage we seek a step for which the modified function has 
	        //a nonpositive value and nonnegative derivative.
	        if (stage && value <= finit+stp[0]*t && Math.min(m_ftol, m_gtol) * dg_init <= dg)
	            stage = false;
	        
	        /*
            A modified function is used to predict the step only if
            we have not obtained a step for which the modified
            function has a nonpositive function value and nonnegative
            derivative, and if a lower function value has been
            obtained but the decrease is not sufficient.
	        */
	        if (stage && finit+stp[0]*t < value && value <= fx[0]) {
	            /* Define the modified function and derivative values. */
	            fm = value - stp[0] * t;
	            fxm[0] = fx[0] - stx[0] * t;
	            fym[0] = fy[0] - sty[0] * t;
	            dgm = dg - t;
	            dgxm[0] = dgx[0] - t;
	            dgym[0] = dgy[0] - t;
	
	            /*
	                Call update_trial_interval() to update the interval of
	                uncertainty and to compute the new step.
	             */
	            intervalUpdate = update_trial_interval(
	                stx, fxm, dgxm,
	                sty, fym, dgym,
	                stp, fm, dgm,
	                stmin, stmax);
	
	            /* Reset the function and gradient values for f. */
	            fx[0] = fxm[0] + stx[0] * t;
	            fy[0] = fym[0] + sty[0] * t;
	            dgx[0] = dgxm[0] + t;
	            dgy[0] = dgym[0] + t;
	        } else {
	        	// Call update_trial_interval() to update the interval of uncertainty and to compute the new step.
	        	intervalUpdate = update_trial_interval(
	                stx, fx, dgx,
	                sty, fy, dgy,
	                stp, value, dg,
	                stmin, stmax);
	        }
	        
	        // Force a sufficient decrease in the interval of uncertainty.
	        if (m_bracket) {
	            if (0.66 * pwidth <= Math.abs(sty[0] - stx[0]))
	                stp[0] = stx[0] + 0.5 * (sty[0] - stx[0]);
	            pwidth = width;
	            width = Math.abs(sty[0] - stx[0]);
	        }
		}
		
		return value;
	}

	/**
	 * Update a safeguarded trial value and interval for line search.
	 *
	 *  The parameter x represents the step with the least function value.
	 *  The parameter t represents the current step. This function assumes
	 *  that the derivative at the point of x in the direction of the step.
	 *  If the bracket is set to true, the minimizer has been bracketed in
	 *  an interval of uncertainty with endpoints between x and y.
	 *
	 *  @param  x       The pointer to the value of one endpoint.
	 *  @param  fx      The pointer to the value of f(x).
	 *  @param  dx      The pointer to the value of f'(x).
	 *  @param  y       The pointer to the value of another endpoint.
	 *  @param  fy      The pointer to the value of f(y).
	 *  @param  dy      The pointer to the value of f'(y).
	 *  @param  t       The pointer to the value of the trial value, t.
	 *  @param  ft      The pointer to the value of f(t).
	 *  @param  dt      The pointer to the value of f'(t).
	 *  @param  tmin    The minimum value for the trial value, t.
	 *  @param  tmax    The maximum value for the trial value, t.
	 *  @param  brackt  The pointer to the predicate if the trial value is
	 *                  bracketed.
	 *  @retval int     Status value. Zero indicates a normal termination.
	 *  
	 *  @see
	 *      Jorge J. More and David J. Thuente. Line search algorithm with
	 *      guaranteed sufficient decrease. ACM Transactions on Mathematical
	 *      Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
	 */	
	boolean update_trial_interval(double[] x, double[] fx, double[] dx,
			double[] y, double[] fy, double[] dy,
			double[] t, double ft, double dt,
			double tmin, double tmax) {
		
		if ( m_bracket && 
				( (t[0] <= Math.min(x[0], y[0]) || t[0] >= Math.max(x[0], y[0]))  
				|| dx[0] * (t[0] - x[0]) >= 0.0 
				|| tmax < tmin) ) 
			return false;
		
		int bound;
		double mc, mq, newt;
		boolean dsign = (dx[0]*dt<0);
		
		if (fx[0]<ft) {
			/*
            Case 1: a higher function value.
            The minimum is bracket. If the cubic minimizer is closer
            to x than the quadratic one, the cubic one is taken, else
            the average of the minimizers is taken.
			*/
			
			m_bracket = true;
	        bound = 1;
	        mc = cubic_minimizer(x[0], fx[0], dx[0], t[0], ft, dt);
	        mq = quard_minimizer(x[0], fx[0], dx[0], t[0], ft);
	        if (Math.abs(mc - x[0]) < Math.abs(mq - x[0]))
	            newt = mc;
	        else
	            newt = mc + 0.5 * (mq - mc);
		} else if (dsign) {
			/*
            Case 2: a lower function value and derivatives of
            opposite sign. The minimum is bracket. If the cubic
            minimizer is closer to x than the quadratic (secant) one,
            the cubic one is taken, else the quadratic one is taken.
	         */
			m_bracket = true;
	        bound = 0;
	        mc = cubic_minimizer(t[0], ft, dt, x[0], fx[0], dx[0]);
	        mq = quard_minimizer(x[0], dx[0], t[0], dt);
	        
	        if (Math.abs(mc-t[0]) > Math.abs(mq-t[0]))
	            newt = mc;
	        else
	            newt = mq;
		} else if (Math.abs(dt) < Math.abs(dx[0])) {
	        /*
	            Case 3: a lower function value, derivatives of the
	            same sign, and the magnitude of the derivative decreases.
	            The cubic minimizer is only used if the cubic tends to
	            infinity in the direction of the minimizer or if the minimum
	            of the cubic is beyond t. Otherwise the cubic minimizer is
	            defined to be either tmin or tmax. The quadratic (secant)
	            minimizer is also computed and if the minimum is brackt
	            then the the minimizer closest to x is taken, else the one
	            farthest away is taken.
	         */
	        bound = 1;
	        mc = cubic_minimizer(x[0], fx[0], dx[0], t[0], ft, dt, tmin, tmax);
	        mq = quard_minimizer(x[0], dx[0], t[0], dt);
	        if (m_bracket) {
	            if (Math.abs(t[0] - mc) < Math.abs(t[0] - mq))
	                newt = mc;
	            else
	                newt = mq;
	        } else {
	            if (Math.abs(t[0] - mc) > Math.abs(t[0] - mq))
	                newt = mc;
	            else
	                newt = mq;
	        }
	    } else {
	    	/*
            Case 4: a lower function value, derivatives of the
            same sign, and the magnitude of the derivative does
            not decrease. If the minimum is not brackt, the step
            is either tmin or tmax, else the cubic minimizer is taken.
	        */
	        bound = 0;
	        if (m_bracket)
	        	newt = cubic_minimizer(t[0], ft, dt, y[0], fy[0], dy[0]);
	        else if (x[0] < t[0])
	            newt = tmax;
	        else
	            newt = tmin;
	    }
		
		
		/*
        Update the interval of uncertainty. This update does not
        depend on the new step or the case analysis above.

        - Case a: if f(x) < f(t),
            x <- x, y <- t.
        - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
            x <- t, y <- y.
        - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0, 
            x <- t, y <- x.
	    */
	    if (fx[0] < ft) {
	        /* Case a */
	        y[0] = t[0];
	        fy[0] = ft;
	        dy[0] = dt;
	    } else {
	        /* Case c */
	        if (dsign) {
	            y[0] = x[0];
	            fy[0] = fx[0];
	            dy[0] = dx[0];
	        }
	        /* Cases b and c */
	        x[0] = t[0];
	        fx[0] = ft;
	        dx[0] = dt;
	    }
	    
	    /* Clip the new trial value in [tmin, tmax]. */
	    if (newt > tmax) 
	    	newt = tmax;
	    if (newt < tmin) 
	    	newt = tmin;
	    
	    /*
        Redefine the new trial value if it is close to the upper bound
        of the interval.
	     */
	    if (m_bracket && bound==1) {
	        mq = x[0] + 0.66 * (y[0] - x[0]);
	        if (x[0] < y[0]) {
	            newt = Math.min(newt, mq);
	        } else {
	        	newt = Math.max(newt, mq);
	        }
	    }
	
	    /* Return the new trial value. */
	    t[0] = newt;
	    
	    return true;
	}
	
	/**
	 * Find a minimizer of an interpolated cubic function.
	 *  @param  cm      The minimizer of the interpolated cubic.
	 *  @param  u       The value of one point, u.
	 *  @param  fu      The value of f(u).
	 *  @param  du      The value of f'(u).
	 *  @param  v       The value of another point, v.
	 *  @param  fv      The value of f(v).
	 *  @param  du      The value of f'(v).
	 */
	double cubic_minimizer(double u, double fu, double du, double v, double fv, double dv) {
		double d = v - u;
	    double theta = (fu - fv) * 3 / d + du + dv;
	    double p = Math.abs(theta);
	    double q = Math.abs(du);
	    double r = Math.abs(dv);
	    double s = Math.max(Math.max(p, q), r);
	    double a = theta / s;
	    double gamma = s * Math.sqrt(a*a - (du/s) * (dv/s));
	    if (v < u) 
	    	gamma = -gamma;
	    p = gamma - du + theta;
	    q = 2*gamma - du + dv;
	    r = p / q;
	    return u + r * d;
	}
	
	/**
	 * Find a minimizer of an interpolated cubic function.
	 *  @param  cm      The minimizer of the interpolated cubic.
	 *  @param  u       The value of one point, u.
	 *  @param  fu      The value of f(u).
	 *  @param  du      The value of f'(u).
	 *  @param  v       The value of another point, v.
	 *  @param  fv      The value of f(v).
	 *  @param  du      The value of f'(v).
	 *  @param  xmin    The maximum value.
	 *  @param  xmin    The minimum value.
	 */
	double cubic_minimizer(double u, double fu, double du, double v, double fv, double dv, double xmin, double xmax) {
	    double d = v - u;
	    double theta = (fu - fv) * 3 / d + du + dv;
	    double p = Math.abs(theta);
	    double q = Math.abs(du);
	    double r = Math.abs(dv);
	    double s = Math.max(p, Math.max(q, r));
	    double a = theta / s;
	    double gamma = s * Math.sqrt(Math.max(0, a * a - (du / s) * (dv / s)));
	    if (u < v)
	    	gamma = -gamma;
	    p = gamma - dv + theta;
	    q = 2*gamma - dv + du;
	    r = p / q;
	    if (r < 0 && gamma != 0)
	        return v - r * d;
	    else if (a > 0)
	        return xmax;
	    else
	        return xmin;
	}
	
	/**
	 * Find a minimizer of an interpolated quadratic function.
	 *  @param  qm      The minimizer of the interpolated quadratic.
	 *  @param  u       The value of one point, u.
	 *  @param  fu      The value of f(u).
	 *  @param  du      The value of f'(u).
	 *  @param  v       The value of another point, v.
	 *  @param  fv      The value of f(v).
	 */
	double quard_minimizer(double u, double fu, double du, double v, double fv) {
	    double a = v - u;
	    return u + du / ((fu - fv) / a + du) / 2 * a;
	}
	
	/**
	 * Find a minimizer of an interpolated quadratic function.
	 *  @param  qm      The minimizer of the interpolated quadratic.
	 *  @param  u       The value of one point, u.
	 *  @param  du      The value of f'(u).
	 *  @param  v       The value of another point, v.
	 *  @param  dv      The value of f'(v).
	 */
	double quard_minimizer(double u, double du, double v, double dv) {
	    double a = u - v;
	    return v + dv / (dv - du) * a;
	}
}	
