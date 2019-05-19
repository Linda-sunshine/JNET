/**
 * 
 */
package LBFGS;

import utils.Utils;

/**
 * @author hongning
 * Projected L-BFGS algorithm for bounded constraint optimization
 * Implementation details based on "Tackling Box-Constrained Optimization Via A New Projected Quasi-Newton Approach"
 */
public class ProjectedLBFGS extends ProjectedGradient {
	int m_M; // number of Hessian corrections
	boolean m_diag; // whether the user will provide diagnoal
	double[] m_Hdiag; // diagnoal of Hessian
	double[] m_g_old; // gradient at previous step
	double[] m_alphas; // correction coefficient
	double[] m_rhos; // scaling coefficient	
	double[][] m_ys, m_ss; // storage for differences of gradients and xs in the last M steps
	
	public ProjectedLBFGS(Optimizable objFunc, int m, boolean diag, int maxIter, double fdelta, double gdelta, double istp, double ftol, double gtol) {
		super(objFunc, maxIter, fdelta, gdelta, istp, ftol, gtol);
		
		m_M = m;
		m_diag = diag;
	}
	
	@Override
	void init() {
		super.init();
		
		m_g_old = new double[m_x.length];
		m_alphas = new double[m_M];	
		m_rhos = new double[m_M];
		m_ys = new double[m_M][m_x.length];
		m_ss = new double[m_M][m_x.length];
		
		if (m_diag)
			m_Hdiag = new double[m_x.length];
	}
	
	@Override
	public boolean optimize() {
		init();

		int k = 1; // the first step has been explore in init()
		double gNorm, xNorm, fx_old, converge;//get the initial function value and gradient
		
		//initial step for L-BFGS
		double fx = m_func.calcFuncGradient(m_g);		
		System.arraycopy(m_g, 0, m_g_old, 0, m_g.length);//store the current gradient
		getSearchDirection();//using the gradient direction as the search direction
		m_linesearch.setInitStep(1.0 / Utils.L2Norm(m_g));
		fx = m_linesearch.linesearch(fx, m_x, m_x_old, m_g, m_sd);
		updateCorrectionVcts(0);
		
		m_linesearch.setInitStep(1.0);
		do {
			fx_old = fx;
			
			quasiNewtonDirection(k);
			fx = m_linesearch.linesearch(fx, m_x, m_x_old, m_g, m_sd);
			updateCorrectionVcts(k);			
			
			gNorm = Utils.L2Norm(m_g);
			xNorm = Utils.L2Norm(m_x);
			if (fx_old==0)
				converge = 1.0;//no way to compute
			else
				converge = (fx_old-fx)/fx_old;
			
			//System.out.format("%d. f(x)=%.10f |g(x)|=%.10f converge=%.5f\n", k, fx, gNorm, converge);
		} while (++k<m_maxIter && gNorm>xNorm*m_gdelta && Math.abs(converge)>m_fdelta);
		
		return k<m_maxIter;//also need other convergence condition checking
	}
	
	void updateCorrectionVcts(int k) {
		int j = k%m_M; // index in the cache
		for(int i=0; i<m_g.length; i++) {
			//g_{k} - g_{k-1}
			m_ys[j][i] = m_g[i] - m_g_old[i];
			m_g_old[i] = m_g[i];
			
			//x_{k} - x_{k-1}
			m_ss[j][i] = m_x[i] - m_x_old[i];
			m_x_old[i] = m_x[i];
		}
		
		// to save computation
		m_rhos[j] = 1.0 / Utils.dotProduct(m_ss[j], m_ys[j]); 
	}
	
	void quasiNewtonDirection(int k) {
		getSearchDirection();
		
		//start to correct the search direction
		int m = Math.min(k, m_M), j=(k+m-1)%m;//the latest step
		for(int i=0; i<m; i++) {			
			m_alphas[j] = Utils.dotProduct(m_ss[j], m_sd) * m_rhos[j]; 
			Utils.add2Array(m_sd, m_ys[j], -m_alphas[j]);
			
			j = (j+m-1)%m; // go backward
		}
		
		if (!m_diag) {//we will use the linear scaling
			j = (k+m-1)%m; // use the latest information
			double scale = 1.0 / m_rhos[j] / Utils.dotProduct(m_ys[j], m_ys[j]); 
			Utils.scaleArray(m_sd, scale);
		} else {
			m_func.calcDiagnoal(m_Hdiag);
			for(int i=0; i<m_Hdiag.length; i++) 
				m_sd[i] *= m_Hdiag[i];
		}
		
		double beta;
		for(int i=0; i<m; i++) {
			beta = m_rhos[j] * Utils.dotProduct(m_sd, m_ys[j]);
			Utils.add2Array(m_sd, m_ss[j], m_alphas[j] - beta);
			j = (j+1)%m; // go forward
		}
		
		//correction of search direction finishes, ready for line search
	}
}
