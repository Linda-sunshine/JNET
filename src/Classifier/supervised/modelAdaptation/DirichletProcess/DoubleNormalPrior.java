package Classifier.supervised.modelAdaptation.DirichletProcess;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;

public class DoubleNormalPrior extends NormalPrior {
	protected Normal m_2ndNormal; // Normal distribution for second half of target random variable
	double m_meanB, m_sdB;
	
	public DoubleNormalPrior(double mean1, double sd1, double mean2, double sd2) {
		super(mean1, sd1);

		m_meanB = mean2;
		m_sdB = sd2;
		m_2ndNormal = new Normal(mean2, sd2, new DoubleMersenneTwister());
	}

	@Override
	public void sampling(double[] target) {
		int i = 0;
		for(; i<target.length/2; i++)
			target[i] = m_normal.nextDouble();//first half for scaling
		
		for(; i<target.length; i++)
			target[i] = m_2ndNormal.nextDouble();//second half for shifting
	}
	
	public double logLikelihood(double[] target, double normScaleA, double normScaleB) {
		double L = 0;
		int i=0;
		for(; i<target.length/2; i++)
			L += normScaleA * (target[i]-m_meanA)*(target[i]-m_meanA)/m_sdA/m_sdA;
		
		for(; i<target.length; i++)
			L += normScaleB * (target[i]-m_meanB)*(target[i]-m_meanB)/m_sdB/m_sdB;
		return L/2;
	}
}
