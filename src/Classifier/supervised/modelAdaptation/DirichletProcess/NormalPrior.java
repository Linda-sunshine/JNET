package Classifier.supervised.modelAdaptation.DirichletProcess;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;

public class NormalPrior {
	protected Normal m_normal; // Normal distribution.
	double m_meanA, m_sdA, m_meansA[];
	
	public NormalPrior (double mean, double sd) {
		m_meanA = mean;
		m_sdA = sd;
		m_normal = new Normal(mean, sd, new DoubleMersenneTwister());
	}
	
	public NormalPrior (double[] mean, double sd) {
		m_meansA = mean;
		m_sdA = sd;
		m_normal = new Normal(0, sd, new DoubleMersenneTwister());//we will set the means later
	}
	
	public boolean hasVctMean() {
		return m_meansA != null;
	}
	
	public void sampling(double[] target) {
		for(int i=0; i<target.length; i++) {
			if (m_meansA==null)//we have not specified the mean vector yet
				target[i] = m_normal.nextDouble();
			else
				target[i] = m_normal.nextDouble(m_meansA[i], m_sdA);
		}
	}
	
	public void sampling(double[] mean, double[] target) {
		for(int i=0; i<target.length; i++) {
			target[i] = m_normal.nextDouble(mean[i], m_sdA);
		}
	}
	
	public double logLikelihood(double[] target, double normScaleA, double normScaleB) {
		double L = 0;
		for(int i=0; i<target.length; i++) {
			if (m_meansA==null)
				L += (target[i]-m_meanA)*(target[i]-m_meanA)/m_sdA/m_sdA;
			else
				L += (target[i]-m_meansA[i])*(target[i]-m_meansA[i])/m_sdA/m_sdA;
		}
		return normScaleA * L / 2;
	}
	
	public double logLikelihood(double[] mean, double[] target, double normScale) {
		double L = 0;
		for(int i=0; i<target.length; i++) 
			L += (target[i]-mean[i])*(target[i]-mean[i])/m_sdA/m_sdA;
		
		return normScale * L / 2;
	}
}
