/**
 * 
 */
package topicmodels.posteriorRegularization;

import java.util.Arrays;

import posteriorRegularization.logisticRegression.PosteriorConstraints;

/**
 * @author hongning
 *
 */
public class PairwiseAttributeConstraints extends PosteriorConstraints {

	public PairwiseAttributeConstraints(int tSize) {
		super(null, tSize);
		
		CONT_SIZE = C;// pairwise constraint size
		parameters = new double[C];//start from a legal point
		gradient = new double[C];
		m_b = new double[CONT_SIZE];
		m_q = new double[C];
		
		m_phi_Z_x = new double[C][CONT_SIZE];
	}
	
	public PairwiseAttributeConstraints(double[] p, double[] ss) {
		super(p, p.length);

		CONT_SIZE = C;// pairwise constraint size
		parameters = new double[C];//start from a legal point
		gradient = new double[C];
		m_b = new double[CONT_SIZE];
		m_q = new double[C];
		
		m_phi_Z_x = new double[C][CONT_SIZE];
		for(int i=0; i<C; i++) 
			m_phi_Z_x[i][i] = ss[i];
	}
	
	public void reset(double[] p, double[] ss) {
		this.m_p = p;
		for(int i=0; i<C; i++) {
			if (i%2==0)
				m_phi_Z_x[i][i] = ss[i+1];
			else
				m_phi_Z_x[i][i] = ss[i-1];
		}
		Arrays.fill(m_q, 0);
		Arrays.fill(parameters, 0);
		Arrays.fill(parameters, 0);
	}

	@Override
	protected void initiate_constraint_feature(int label) {
		
	}

	@Override
	public String toString() {
		return "Pairwise Attribute Constraint for topic models";
	}
}
