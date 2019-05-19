/**
 * 
 */
package topicmodels.pLSA;

import java.util.Arrays;
import java.util.LinkedList;

import structures._Corpus;
import structures._Doc;

/**
 * @author hongning
 * group the priors by category when initialization 
 */
public class pLSAGroup extends pLSA {
	double[] m_thetas;
	LinkedList<_Doc> m_group;
	
	public pLSAGroup(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha);
		
		m_thetas = new double[this.number_of_topics];
		m_group = new LinkedList<_Doc>();
	}
	
	void enforceGroupPrior() {
		Arrays.fill(m_thetas, d_alpha - 1.0);
		
		for(_Doc di:m_group) {
			for(int i=0; i<m_thetas.length; i++)
				m_thetas[i] += di.m_topics[i];
		}
		
		for(int i=0; i<m_thetas.length; i++) 
			m_thetas[i] /= 1 + m_group.size();
		
		for(_Doc di:m_group) {
			for(int i=0; i<m_thetas.length; i++) 
				di.m_sstat[i] = m_thetas[i];
		}
		
		m_group.clear();
	}

	@Override
	protected void init() { // clear up for next iteration
		for(int k=0;k<this.number_of_topics;k++)
			Arrays.fill(word_topic_sstat[k], d_beta-1.0);//pseudo counts for p(w|z)
		
		//initiate sufficient statistics
		String lastID = null;
		for(_Doc d:m_trainSet) {
			if (lastID == null)
				lastID = d.getItemID();
			else if (!d.getItemID().equals(lastID)) {
				enforceGroupPrior();
				lastID = d.getItemID();
			}
			m_group.add(d);
		}
		
		if (!m_group.isEmpty())
			enforceGroupPrior();
	}
}
