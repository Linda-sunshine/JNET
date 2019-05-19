/**
 * 
 */
package structures;

/**
 * @author hongning
 * An edge in random walk graph
 */
public class _Edge implements Comparable<_Edge> {

	double m_similarity; // similarity between the nodes 
	_Node m_node; // pointer to the neighboring node
	
	public _Edge(_Node n, double sim) {
		m_node = n;
		m_similarity = sim;
	}
	
	public _Edge() {
		m_similarity = -1;
		m_node = null;
	}
	
	public int getNodeId() {
		return m_node.m_id;
	}
	
	public double getLabel() {
		return m_node.m_label;
	}
	
	public double getPred() {
		return m_node.m_pred;
	}
	
	public double getClassifierPred() {
		return m_node.m_classifierPred;
	}
	
	public double getSimilarity() {
		return m_similarity;
	}

	@Override
	public int compareTo(_Edge e) {
		if (e.m_similarity>this.m_similarity)
			return 1;
		else if (e.m_similarity<this.m_similarity)
			return -1;
		else
			return 0;
	}

}
