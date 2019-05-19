/**
 * 
 */
package structures;

import java.util.ArrayList;
import java.util.Collections;

/**
 * @author hongning
 * A node in random walk graph
 */
public class _Node {
	
	int m_id; // index in the original document list
	public ArrayList<_Edge> m_labeledEdges; // edge to the labeled neighbors
	public ArrayList<_Edge> m_unlabeledEdges; // edge to the unlabeled neighbors
	public double m_label; // ground-truth label
	public double m_pred; // predicted label (assigned by random walk)
	public double m_classifierPred; // classifier's prediction (assigned by classifier)
	
	public _Node(int id, double label, double classifier) {
		m_id = id;
		m_labeledEdges = null;
		m_unlabeledEdges = null;
		m_label = label;
		m_classifierPred = classifier;
		m_pred = classifier; // start from classifier's prediction
	}

	public void addLabeledEdge(_Node n, double sim) {
		if (m_labeledEdges == null)
			m_labeledEdges = new ArrayList<_Edge>();
		
		m_labeledEdges.add(new _Edge(n, sim));
	}
	
	public void addUnlabeledEdge(_Node n, double sim) {
		if (m_unlabeledEdges == null)
			m_unlabeledEdges = new ArrayList<_Edge>();
		
		m_unlabeledEdges.add(new _Edge(n, sim));
	}
	
	public void sortEdges() {
		Collections.sort(m_labeledEdges);
		Collections.sort(m_unlabeledEdges);
	}
	
	public double weightAvgInLabeledNeighbors() {
		double wijSumL = 0, fSumL = 0;
		for (_Edge edge:m_labeledEdges) {				
			wijSumL += edge.getSimilarity(); //get the similarity between two nodes.
			fSumL += edge.getSimilarity() * edge.getLabel();
		}
		
		return fSumL / wijSumL;
	}
	
	public double weightAvgInUnlabeledNeighbors() {
		double wijSumU = 0, fSumU = 0;
		for (_Edge edge:m_unlabeledEdges) {				
			wijSumU += edge.getSimilarity(); //get the similarity between two nodes.
			fSumU += edge.getSimilarity() * edge.getPred();
		}
		
		return fSumU / wijSumU;
	}
}

