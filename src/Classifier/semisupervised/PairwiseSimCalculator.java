package Classifier.semisupervised;

import structures.MyPriorityQueue;
import structures._Doc;
import structures._Node;
import structures._RankItem;

public class PairwiseSimCalculator implements Runnable {
	
	public enum ActionType {
		AT_node,
		AT_graph
	}
	
	//pointer to the Gaussian Field object to calculate similarity in parallel
	GaussianFields m_GFObj; 
	int m_start, m_end;
	ActionType m_aType;
	
	//in the range of [start, end)
	public PairwiseSimCalculator(GaussianFields obj, int start, int end, ActionType atype) {
		m_GFObj = obj;
		m_start = start;
		m_end = end;
		m_aType = atype;
	}
	
	void constructNodes() {
		_Doc d;
		for (int i = m_start; i < m_end; i++) {
			d = m_GFObj.getTestDoc(i);
			m_GFObj.m_nodeList[i] = new _Node(i, d.getYLabel(), m_GFObj.predict(d));
		}
	}
	
	void constructNearestGraph() {
		_Doc di, dj;
		double similarity;
		
		MyPriorityQueue<_RankItem> kUL = new MyPriorityQueue<_RankItem>(m_GFObj.m_k);
		MyPriorityQueue<_RankItem> kUU = new MyPriorityQueue<_RankItem>(m_GFObj.m_kPrime);
		_Node node;
		
		for (int i = m_start; i < m_end; i++) {
			di = m_GFObj.getTestDoc(i);
			node = m_GFObj.m_nodeList[i];
			
			//find the nearest unlabeled examples among all candidates since the similarity might not be symmetric
			for (int j = 0; j < m_GFObj.m_U; j++) {
				if (i==j)
					continue;
				
				dj = m_GFObj.getTestDoc(j);
				similarity = m_GFObj.getSimilarity(di, dj);
				kUU.add(new _RankItem(j, similarity));
			}
			
			for(_RankItem it:kUU) 
				node.addUnlabeledEdge(m_GFObj.m_nodeList[it.m_index], it.m_value);
			kUU.clear();

			//find the nearest labeled examples
			for (int j = 0; j < m_GFObj.m_L; j++) {
				dj = m_GFObj.getLabeledDoc(j);
				similarity = m_GFObj.getSimilarity(di, dj);
				kUL.add(new _RankItem(m_GFObj.m_U + j, similarity));
			}
			
			for(_RankItem it:kUL) 
				node.addLabeledEdge(m_GFObj.m_nodeList[it.m_index], it.m_value);
			kUL.clear();
			
			// sort the edges to accelerate debug output
			node.sortEdges();
		}	
		
		System.out.format("[%d,%d) finished...\n", m_start, m_end);
	}

	@Override
	public void run() {
		if (m_aType.equals(ActionType.AT_graph))
			constructNearestGraph();
		else
			constructNodes();
	}

}
