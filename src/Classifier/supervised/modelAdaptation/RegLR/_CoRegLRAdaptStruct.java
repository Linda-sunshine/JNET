/**
 * 
 */
package Classifier.supervised.modelAdaptation.RegLR;

import java.util.Collection;
import java.util.LinkedList;

import Classifier.supervised.modelAdaptation.CoAdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._User;

/**
 * @author Hongning Wang
 * shared structure for CoRegLR adaptation algorithm (most of code is duplicated from _CoLinStruct since Java does not support multi-class inheritance)
 */
public class _CoRegLRAdaptStruct extends _AdaptStruct implements CoAdaptStruct {

	static double[] sharedW;//this stores shared model weights across all users	
	MyPriorityQueue<_RankItem> m_neighbors; //top-K neighborhood, we only store an asymmetric graph structure
	LinkedList<_RankItem> m_reverseNeighbors; // this user contributes to the other users' neighborhood
	int m_featureSize;//this feature size already included the bias term
	
	public _CoRegLRAdaptStruct(_User user, int id, int featureSize, int topK) {
		super(user);
		m_id = id;
		m_featureSize = featureSize;
		m_neighbors = new MyPriorityQueue<_RankItem>(topK);
		m_reverseNeighbors = new LinkedList<_RankItem>();
	}

	@Override
	public void addNeighbor(int id, double similarity) {
		m_neighbors.add(new _RankItem(id, similarity));
	}
	
	@Override
	public void addReverseNeighbor(int id, double similarity) {
		for(_RankItem it:m_neighbors) {//we need to check this condition to avoid duplicates
			if (it.m_index == id)
				return;
		}
		
		m_reverseNeighbors.add(new _RankItem(id, similarity));
	}
	
	@Override
	public Collection<_RankItem> getNeighbors() {
		return m_neighbors;
	}
	
	@Override
	public Collection<_RankItem> getReverseNeighbors() {
		return m_reverseNeighbors;
	}
	
	static public double[] getSharedW() {
		return sharedW;
	}
	
	@Override
	public double getPWeight(int n) {
		return sharedW[m_id*m_featureSize + n];
	}
	
	public void setPWeight(int n, double v) {
		sharedW[m_id*m_featureSize + n] = v;
	}
}
