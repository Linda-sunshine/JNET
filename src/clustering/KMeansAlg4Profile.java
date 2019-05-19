package clustering;

import java.util.ArrayList;

import clustering.KMeansAlg4Vct.fvInstance;
import cc.mallet.cluster.Clustering;
import cc.mallet.cluster.KMeans;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import structures._User;

public class KMeansAlg4Profile extends KMeansAlg {

	public KMeansAlg4Profile(int classNo, int featureSize, int k) {
		super(classNo, featureSize, k);
	}
	
	FeatureVector createInstance(_User u) {
		int[] indices = u.getProfIndices();
		for(int i:indices) {
			m_dict.lookupIndex(i, true);
		}
		return new FeatureVector(m_dict, indices, u.getProfValues());
	}
	
	public double train(ArrayList<_User> users) {
		init();
		_User u;
		
		for(int i=0; i< users.size(); i++){			
			u = users.get(i);
			u.setClusterIndex(i);
			m_instances.add(new Instance(createInstance(u), null, null, u));
		}
		
		KMeans alg = new KMeans(m_instances.getPipe(), m_k, m_distance);
		Clustering result = alg.cluster(m_instances);	
		m_centroids = alg.getClusterMeans();
		m_clusters = result.getClusters();
		return 0; // we can compute the corresponding loss function
	}
	
	// Return the corresponding cluster numbers.
	public int[] assignClusters(){
		int[] clusterNos = new int[m_instances.size()];
		_User user;
		for(int i=0; i<m_clusters.length; i++){
			for(Instance ins: m_clusters[i]){
				user = (_User)ins.getSource();
				clusterNos[user.getClusterIndex()] = i;
			}
		}
		return clusterNos;
	}
}
