package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

import utils.Utils;

/**
 * The structure wraps both \phi(_thetaStar) and \psi.
 * @author lin
 */
public class _HDPThetaStar extends _thetaStar {
	// beta in _thetaStar is \phi used in HDP.

	// for each B_gh, we need to record the theta_h, the edge count and corresponding B value
	// thus, we define a data structure to represent it.
	static public class _Connection{
		_HDPThetaStar m_theta_h;
		int[] m_edge; // the edge count assigned to B_gh
		double m_prob;

		public _Connection(_HDPThetaStar theta_h){
			m_theta_h = theta_h;
			m_edge = new int[2];
			m_prob = 0;
		}

		public int[] getEdge(){
			return m_edge;
		}

		public int getEdgeCount(int e){
			return m_edge[e];
		}

		public void updateEdgeCount(int e, int delta){
			m_edge[e] += delta;
		}
		// set B_gh
		public void setProb(double p){
			m_prob = p;
		}

		public double getProb(){
			return m_prob;
		}

	}
	private double m_gamma;

	// This variable is used to decide whether the theta star is valid or not.
	private boolean m_isValid = false;

	/****edge is indicator for one direction, i->j \in g (current theta),
	 ****connection contains two indicators: i->j \in g and j->i \in h.****/
	// Parameters used in MMB model.
	protected int m_edgeSize[];//0: zero-edge count;1: one-edge count.

	// The count of the features inside clusters.
	protected double[] m_lmStat = null;

	// key: theta, value: corresponding edge counts.
	protected HashMap<_HDPThetaStar, _Connection> m_connectionMap;

	protected ArrayList<String> m_reviewNames = new ArrayList<String>();

	// Total number of local groups in the component, in log space.
	public int m_hSize;

	// for debug purpose add the perf stat for each cluster
	protected _PerformanceStat m_perfStat;


	public _HDPThetaStar(int dim) {
		super(dim);
		m_gamma = 0;
		m_edgeSize = new int[2];
		m_connectionMap = new HashMap<_HDPThetaStar, _Connection>();
	}

	public _HDPThetaStar(int dim, double gamma) {
		super(dim);
		m_gamma = gamma;
		m_edgeSize = new int[2];
		m_connectionMap = new HashMap<_HDPThetaStar, _Connection>();
	}

	// enable is called when the theta is counted as one of the current thetas
	// theta may be created as the auxiliary variables
	public void enable(){
		m_isValid = true;
	}

	public void disable(){
		m_isValid = false;
		m_lmStat = null;
		m_gamma = 0;
	}

	public boolean isValid(){
		return m_isValid;
	}

	public void initLMStat(int lmDim){
		if(m_lmStat == null)
			m_lmStat = new double[lmDim];
		else
			Arrays.fill(m_lmStat, 0);
	}

	// reset lmStat to null, for likelihoodX calculation.
	public void resetLMStat(){
		m_lmStat = null;
	}
	public void clearLMStat(){
		Arrays.fill(m_lmStat, 0);
	}

	public void addLMStat(_SparseFeature[] fvs){
		for(_SparseFeature fv: fvs){
			m_lmStat[fv.getIndex()] += fv.getValue();
		}
	}

	public void rmLMStat(_SparseFeature[] fvs){
		for(_SparseFeature fv: fvs){
			m_lmStat[fv.getIndex()] -= fv.getValue();
			if(m_lmStat[fv.getIndex()] < 0)
				System.out.println("Bug");
		}
	}

	public double[] getLMStat(){
		return m_lmStat;
	}

	public double getOneLMStat(int index){
		return m_lmStat[index];
	}

	public double getLMSum(){
		double sum = 0;
		for(double v: m_lmStat)
			sum += v;
		return sum;
	}

	public void resetGamma(){
		setGamma(0);
	}
	public void setGamma(double g){
		m_gamma = g;
	}

	public double getGamma(){
		return m_gamma;
	}

	@Override
	public String showStat() {
		return String.format("%d(%.2f/%.3f)", m_memSize, m_pCount/(m_pCount+m_nCount), m_gamma);
	}

	public double getPosCount(){
		return m_pCount;
	}

	public double getNegCount(){
		return m_nCount;
	}

	public void resetReviewNames(){
		m_reviewNames.clear();
	}
	public void addReviewNames(String s){
		m_reviewNames.add(s);
	}

	public int getReviewSize(){
		return m_reviewNames.size();
	}

	public ArrayList<String> getReviewNames(){
		return m_reviewNames;
	}

//	@Override
//	// override the function to make the disabling and enabling by itself.
//	public void updateMemCount(int c){
//		m_memSize += c;
//	}

	// Functions used in MMB model.
	public void updateEdgeCount(int e, int c){
		m_edgeSize[e] += c;
	}

	// Get the edge count for i->j falls in the current theta.
	public int getEdgeSize(int e){
		return m_edgeSize[e];
	}

	public int getTotalEdgeSize(){
		return Utils.sumOfArray(m_edgeSize);
	}

	// Get the edge count for B_gh (i->j falls in theta_g, j->i falls in theta_h)
	// And 'e' indicates whether it is 0 edge or 1 edge.
	public int getConnectionEdgeCount(_HDPThetaStar theta, int e){
		if(m_connectionMap.containsKey(theta))
			return m_connectionMap.get(theta).getEdgeCount(e);
		else{
			//System.err.println("No such connections!");
			return 0;
		}
	}

	public void rmConnection(_HDPThetaStar theta, int e){
		if(m_connectionMap.containsKey(theta)){
			m_connectionMap.get(theta).updateEdgeCount(e, -1);
			if(m_connectionMap.get(theta).getEdgeCount(0) + m_connectionMap.get(theta).getEdgeCount(1) == 0)
				m_connectionMap.remove(theta);
		} else
			System.err.println("No such thetas!");
	}

	public void addConnection(_HDPThetaStar theta, int e){
		if(!m_connectionMap.containsKey(theta))
			m_connectionMap.put(theta, new _Connection(theta));

		m_connectionMap.get(theta).updateEdgeCount(e, 1);
	}

	public HashMap<_HDPThetaStar, _Connection> getConnectionMap(){
		return m_connectionMap;
	}

	public boolean hasConnection(_HDPThetaStar theta){
		return m_connectionMap.containsKey(theta);
	}

	public _Connection getConnection(_HDPThetaStar theta){
		if(m_connectionMap.containsKey(theta))
			return m_connectionMap.get(theta);
		else{
			System.out.println("[Bug]The connection does not exist!");
			return null;
		}
	}
	public Set<_HDPThetaStar> getConnectionKeys(){
		return m_connectionMap.keySet();
	}
	@Override
	public void reset(){
		super.reset();
		m_gamma = 0;
		m_isValid = false;
		Arrays.fill(m_edgeSize, 0);
		m_lmStat = null;
		m_connectionMap.clear();
		m_reviewNames.clear();
		m_hSize = 0;
	}
	// for debug purpose we print out the performance of each cluster
	double[][] m_tpTable = new double[2][2];

	public _PerformanceStat getPerfStat(){
		return m_perfStat;
	}

	public void setPerfStat(int classNo){
		m_perfStat = new _PerformanceStat(classNo);
	}

}