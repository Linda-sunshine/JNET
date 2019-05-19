package structures;

public class _WeightedCount {
	double m_weight;
	int m_count;
	
	public _WeightedCount(double w, int c){
		m_weight = w;
		m_count = c;
	}
	
	protected int getCount(){
		return m_count;
	}
	
	protected double getWeight(){
		return m_weight;
	}
	
	public double getWeightedCount(){
		return m_weight * m_count;
	}
}
