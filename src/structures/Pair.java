package structures;

//Structure: pair for storing real rank and ideal rank.
//It is used in nDCG calculation.

public class Pair {

	double m_label;
	double m_rankValue;
		
	public Pair(){
		m_label = 0;
		m_rankValue = 0;
	}
		
	public Pair(double l, double rv){
		m_label = l;
		m_rankValue = rv;
	}
		
	public double getLabel(){
		return m_label;
	}
	
	public double getValue(){
		return m_rankValue;
	}
	
	//rank by predicted score
	public int compareTo (Pair p){
		if (this.m_rankValue > p.m_rankValue)
			return -1;
		else if (this.m_rankValue < p.m_rankValue)
			return 1;
		else 
			return 0;
	}
}