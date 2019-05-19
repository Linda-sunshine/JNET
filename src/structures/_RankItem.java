package structures;


public class _RankItem implements Comparable<_RankItem> {
	public double m_value;
	public String m_name;
	public int m_index;
	public int m_label;
	
	public _RankItem(String name, double v) {
		m_value = v;
		m_name = name;
	}
	
	public _RankItem(int index, double v) {
		m_value = v;
		m_index = index;
	}
	
	public _RankItem(int index, double v, int label) {
		m_value = v;
		m_index = index;
		m_label = label;
	}

	@Override
	public int compareTo(_RankItem it) {
		if (this.m_value < it.m_value)
			return -1;
		else if (this.m_value > it.m_value)
			return 1;
		else
			return 0;
	}
	
	@Override
	public String toString() {
		if (m_name==null)
			return String.format("%d-%.4f-%d", m_index, m_value, m_label);
		else
			return String.format("%d-%s-%.4f-%d", m_index, m_name, m_value, m_label);
	}
}
