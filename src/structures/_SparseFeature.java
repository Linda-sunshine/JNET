/**
 * 
 */
package structures;

/**
 * @author lingong
 * Feature structure for sparse feature representation
 */
public class _SparseFeature implements Comparable<_SparseFeature> {
	String m_content; //Content of the feature.
	int m_index; // Index of the feature
	double m_TF; // raw TF value of this feature
	double m_value; // Value of the feature (non-zero)
	
	// feature value under different segments (E.g., in NewEgg we have Pro/Con/Comments segments)
	double[] m_values; //NOTE: special attention has to be made when computing feature values (except TF feature) in each section
	
	//Constructor.
	public _SparseFeature(){
		m_content = "";
		m_index = -1;
		m_value = 0;
		m_values = null;
	}
	
	//Constructor.
	public _SparseFeature(int index, String content) {
		m_content = content;
		m_index = index;
		m_value = 0;
		m_values = null;
	}
	
	public _SparseFeature(int index, double value){
		m_content = "";
		m_index = index;
		m_value = value;
		m_values = null;
	}
	
	public _SparseFeature(int index, double value, int dim){
		m_content = "";
		m_index = index;
		m_value = value;
		m_values = new double[dim];
	}
	
	//Get the content of the feature.
	public String getContent(){
		return m_content;
	}
	
	public String setContent(String content){
		m_content = content;
		return m_content;
	}
	
	//Get the index of the feature.
	public int getIndex(){
		return this.m_index;
	}
	
	//Set the index for the feature.
	public void setIndex(int index){
		this.m_index = index;
	}
	
	//Get the value of the feature.
	public double getValue(){
		return this.m_value;
	}
	
	//Set the value for the feature.
	public void setValue(double value){
		this.m_value = value;
	}	
	
	public void addValue(double value) {
		m_value += value;
	}
	
	
	public void setValue4Dim(double value, int d) {
		this.m_values[d] = value;//we will not check the index range@
	}
	
	public double[] getValues() {
		return this.m_values;
	}

	//added by Renqin to set the probit feature values
	public void setValues(double[] values){
		int valueDim = values.length;
		m_values = new double[valueDim];
		System.arraycopy(values, 0, m_values, 0, valueDim);
	}
	
	@Override
	public int compareTo(_SparseFeature sfv) {
		return m_index - sfv.m_index;
	}	
	
	// Added by Lin for storing raw TF for language models.
	public void setTF(double tf){
		m_TF = tf;
	}
	public double getTF(){
		return m_TF;
	}
}