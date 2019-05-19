/**
 * 
 */
package structures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import utils.Utils;

/**
 * @author Hongning Wang
 * Abstract class for document structure, should only contain the most essential component of a text document
 */
public abstract class _DocBase {
	String m_name; // document ID string	
	int m_ID; // unique id of the document in the collection
	
	String m_source; //The content of the source file.
	int m_totalLength; //The total length of the document in tokens
	
	int m_y_label; // classification target, that is the index of the labels.
	int m_predict_label; //The predicted result.
	long m_timeStamp; //The timeStamp for this review.
	
	//importance of document for statistic model learning
	double m_weight = 1.0; // instance weight for supervised model training (will be reset by PageRank)
	protected _SparseFeature[] m_x_sparse; // sparse representation of features: default value will be zero.
	
	public void setWeight(double w) {
		m_weight = w;
	}
	
	public double getWeight() {
		return m_weight;
	}
	
	//Get the ID of the document.
	public int getID(){
		return m_ID;
	}
	
	//Set a new ID for the document.
	public void setID(int id){
		m_ID = id;
	}
	
	public void setName(String name) {
		m_name = name;
	}
	
	public String getName() {
		return m_name;
	}
	
	//Get the source content of a document.
	public String getSource(){
		return this.m_source;
	}
	
	public void clearSource() {
		m_source = null;
	}
	
	//Get the real label of the doc.
	public int getYLabel() {
		return this.m_y_label;
	}
	
	//Set the Y value for the document, Y represents the class.
	public void setYLabel(int label){
		this.m_y_label = label;
	}
	
	//Get the time stamp of the document.
	public long getTimeStamp(){
		return this.m_timeStamp;
	}
	
	//Get the predicted result, which is used for comparison.
	public int getPredictLabel() {
		return this.m_predict_label;
	}
	
	//Get the predicted result, which is used for comparison.
	public int getPredictLabelG() {
		return this.m_predict_label_g;
	}	
	
	//Set the predict result back to the doc.
	public int setPredictLabel(int label){
		this.m_predict_label = label;
		return this.m_predict_label;
	}

	int m_predict_label_g = 0;
	//Set the predict result back to the doc.
	public int setPredictLabelG(int label){
		this.m_predict_label_g = label;
		return this.m_predict_label_g;
	}
	//Set the time stamp for the document.
	public void setTimeStamp(long t){
		this.m_timeStamp = t;
	}
	
	//Get the sparse vector of the document.
	public _SparseFeature[] getSparse(){
		return this.m_x_sparse;
	}

//	//get the sparse feature indices for this document
//	public int[] getIndices() {
//		int[] indices = new int[m_x_sparse.length];
//		for(int i=0; i<m_x_sparse.length; i++) 
//			indices[i] = m_x_sparse[i].m_index;
//		
//		return indices;
//	}
//	
//	//get the sparse feature values for this document
//	public double[] getValues() {
//		double[] values = new double[m_x_sparse.length];
//		for(int i=0; i<m_x_sparse.length; i++) 
//			values[i] = m_x_sparse[i].m_value;
//		
//		return values;
//	}
	
	//return the unique number of features in the doc
	public int getDocLength() {
		return m_x_sparse.length;
	}	
	
	//Get the total number of tokens in a document.
	public int getTotalDocLength(){
		return m_totalLength;
	}
	
	void calcTotalLength() {
		m_totalLength = 0;
		for(_SparseFeature fv:m_x_sparse)
			m_totalLength += fv.getValue();
	}
	
	//Create the sparse vector for the document, taking value from different sections
	public void createSpVct(ArrayList<HashMap<Integer, Double>> spVcts) {
		m_x_sparse = Utils.createSpVct(spVcts);
		calcTotalLength();
	}
	
	//Create the sparse vector for the document.
	public void createSpVct(HashMap<Integer, Double> spVct) {
		m_x_sparse = Utils.createSpVct(spVct);
		calcTotalLength();
	}
	
	// Added by Lin for language model.
	_SparseFeature[] m_lm_x_sparse;
	public void createLMSpVct(HashMap<Integer, Double> spVct){
		m_lm_x_sparse = Utils.createSpVct(spVct);
	}
	
	public void setSpVct(_SparseFeature[] x) {
		m_x_sparse = x;
		calcTotalLength();
	}
	
	public _SparseFeature[] getLMSparse(){
		if (m_lm_x_sparse!=null)
			return m_lm_x_sparse;
		else
			return m_x_sparse;//this will make all old implementation consistent 
	}
	
	//added by Lin for sanity check purpose.
	int[] m_indices;
	double[] m_values;
	public void filterIndicesValues(HashSet<Integer> indices){
		ArrayList<Integer> inds = new ArrayList<Integer>();
		ArrayList<Double> vals = new ArrayList<Double>();
		for(_SparseFeature sf: m_x_sparse){
			if(indices.contains(sf.getIndex())){
				inds.add(sf.getIndex());
				vals.add(sf.getValue());
			}
		}
		m_indices = new int[inds.size()];
		m_values = new double[vals.size()];
		for(int i=0; i<inds.size(); i++){
			m_indices[i] = inds.get(i);
			m_values[i] = vals.get(i);
		} 		
	}
	public int[] getIndices(){
		return m_indices;
	}
	public double[] getValues(){
		return m_values;
	}
}
