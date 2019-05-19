package structures;

import java.util.ArrayList;

import utils.Utils;

public class _Item {
	protected String m_itemID;
	protected ArrayList<_Review> m_reviews;
	protected _SparseFeature[] m_BoWProfile; //The BoW representation of a user.
	protected double[] m_itemWeights; // the learned eta from ETBIR
	
	public _Item(String id){
		m_itemID = id;
		m_reviews = new ArrayList<_Review>();
	}
	
	public void addOneReview(_Review r){
		m_reviews.add(r);
	}
	
	// build the profile for the user
	public void buildProfile(String model){
		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();

		if(model.equals("lm")){
			for(_Review r: m_reviews){
				reviews.add(r.getLMSparse());
			}
			m_BoWProfile = Utils.MergeSpVcts(reviews);
		} else{
			for(_Review r: m_reviews){
				reviews.add(r.getSparse());
			}
			m_BoWProfile = Utils.MergeSpVcts(reviews);	
		}
	}
	
	public void normalizeProfile(){
		double sum = 0;
		for(_SparseFeature fv: m_BoWProfile){
			sum += fv.getValue();
		}
		for(_SparseFeature fv: m_BoWProfile){
			double val = fv.getValue() / sum;
			fv.setValue(val);
		}
	}
	
	public _SparseFeature[] getBoWProfile(){
		return m_BoWProfile;
	}
	
	public double[] getItemWeights(){
		return m_itemWeights;
	}
	
	public void setItemWeights(double[] ws){
		m_itemWeights = ws;
	}
		
}
