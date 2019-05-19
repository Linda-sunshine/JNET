package structures;

import utils.Utils;

/**
 * Created by lulin on 3/29/18.
 */
public class _Product4ETBIR extends _Product {
    public double[] m_eta; // variational inference for p(\gamma|\eta)
   
    public _Product4ETBIR(String ID){
    	super(ID);
    }
    
    //create necessary structure for variational inference    
  	public void setTopics4Variational(int k, double eta) {
  		m_eta = new double[k];
  		Utils.randomize(m_eta, eta);
  	}
}
