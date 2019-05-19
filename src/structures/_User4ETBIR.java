package structures;

import utils.Utils;

/**
 * Created by lulin on 3/29/18.
 */
public class _User4ETBIR extends _User{
    public double[][] m_nuP;
    public double[][][] m_SigmaP;

    public _User4ETBIR(String userID){
    	super(userID);
    }
    
    public void setTopics4Variational(int k, double nu, double sigma) {
    	m_nuP = new double[k][k];
        m_SigmaP = new double[k][k][k];
        for(int i = 0; i < k; i++){
            Utils.randomize(m_nuP[i], nu);
            for(int j=0; j<k; j++)
            	m_SigmaP[i][j][j] = sigma;
        }
    }
}
