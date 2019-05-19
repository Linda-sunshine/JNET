package structures;

import utils.Utils;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The data structure for topic embedding
 * We can either infer the topic embedding or pre-train the topic embedding
 * The dimension of the topic embedding is k.
 */
public class _Topic4EUB {
    public int m_index;
    public double[] m_mu_phi;
    public double[][] m_sigma_phi;

    public _Topic4EUB(int index){
        m_index = index;
    }

    public int getIndex(){
        return m_index;
    }

    public void setTopics4Variational(int k, double mu, double sigma){
        m_mu_phi = new double[k];
        m_sigma_phi = new double[k][k];


        Utils.randomize(m_mu_phi, mu);
        for(int i=0; i<k; i++){
            Utils.randomize(m_sigma_phi[i], sigma);
            m_sigma_phi[i][i] += 1;//make it diagonal dominate
        }
    }
}