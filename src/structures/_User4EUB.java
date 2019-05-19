package structures;

import java.util.ArrayList;
import topicmodels.UserEmbedding.EUB;
import utils.Utils;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * Each user has a set of variational parameters:
 * variational parameters (\mu, \simga) for user embedding u
 * variational parameters (\mu, \sigma) for affinity \delta
 * variational parameters for taylor parameter \epsilon
 */
public class _User4EUB extends _User {
    public double[] m_mu_u;
    public double[][] m_sigma_u;

    public double[] m_epsilon;
    // the introduction of sparsity needs one more taylor parameter
    public double[] m_epsilon_prime;

    // the variational paramters for the affinity with other users
    public double[] m_mu_delta;
    public double[] m_sigma_delta;

    public _User4EUB(_User u){
        super(u.getUserID());
        if(u.getFriends() != null)
            setFriends(u.getFriends());
    }

    public void setReviews(ArrayList<_Doc4EUB> docs){
        ArrayList<_Review> reviews = new ArrayList<>();
        for(_Doc4EUB doc: docs)
            reviews.add(doc);
        m_reviews = reviews;
    }

    public void setTopics4Variational(int dim, int userSize, double mu, double sigma){
        m_mu_u = new double[dim];
        m_sigma_u = new double[dim][dim];

        m_mu_delta = new double[userSize];
        m_sigma_delta = new double[userSize];
        m_epsilon = new double[userSize];
        m_epsilon_prime = new double[userSize];

        // init mu_u/sigma_u
        for(int m=0; m<dim; m++){
            m_mu_u[m] = mu + Math.random();
            for(int l=0; l<dim; l++){
                m_sigma_u[m][l] = sigma + Math.random();
            }
            m_sigma_u[m][m] += 1;
        }

        for(int j=0; j<userSize; j++){
            m_mu_delta[j] = mu + Math.random();
            m_sigma_delta[j] = sigma + Math.random();
        }

//        Utils.normalize(m_mu_u);
//        Utils.normalize(m_mu_delta);
//        Utils.normalize(m_sigma_delta);

        // thus we scale the two set of parameters by 100
//        Utils.scaleArray(m_mu_delta, 10);
//        Utils.scaleArray(m_sigma_delta, 10);

        for(int i=0; i<userSize; i++){
            m_epsilon[i] = Math.exp(m_mu_delta[i] + 0.5 * m_sigma_delta[i] * m_sigma_delta[i]) + 1;
            m_epsilon_prime[i] = (1-EUB.m_rho) * Math.exp(m_mu_delta[i] + 0.5 * m_sigma_delta[i] * m_sigma_delta[i]) + 1;
        }
    }
}