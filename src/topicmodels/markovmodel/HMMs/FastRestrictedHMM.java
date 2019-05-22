package topicmodels.markovmodel.HMMs;

import java.util.Arrays;

import structures._Doc;
import utils.Utils;

/**
 * 
 * @author Hongning Wang
 * Implementation of restricted hidden Markov model, where transition probability is specified by a binomial distribution (same state or a new state)
 * Specifications can be found in 
 * Gruber, Amit, Yair Weiss, and Michal Rosen-Zvi. "Hidden Topic Markov Models." AISTATS. Vol. 7. 2007.
 * 
 */
public class FastRestrictedHMM {

	int number_of_topic;
	int length_of_seq;
	double alpha[][];
	double beta[][];//when using Viterbi, this stores backtracking pointers (with int value)
	double norm_factor[];//normalize the alpha beta matrices per column
	double m_epsilon;//single epsilon shared by all the sentences (transition probability)
	int constant;//in HTSM, transition is between topic and sentiment
	
	_Doc m_docPtr;//pointer to the input document
	
	public FastRestrictedHMM(double epsilon, int maxSeqSize, int topicSize, int constant) {
		m_epsilon = epsilon;//in real space!
		this.constant = constant;
		this.number_of_topic = topicSize;
		alpha  = new double[maxSeqSize][this.constant*this.number_of_topic];
		beta = new double[maxSeqSize][this.constant*this.number_of_topic];
		norm_factor = new double[maxSeqSize];
	}
	
	//NOTE: in real space!!!!
	double getEpsilon(int t) {
		return m_epsilon;
	}
	
	public void setEpsilon(double epsilon) {
		m_epsilon = epsilon;
	}
		
	public double ForwardBackward(_Doc d, double[][] emission) {
		m_docPtr = d;
		this.length_of_seq = d.getSenetenceSize();
		
		double loglik = initAlpha(d.m_topics, emission[0]) + forwardComputation(emission, d.m_topics);
		backwardComputation(emission, d.m_topics);		
		
		return loglik;
	}
	
	//NOTE: all computation in log space
	double initAlpha(double[] theta, double[] local0) {
		double norm = Double.NEGATIVE_INFINITY;//log0
		
		for (int i = 0; i < this.number_of_topic; i++) {
			alpha[0][i] = local0[i] + theta[i];//document must start with a new topic and a new sentiment
			norm = Utils.logSum(norm, alpha[0][i]);
			//the rest set to zero
			for(int j=1; j<this.constant; j++)
				alpha[0][i+j*this.number_of_topic] = Double.NEGATIVE_INFINITY;
		}
		
		//normalization
		for (int i = 0; i < this.number_of_topic; i++)
			alpha[0][i] -= norm;
		
		norm_factor[0] = norm;
		return norm;
	}
	
	double forwardComputation(double[][] emission, double[] theta) {
		double logLikelihood = 0, norm, logEpsilon, logOneMinusEpsilon;
		for (int t = 1; t < this.length_of_seq; t++) {
			norm = Double.NEGATIVE_INFINITY;//log0
			logEpsilon = Math.log(getEpsilon(t));
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t));
			
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] = logEpsilon + theta[i] + emission[t][i];  // regardless of the previous
				alpha[t][i+this.number_of_topic] = logOneMinusEpsilon + Utils.logSum(alpha[t-1][i], alpha[t-1][i+this.number_of_topic]) + emission[t][i];
				
				norm = Utils.logSum(norm, Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]));
			}
			
			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] -= norm;
				alpha[t][i+this.number_of_topic] -= norm;
			}
			
			logLikelihood += norm; 
			norm_factor[t] = norm;
		}
		return logLikelihood;
	}
	
	void backwardComputation(double[][] emission, double[] theta) {
		//initiate beta_n
		Arrays.fill(beta[this.length_of_seq-1], 0); // since log 1 = 0
		
		double sum, logEpsilon, logOneMinusEpsilon;
		for(int t=this.length_of_seq-2; t>=0; t--) {
			logEpsilon = Math.log(getEpsilon(t+1));
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t+1));
			
			sum = logEpsilon;
			for (int j = 0; j < this.number_of_topic; j++)
				sum = Utils.logSum(sum, theta[j] + emission[t+1][j] + beta[t+1][j]);
			
			for (int i = 0; i < this.number_of_topic; i++) {
				beta[t][i] = Utils.logSum(logOneMinusEpsilon + beta[t+1][i] + emission[t+1][i], sum) - norm_factor[t];
				beta[t][i + this.number_of_topic] = beta[t][i];
			}
		}
	}
	
	public void collectExpectations(double[][] sstat) {
		for(int t=0; t<this.length_of_seq; t++) {
			double norm = Double.NEGATIVE_INFINITY;//log0
			for(int i=0; i<this.constant*this.number_of_topic; i++) 
				norm = Utils.logSum(norm, alpha[t][i] + beta[t][i]);
			
			for(int i=0; i<this.constant*this.number_of_topic; i++) 
				sstat[t][i] = Math.exp(alpha[t][i] + beta[t][i] - norm); // convert into original space
		}
	}
	
	//-----------------Viterbi Algorithm--------------------//
	//NOTE: all computation in log space
	public void computeViterbiAlphas(double[][] emission, double[] theta) {
		double logEpsilon, logOneMinusEpsilon;
		
		for (int t = 1; t < this.length_of_seq; t++) {
			logEpsilon = Math.log(getEpsilon(t));
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t));
			
			int prev_best = FindBestInLevel(t-1);
			for (int i = 0; i < this.number_of_topic; i++) {
				//\psi=1: random sample a topic
				alpha[t][i] = alpha[t-1][prev_best] + theta[i] + emission[t][i] + logEpsilon;
				beta[t][i] = prev_best;
				
				//\psi=0: keep previous topic
				if(alpha[t-1][i] > alpha[t-1][i+this.number_of_topic]) {
					alpha[t][i+this.number_of_topic] = alpha[t-1][i] + logOneMinusEpsilon + emission[t][i];
					beta[t][i+this.number_of_topic] = i;
				} else {
					alpha[t][i+this.number_of_topic] = alpha[t-1][i+this.number_of_topic] + logOneMinusEpsilon + emission[t][i];
					beta[t][i+this.number_of_topic] = i + this.number_of_topic;
				}
			}// End for i
		}//End For t
	}

	int FindBestInLevel(int t) {
		double best = alpha[t][0];
		int best_index = 0;
		for(int i = 1; i<this.constant*this.number_of_topic; i++){
			if(alpha[t][i] > best){
				best = alpha[t][i];
				best_index = i;
			}
		}
		return best_index;
	}
	
	public void BackTrackBestPath(_Doc d, double[][] emission, int[] path) {
		this.length_of_seq = d.getSenetenceSize();
		
		initAlpha(d.m_topics, emission[0]);
		computeViterbiAlphas(emission, d.m_topics);
		
		int level = this.length_of_seq - 1;
		path[level] = FindBestInLevel(level);
		for(int i = this.length_of_seq - 2; i>=0; i--)
			path[i] = (int)beta[i+1][path[i+1]];  
	}	
}
