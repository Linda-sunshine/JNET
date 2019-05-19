package Classifier.supervised.modelAdaptation.HDP;

import java.util.Arrays;

import cern.jet.random.tdouble.Gamma;
import structures._SparseFeature;
/**
 * Dirichlet distribution, implemented by gamma function.
 * Referal: https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
 * @author lin
 */
public class DirichletPrior {
	
	//Sampling from parameters [\alpha_1, \alpha_2,...,\alpha_k].
	public void sampling(double[] target, double[] alphas, boolean toLog){
		double sum = 0;
		for(int i=0; i<alphas.length; i++){
			while(target[i] == 0)
				target[i] = Gamma.staticNextDouble(alphas[i], 1);
			sum += target[i];
		}
		
		for(int i=0; i<alphas.length; i++) {
				target[i]/=sum;
			if (toLog)
				target[i] = Math.log(target[i]);
		}
	}
	
	//Sampling from posterior [\alpha_1+n_1, \alpha_2+n_2,...,\alpha_k+n_k].
	public void sampling(double[] target, double[] alphas, _SparseFeature[] fvs, boolean toLog){
		if(fvs.length == 0){
			sampling(target, alphas, toLog);
			return;
		}
		
		double sum = 0;
		int count = 0;
		for(int i=0; i<alphas.length; i++){
			if(i == fvs[count].getIndex()){
				while(target[i] == 0)
					target[i] = Gamma.staticNextDouble(alphas[i]+fvs[count].getValue(), 1);
				sum += target[i];
				if(count != fvs.length-1 )
					count++;
			} else{
				while(target[i] == 0)
					target[i] = Gamma.staticNextDouble(alphas[i], 1);
				sum += target[i];
			}
		}
		
		for(int i=0; i<alphas.length; i++) {
				target[i]/=sum;
			if (toLog)
				target[i] = Math.log(target[i]);
		}
	}
	
	//Sampling given dim of target vector.
	public void sampling(double[] target, int dim, double alpha, boolean toLog){
		if (dim>target.length) {
			System.err.println("[Error]The length of target vector is shorter than the specified size!");
			return ;
		}
		
		double[] alphas = new double[dim];
		Arrays.fill(alphas, alpha);
		sampling(target, alphas, toLog);
	}
}
