/**
 * 
 */
package LBFGS;

/**
 * @author hongning
 * This is the interface for any differentiable function 
 */
public interface Optimizable {
	//return the current parameters, i.e., X
	public double[] getParameters();
	
	//set the current parameters to x
	public void setParameters(double[] x);
	
	//compute the function value and gradient at current point
	public double calcFuncGradient(double[] g);
	
	public void calcDiagnoal(double[] x);
	
	//compute the function value at the given point
	public double calcFunc(double[] x);
	
	//perform projection of the function's parameters, if necessary
	public void projection(double[] x);
	
	public int getNumParameters();
	
	//return the total number of function and gradient evaluation, just for debugging purpose
	public int getNumEval();
}
