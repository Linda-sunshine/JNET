package Classifier.supervised.libsvm;
public class svm_node implements java.io.Serializable, Comparable<svm_node>
{
	private static final long serialVersionUID = -3046511301730620312L;
	public int index;
	public double value;
	
	@Override
	public int compareTo(svm_node node) {
		return index - node.index;
	}
}
