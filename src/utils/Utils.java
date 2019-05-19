package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import json.JSONException;
import json.JSONObject;
import structures._Doc;
import structures._SparseFeature;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tfloat.FloatUniform;

public class Utils {
	
	public static double MAX_VALUE = 1e7;
	
	public static void shuffle(int[] order, int size){
		Random rand = new Random();
		int t, j;
		for(int i=size-1; i>0; i--){
			j = rand.nextInt(i+1);			
			
			t = order[i];
			order[i] = order[j];
			order[j] = t;
		}
	}
	
	public static double max(double[] w) {
		return max(w, 0, w.length);
	}
	
	public static double max(double[] w, int start, int size) {
		double max = w[start];
		int index = 0;
		for(int i=1; i<size; i++) {
			if (w[start+i] > max) {
				max = w[start+i];
				index = i;
			}
		}
		return w[start+index];
	}
	
	//Find the max value's index of an array, return Index of the maximum.
	public static int argmax(double[] p){
		return argmax(p, p.length);
	}
	
	public static int argmax(double[] p, int length){
		int maxIndex = 0;
		double maxValue = p[0];
		for(int i = 1; i < length; i++){
			if(p[i] > maxValue){
				maxValue = p[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int argmin(double[] p){
		return argmin(p, p.length);
	}
	
	public static int argmin(double[] p, int length){
		int minIndex = 0;
		double minValue = p[0];
		for(int i = 1; i < length; i++){
			if(p[i] < minValue){
				minValue = p[i];
				minIndex = i;
			}
		}
		return minIndex;
	}
	
	public static int count(String text, String pat) {
		int start = 0, count = 0;
		while ((start = text.indexOf(pat, start))!=-1) {
			count ++;
			start ++;
		}
		return count;
	}
	
	//Calculate the sum of a column in an array.
	public static double sumOfRow(double[][] mat, int rid){
		return sumOfArray(mat[rid]);
	}
	
	//Calculate the sum of a row in an array.
	public static double sumOfColumn(double[][] mat, int cid){
		double sum = 0;
		for(int j = 0; j < mat.length; j++)
			sum += mat[j][cid];
		return sum;
	}
	
	//Calculate the sum of a column in an array.
	public static int sumOfRow(int[][] mat, int rid){
		return sumOfArray(mat[rid]);
	}
	
	//Calculate the sum of a row in an array.
	public static int sumOfColumn(int[][] mat, int cid){
		int sum = 0;
		for(int j = 0; j < mat.length; j++){
			sum += mat[j][cid];
		}
		return sum;
	}
	
	//this requires the sparse feature vector has been sorted in ascending order
	public static int indexOf(_SparseFeature[] vct, int wid) {
		int start = 0, end = vct.length-1, mid=0;
		do {
			mid = (start+end)/2;
			if (vct[start].getIndex() > wid || vct[end].getIndex() < wid)
				return -1; //out of range
			else if (vct[mid].getIndex() > wid)
				end = mid-1;
			else if (vct[mid].getIndex() < wid)
				start = mid + 1;
			else
				return mid;	
		} while (end>=start);
		return -1;
	}
	
	//logScale: if prob is in log scale
	public static double entropy(double[] prob, boolean logScale) {
		double ent = 0;
		for(double p:prob) {
			if (logScale)
				ent += Math.exp(p) * p;
			else
				ent += Math.log(p) * p;
		}
		return -ent;
	}
	
	//This function is used to calculate the log of the sum of several values.
	public static double logSum(double[] xs){
		return logSum(xs, xs.length);
	}
	
	public static double logSum(double[] xs, int length){
		if(xs.length == 1){
			return xs[0];
		}
		
		double max = max(xs), sum = 0.0;
		for (int i = 0; i < xs.length; i++) {
			if (!Double.isInfinite(xs[i])) 
				sum += Math.exp(xs[i] - max);
		}
		
		if (sum==0)
			return max;
		return Math.log(sum) + max;
	}
	
	public static double logSum(double log_a, double log_b) {
		if (Double.isInfinite(log_a))
			return log_b;
		else if (Double.isInfinite(log_b))
			return log_a;
		else if (log_a < log_b)
			return log_b+Math.log(1 + Math.exp(log_a-log_b));
		else
			return log_a+Math.log(1 + Math.exp(log_b-log_a));
	}	
	
	//The function defines the dot product of beta and sparse Vector of a document.
	public static double dotProduct(double[] beta, _SparseFeature[] sf, int offset){
		double sum = beta[offset];
		for(int i = 0; i < sf.length; i++){
			int index = sf[i].getIndex() + offset + 1;
			sum += beta[index] * sf[i].getValue();
		}
		return sum;
	}
	
	public static double dotProduct(double[] a, double[] b) {
		if (a.length != b.length)
			return Double.NaN;
		double sum = 0;
		for(int i=0; i<a.length; i++)
			sum += a[i] * b[i];
		return sum;
	}

	public static double[] matrixMultVector(double[][] mtx, double[] vct){
		if(mtx[0].length != vct.length)
			return null;

		double[] res = new double[mtx.length];
		for(int i=0; i<mtx.length; i++)
			res[i] = Utils.dotProduct(mtx[i], vct);

		return res;
	}

	public static double L2Norm(double[] a) {
		return Math.sqrt(dotProduct(a,a));
	}
	
	//Logistic function: 1.0 / (1.0 + exp(-wf))
	//We assume w[0] is the bias term
	public static double logistic(double[] fv, double[] w){
		double sum = w[0];//start from bias term
		for(int i = 0; i < fv.length; i++)
			sum += fv[i] * w[1+i];
		return logistic(sum);
	}
	
	public static double logistic(double v) {
		return 1.0 / (1.0 + Math.exp(-v));
	}
	
	//The function defines the sum of an array.
	public static int sumOfArray(int[] a){
		int sum = 0;
		for (int i: a)
			sum += i;
		return sum;
	}
	
	//The function defines the sum of an array.
	public static double sumOfArray(double[] a) {
		double sum = 0;
		for (double i : a)
			sum += i;
		return sum;
	}
	
	//The function defines the sum of an array.
	public static double sumOfArray(_SparseFeature[] fv) {
		double sum = 0;
		for (_SparseFeature f : fv)
			sum += f.getValue();
		return sum;
	}
		
	//sum_i a[i] - b[i]
	public static double[] diff(double[] a, double[] b) {
		if (a.length != b.length)
			return null;
		
		double[] diff = new double[a.length];
		boolean nonzero = false;
		for(int i=0; i<a.length; i++) {
			diff[i] = a[i] - b[i];
			if (Math.abs(diff[i])>1e-10)
				nonzero = true;
		}
		return nonzero?diff:null;
	}
	
	public static void scaleArray(double[] a, double b) {
		for (int i=0; i<a.length; i++)
			a[i] *= b;
	}
	
	//set array a by array b
	public static void setArray(double[] a, double[] b, double scale) {
		if (a.length != b.length)
			return;
		
		for (int i=0; i<a.length; i++)
			a[i] = b[i] * scale;
	}
	
	static public void add2Array(double[] vct, double[] add, double weight){
		if (vct.length != add.length)
			return;
		for(int i=0; i<vct.length; i++)
			vct[i] += weight * add[i];
	}
	
	//L1 normalization: fsValue/sum(abs(fsValue))
	static public double sumOfFeaturesL1(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs)
			sum += Math.abs(feature.getValue());
		return sum;
	}
	
	//Set the normalized value back to the sparse feature.
	static public void L1Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeaturesL1(fs);
		if (sum>0) {
			//L1 length normalization
			for(_SparseFeature f:fs)
				f.setValue(f.getValue()/sum);
		}//otherwise, this is an empty feature vector
	}
	
	//L1 normalization
	static public void L1Normalization(double[] v) {
		double sum = sumOfArray(v);
		if (sum>0) {			
			for(int i=0; i<v.length; i++){
				v[i] /= sum;
			}
		}//what if the sum is negative?
	}
	
	//L2 normalization: fsValue/sqrt(sum of fsValue*fsValue)
	static public double sumOfFeaturesL2(_SparseFeature[] fs) {
		if(fs == null) 
			return 0;
		
		double sum = 0;
		for (_SparseFeature feature: fs){
			double value = feature.getValue();
			sum += value * value;
		}
		return Math.sqrt(sum);
	}
	
	static public void L2Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeaturesL2(fs);
		if (sum>0) {			
			for(_SparseFeature f: fs)
				f.setValue(f.getValue()/sum);
		}//otherwise, this is an empty feature vector
	}
	
	static public String getJSONValue(JSONObject json, String key) {
		try {
			if (json.has(key))				
				return(json.getString(key));
			else
				return "NULL";
		} catch (JSONException e) {
			return "NULL";
		}
	}
	
	//Calculate the similarity between two documents.
	public static double dotProduct(_Doc d1, _Doc d2){
		return dotProduct(d1.getSparse(), d2.getSparse());
	}
	
	public static double jaccard(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		if (spVct1==null || spVct2==null)
			return 0; // What is the minimal value of similarity?
		
		double overlap = 0;
		int p1 = 0, p2 = 0;
		while (p1 < spVct1.length && p2 < spVct2.length) {
			_SparseFeature t1 = spVct1[p1];
			_SparseFeature t2 = spVct2[p2];
			if (t1.getIndex() == t2.getIndex()) {
				overlap ++;
				p1++;
				p2++;
			} else if (t1.getIndex() > t2.getIndex())
				p2++;
			else
				p1++;
		}
		return overlap/(spVct1.length + spVct2.length - overlap);
	}
	
	public static double cosine(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		double spVct1L2 = sumOfFeaturesL2(spVct1), spVct2L2 = sumOfFeaturesL2(spVct2);
		if (spVct1L2==0 || spVct2L2==0)
			return 0;
		else
			return dotProduct(spVct1, spVct2) / spVct1L2 / spVct2L2;
	}
	
	public static double cosine(double[] a, double[] b) {
		if(L2Norm(a)==0 || L2Norm(b)==0)
			return 0;
		else
			return dotProduct(a, b) / L2Norm(a) / L2Norm(b);
	}
	
	//Calculate the similarity between two sparse vectors.
	public static double dotProduct(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		if (spVct1==null || spVct2==null)
			return 0; // What is the minimal value of similarity?
		
		double similarity = 0;
		int p1 = 0, p2 = 0;
		while (p1 < spVct1.length && p2 < spVct2.length) {
			_SparseFeature t1 = spVct1[p1];
			_SparseFeature t2 = spVct2[p2];
			if (t1.getIndex() == t2.getIndex()) {
				similarity += t1.getValue() * t2.getValue();
				p1++;
				p2++;
			} else if (t1.getIndex() > t2.getIndex())
				p2++;
			else
				p1++;
		}
		return similarity;
	}
	
	static public boolean isNumber(String token) {
		return token.matches("\\d+");
	}
	
	static public void randomize(double[] pros, double beta) {
        double total = 0;
        for (int i = 0; i < pros.length; i++) {
            pros[i] = beta + Math.random();//to avoid zero probability
            total += pros[i];
        }

        //normalize
        for (int i = 0; i < pros.length; i++)
            pros[i] /= total;
    }
	
	static public String formatArray(double [] array) {
		StringBuffer buffer = new StringBuffer(256);
		for(int i=0;i<array.length;i++)
			if (i==0)
				buffer.append(Double.toString(array[i]));
			else
				buffer.append("," + Double.toString(array[i]));
		return String.format("(%s)", buffer.toString());
	}
	
	static public _SparseFeature[] createSpVct(double[] denseFv) {
		ArrayList<_SparseFeature> spVct = new ArrayList<_SparseFeature>();
		for(int i=0; i<denseFv.length; i++) {
			if (Math.abs(denseFv[i]) > Double.MIN_VALUE)
				spVct.add(new _SparseFeature(i, denseFv[i]));
		}
		return spVct.toArray(new _SparseFeature[spVct.size()]);
	}
	
	static public _SparseFeature[] createSpVct(HashMap<Integer, Double> vct) {
		_SparseFeature[] spVct = new _SparseFeature[vct.size()];
		
		int i = 0;
		Iterator<Entry<Integer, Double>> it = vct.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry<Integer, Double> pairs = it.next();
			spVct[i] = new _SparseFeature(pairs.getKey(), pairs.getValue());
			i++;
		}
		Arrays.sort(spVct);		
		return spVct;
	}
	
	static public HashMap<Integer, Double> revertSpVct(_SparseFeature[] fvs){
		HashMap<Integer, Double> vct = new HashMap<Integer, Double>();
		for(_SparseFeature fv: fvs)
			vct.put(fv.getIndex(), fv.getValue());
		return vct;
	}
	
	static public _SparseFeature[] mergeSpVcts(ArrayList<_SparseFeature[]> vcts) {
		HashMap<Integer, Double> vct = new HashMap<Integer, Double>();
		
		for(_SparseFeature[] fv:vcts) {
			for(_SparseFeature f:fv) {
				int x = f.getIndex();
				if (vct.containsKey(x)) {
					vct.put(x, vct.get(x) + f.getValue());
				} else {
					vct.put(x, f.getValue());
				}
			}
		}
		return Utils.createSpVct(vct);
	}
	
	static public _SparseFeature[] createSpVct(ArrayList<HashMap<Integer, Double>> vcts) {
		HashMap<Integer, _SparseFeature> spVcts = new HashMap<Integer, _SparseFeature>();
		HashMap<Integer, Double> vPtr;
		_SparseFeature spV;
		
		int dim = vcts.size();
		for(int i=0; i<dim; i++) {
			vPtr = vcts.get(i);
			if (vPtr==null || vPtr.isEmpty())
				continue; // it is possible that we are missing this dimension
			
			//iterate through all the features in this section
			Iterator<Entry<Integer, Double>> it = vPtr.entrySet().iterator();
			while(it.hasNext()){
				Map.Entry<Integer, Double> pairs = it.next();
				int index = pairs.getKey();
				double value = pairs.getValue();
				if (spVcts.containsKey(index)) {
					spV = spVcts.get(index);
					spV.addValue(value); // increase the total value
				} else {
					spV = new _SparseFeature(index, value, dim);
					spVcts.put(index, spV);
				}
				spV.setValue4Dim(value, i);
			}
		}
		
		int size = spVcts.size();
		_SparseFeature[] resultVct = spVcts.values().toArray(new _SparseFeature[size]);
		
		Arrays.sort(resultVct);		
		return resultVct;
	}
	
	public static String cleanHTML(String content) {
		if (content.indexOf("<!--")==-1 || content.indexOf("-->")==-1)
			return content;//clean text
		
		int start = 0, end = content.indexOf("<!--");
		StringBuffer buffer = new StringBuffer(content.length());
		while(end!=-1) {
			if (end>start)
				buffer.append(content.substring(start, end).trim());
			start = content.indexOf("-->", end) + 3;
			end = content.indexOf("<!--", start);
		}
		
		if (start<content.length())
			buffer.append(content.substring(start));
		
		return cleanVideoReview(buffer.toString());
	}
	
	public static String cleanVideoReview(String content) {
		if (!content.contains("// <![CDATA[") || !content.contains("Length::"))
			return content;
		
		int start = content.indexOf("// <![CDATA["), end = content.indexOf("Length::", start);
		end = content.indexOf("Mins", end) + 4;
		StringBuffer buffer = new StringBuffer(content.length());
		buffer.append(content.substring(0, start));
		buffer.append(content.substring(end));
		
		if (buffer.length()==0)
			return null;
		else
			return buffer.toString();
	}
	
	public static void mergeVectors(HashMap<Integer, Double> src, HashMap<Integer, Double> dst) {
		Iterator<Entry<Integer, Double>> it = src.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<Integer, Double> pairs = it.next();
			int index = pairs.getKey();
			if (dst.containsKey(index)==false) 
				dst.put(index, pairs.getValue());
			else
				dst.put(index, pairs.getValue() + dst.get(index));
		}
	}
		
	public static boolean endWithPunct(String stn) {
		char lastChar = stn.charAt(stn.length()-1);
		return !((lastChar>='a' && lastChar<='z') 
				|| (lastChar>='A' && lastChar<='Z') 
				|| (lastChar>='0' && lastChar<='9'));
	}
	
	public static _SparseFeature[] negSpVct(_SparseFeature[] fv) {
		_SparseFeature[] result = new _SparseFeature[fv.length];
		for(int i=0; i<fv.length; i++)
			result[i] = new _SparseFeature(fv[i].getIndex(), -fv[i].getValue());
		return result;
	}
	
	//x_i - x_j 
	public static _SparseFeature[] diffVector(_SparseFeature[] spVcti, _SparseFeature[] spVctj){
		//first deal with special case
		if (spVcti==null && spVctj==null)
			return null;
		else if (spVctj==null)
			return spVcti;
		else if (spVcti==null)
			return negSpVct(spVctj);		
		
		ArrayList<_SparseFeature> vectorList = new ArrayList<_SparseFeature>();
		int i = 0, j = 0;
		_SparseFeature fi = spVcti[i], fj = spVctj[j];
		
		double fv;
		while (i < spVcti.length && j < spVctj.length) {
			fi = spVcti[i];
			fj = spVctj[j];
			
			if (fi.getIndex() == fj.getIndex()) {
				fv = fi.getValue() - fj.getValue();
				if (Math.abs(fv)>Double.MIN_VALUE)//otherwise it is too small
					vectorList.add(new _SparseFeature(fi.getIndex(),fv));
				i++; 
				j++; 
			} else if (fi.getIndex() > fj.getIndex()){
				vectorList.add(new _SparseFeature(fj.getIndex(), -fj.getValue()));
				j++;
			}
			else{
				vectorList.add(new _SparseFeature(fi.getIndex(), fi.getValue()));
				i++;
			}
		}
		
		while (i < spVcti.length) {
			fi = spVcti[i];
			vectorList.add(new _SparseFeature(fi.getIndex(), fi.getValue()));
			i++;
		}
		
		while (j < spVctj.length) {
			fj = spVctj[j];
			vectorList.add(new _SparseFeature(fj.getIndex(), -fj.getValue()));
			j++;
		}
		
		return vectorList.toArray(new _SparseFeature[vectorList.size()]);
	}
	
	static public Feature[] createLibLinearFV(_SparseFeature[] spVct, int fSize) {
		Feature[] node;
		if (fSize>0)//include bias term in the end
			node = new Feature[1+spVct.length]; 
		else//ignore bias term
			node = new Feature[spVct.length];
		
		int fid = 0;
		for(_SparseFeature fv:spVct)
			node[fid++] = new FeatureNode(1 + fv.getIndex(), fv.getValue());//svm's feature index starts from 1
		if (fSize>0)
			node[fid] = new FeatureNode(1+fSize, 1.0);
		return node;
	}
	
	static public Feature[] createLibLinearFV(_Doc doc, int fSize) {
		return Utils.createLibLinearFV(doc.getSparse(), fSize);
	}
	
	static public Feature[] createLibLinearFV(HashMap<Integer, Double> spVct) {
		Feature[] node = new Feature[spVct.size()]; 
		int fid = 0;
		for(_SparseFeature fv:createSpVct(spVct))
			node[fid++] = new FeatureNode(1 + fv.getIndex(), fv.getValue());//svm's feature index starts from 1
		return node;
	}
	
	//Get projectSpVct by building a map filter, added by Hongning.
	static public _SparseFeature[] projectSpVct(_SparseFeature[] fv, Map<Integer, Integer> filter) {
		ArrayList<_SparseFeature> pFv = new ArrayList<_SparseFeature>();
		for(_SparseFeature f:fv) {
			if (filter.containsKey(f.getIndex())) {
				pFv.add(new _SparseFeature(filter.get(f.getIndex()), f.getValue()));
			}
		}
		
		if (pFv.isEmpty())
			return null;
		else
			return pFv.toArray(new _SparseFeature[pFv.size()]);
	}
	
	//Get projectSpVct by building a hashmap<Integer, String> filter, added by Lin.
	static public _SparseFeature[] projectSpVct(_SparseFeature[] fv, HashMap<Integer, String> filter) {
		ArrayList<_SparseFeature> pFv = new ArrayList<_SparseFeature>();
		for(_SparseFeature f:fv) {
			if (filter.containsKey(f.getIndex())) {
				pFv.add(new _SparseFeature(f.getIndex(), f.getValue()));
			}
		}
		if (pFv.isEmpty())
			return null;
		else
			return pFv.toArray(new _SparseFeature[pFv.size()]);
	}
	
	//Dot product of the random vector and document sparse vector.
	public static double dotProduct(double[] vct, _SparseFeature[] sf){
		if(sf[sf.length-1].getIndex() > vct.length) {
			System.err.print("Error in computing dot product between a sparse vector and a full vector");
			return Double.NaN;
		}
		
		double value = 0;
		for(_SparseFeature fv:sf)
			value += vct[fv.getIndex()] * fv.getValue();
		return value;
	}

	//Sgn function: >= 0 1; < 0; 0.
	public static int sgn(double a){
		if (a >= 0) return 1;
		else return 0;
	}
	
	//Encode the hash value after getting the hash array.
	public static int encode(int[] hash){
		int value = 0;
		for(int i = 0; i < hash.length; i++){
			if (hash[i]>0)
				value += 1<<(hash.length-1-i);
		}
		return value;
	}
	
	/**
	   * Proc to calculate the value of the trigamma, the second
	   * derivative of the loggamma function. Accepts positive matrices.
	   * From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
	   * recurrence formula 6.4.6.  Each requires workspace at least 5
	   * times the size of X.
	   *
	   **/

	public static double trigamma(double x)
	{
	    double p;
	    int i;

	    x = x+6;
	    p = 1/(x*x);
	    p = (((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)*p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
	    for (i=0; i<6 ;i++) {
	        x = x-1;
	        p = 1/(x*x)+p;
	    }
	    return p;
	}


	/*
	 * taylor approximation of first derivative of the log gamma function
	 *
	 */

	public static double digamma(double x)
	{
	    double p;
	    x = x+6;
	    p = 1/(x*x);
	    p = (((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p;
	    p = p+Math.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	    return p;
	}

	public static double lgamma(double x)
	{
	    double x0,x2,xp,gl,gl0;
	    int n=0, k;
	    double a[] = {
	        8.333333333333333e-02,
	       -2.777777777777778e-03,
	        7.936507936507937e-04,
	       -5.952380952380952e-04,
	        8.417508417508418e-04,
	       -1.917526917526918e-03,
	        6.410256410256410e-03,
	       -2.955065359477124e-02,
	        1.796443723688307e-01,
	       -1.39243221690590
	    };
	    
	    x0 = x;
	    if (x <= 0.0) return 1e308;
	    else if ((x == 1.0) || (x == 2.0)) return 0.0;
	    else if (x <= 7.0) {
	        n = (int)(7-x);
	        x0 = x+n;
	    }
	    x2 = 1.0/(x0*x0);
	    xp = 2.0*Math.PI;
	    gl0 = a[9];
	    for (k=8;k>=0;k--) {
	        gl0 = gl0*x2 + a[k];
	    }
	    gl = gl0/x0+0.5*Math.log(xp)+(x0-0.5)*Math.log(x0)-x0;
	    if (x <= 7.0) {
	        for (k=1;k<=n;k++) {
	            gl -= Math.log(x0-1.0);
	            x0 -= 1.0;
	        }
	    }
	    return gl;
	}
	
	public static double KLsymmetric(double[] p, double [] q) {
		if (q.length != p.length)
			return Double.NaN;
		
		double sum = 0;
		for(int i=0; i<p.length; i++) {
			if(p[i] != 0 && q[i] != 0)
				sum += (p[i]-q[i]) * Math.log(p[i]/q[i]);
		}
		return sum;
	}
	
	public static double klDivergence(double[] p, double[] q) {
		if (q.length != p.length)
			return Double.NaN;
		
		double klDiv = 0.0;
		for (int i = 0; i < p.length; ++i) {
			if (p[i] == 0.0) { continue; }
			if (q[i] == 0.0) { continue; } 
			
			klDiv += p[i] * Math.log( p[i] / q[i] );
		}
		return klDiv; 
	}
	
	public static double RBF(_SparseFeature[] spVct1, _SparseFeature[] spVct2, double delta){
		if (spVct1.length==0 || spVct2.length==0)
			return 0;
		else{
			double similarity = 0;
			int p1 = 0, p2 = 0;
			while (p1 < spVct1.length && p2 < spVct2.length) {
				_SparseFeature t1 = spVct1[p1];
				_SparseFeature t2 = spVct2[p2];
				if (t1.getIndex() == t2.getIndex()) {
					similarity += (t1.getValue() - t2.getValue()) * (t1.getValue() - t2.getValue());
					p1++;
					p2++;
				} else if (t1.getIndex() > t2.getIndex())
					p2++;
				else
					p1++;
			}
			return Math.exp(-similarity/delta);
		}
	}
	
	public static double calculateMDistance(_Doc d1, _Doc d2, double[][] A){
		double distance = 0, tmp = 0;
		double[] t1 = d1.getTopics(), t2 = d2.getTopics();
		double[] tmpArray = new double[A.length]; 
		for(int i = 0; i < tmpArray.length; i++){
			for(int j = 0; j < t1.length; j++)
				tmp += (t1[j] - t2[j]) * A[j][i];
			tmpArray[i] = tmp;
			tmp = 0;
		}//(x_i - x_j)^T * A
		
		for(int i = 0; i < tmpArray.length; i++){
			distance += tmpArray[i] * (t1[i] - t2[i]);
		} //(x_i - x_j)^T * A * (x_i - x_j)
		return distance;
	}
	
	//The Euclidean distance for two arrays of the same length.
	public static double euclideanDistance(double[] t1, double[] t2){
		if(t1.length == t2.length){
			double sum = 0;
			for(int i=0; i < t1.length; i++){
				sum += (t1[i] - t2[i]) * (t1[i] - t2[i]);
			}	
			return sum;
		} else 
			return Double.NaN;
		
	}
	
	// added by Lin for computing LCS.
	public static int LCSLength(String[] x, String[] y) {
		int m = x.length;
		int n = y.length;
		int[][] c = new int[m + 1][n + 1];
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (x[i - 1].equals(y[j - 1]))
					c[i][j] = c[i - 1][j - 1] + 1;
				else if (c[i - 1][j] >= c[i][j - 1]) {
					c[i][j] = c[i - 1][j];
				} else
					c[i][j] = c[i][j - 1];
			}
		}
		return c[m][n];// Every cell contains the current longest common
						// sequence of xi, yj.
	}
	
	// The way we used to calcuate the index in a mXm symmetric matrix.
	public static int getIndex(int i, int j){
		//Swap i and j.
		if(i < j){
			int t = j;
			j = i;
			i = t;
		}
		return i*(i-1)/2+j;
	}

	/**
	 * 
	 * @param N: total document size
	 * @param DF: document frequency for term t
	 * @param tcDF: number of documents where t and c co-occur
	 * @param cDF: number of documents where t occurs
	 * @return
	 */
	static public double ChiSquare(int N, int DF, int tcDF, int cDF) {
		double A = tcDF;//t & c
		double B = DF - A;//t & !c
		double C = cDF - A;//!t & c
		double D = N - DF - cDF + A;//!t & !c
		
		return N * ( A * D - B * C ) * ( A * D - B * C ) / cDF / ( B + D ) / DF / ( C + D );
	}
	
//	//Sample with a given log array.
//	public static int sampleInLogArray(double[] logP, int length){
//		double sum = logP[0], rnd = Math.log(FloatUniform.staticNextFloat());
//		int i = 0;
//		while(sum<rnd && i<length){
//			i++;
//			sum = logSum(sum, logP[i]);
//		}
//		return i;
//	}
	
	public static int sampleInLogArray(double[] logP, int length){
		double sum = Utils.logSum(logP, length), rnd = FloatUniform.staticNextFloat();
		int i = -1;
		while(rnd>0 && i<length){
			i++;
			rnd -= Math.exp(logP[i]-sum);
		}
		return i;
	}
	
	public static double sampleFromGamma(double a, double b) {
		double v = 0;
		
		do {
			v = Gamma.staticNextDouble(a, b);
		} while (Math.abs(v)==0);
		
		return v;
	}
	
	
	public static int countOccurrencesOf(String text, String pat) {
		int start = 0, count = 0;
		while ((start = text.indexOf(pat, start))!=-1) {
			count ++;
			start ++;
		}
		return count;
	}
	
	//This function is used to calculate the log of the sum of several values.
	public static double logSumOfExponentials(double[] xs){
		if(xs.length == 1){
			return xs[0];
		}
		
		double max = max(xs, 0, xs.length), sum = 0.0;
		for (int i = 0; i < xs.length; i++) {
			if (!Double.isInfinite(xs[i])) 
				sum += Math.exp(xs[i] - max);
		}
		
		if (sum==0)
			return max;
		return Math.log(sum) + max;
	}
	
	static public _SparseFeature[] MergeSpVcts(ArrayList<_SparseFeature[]> vcts) {
		HashMap<Integer, Double> vct = new HashMap<Integer, Double>();
		
		for(_SparseFeature[] fv:vcts) {
			for(_SparseFeature f:fv) {
				int x = f.getIndex();
				if (vct.containsKey(x)) {
					vct.put(x, vct.get(x) + f.getValue());
				} else {
					vct.put(x, f.getValue());
				}
			}
		}
		return Utils.createSpVct(vct);
	}
	
	public static int maxOfArrayIndex(double[] probs, int length){
		int maxIndex = 0;
		double maxValue = probs[0];
		for(int i = 1; i < length; i++){
			if(probs[i] > maxValue){
				maxValue = probs[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	//scale array a by array b
	public static void scaleArray(double[] a, double[] b, double scale) {
		for (int i=0; i<a.length; i++)
			a[i] += b[i] * scale;
	}

	static public double sumOfFeaturesL2(double[] fs){
		if(fs == null)
			return 0;
		double sum = 0;
		for(double f: fs)
			sum += f * f;
		return Math.sqrt(sum);
	}

	public static void normalize(double[] a){
		double sum = 0;
		for(int i=0; i<a.length; i++){
			if(a[i] < 0)
				a[i] = -a[i];
			sum += a[i];
		}
		for(int i=0; i<a.length; i++) {
			a[i] /= sum;
		}
	}
}