package Classifier.semisupervised;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import structures._Corpus;
import structures._Doc;
import structures._Edge;
import structures._Node;
import utils.Utils;
import Classifier.BaseClassifier;
import Classifier.semisupervised.PairwiseSimCalculator.ActionType;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class GaussianFields extends BaseClassifier {
	
	double m_alpha; //Weight coefficient for labeled neighbors.
	double m_beta; //Weight coefficient for unlabeled neighbors
	double m_M; //Influence of labeled node (similar effect of eta)
	int m_k; // k labeled neighbors.
	int m_kPrime;//k' unlabeled neighbors.
	
	protected int m_U;
	protected int m_L;
	protected _Node[] m_nodeList; // list of nodes with its nearest neighbors in the graph
	SparseDoubleMatrix2D m_graph;
	
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	
	protected BaseClassifier m_classifier; //Multiple learner.
	double[] m_pY;//p(Y), the probabilities of different classes.
	double[] m_pYSum; //\sum_i exp(-|c-fu(i)|)
	
	Thread[] m_threadpool;
	
	public GaussianFields(_Corpus c, String classifier, double C){
		super(c);
		
		m_labelRatio = 0.2;//an arbitrary setting
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = 100;
		m_kPrime = 50;	
		m_labeled = new ArrayList<_Doc>();
		
		int classNumber = c.getClassSize();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];
		
		m_nodeList = null;
		setClassifier(classifier, C);
	}	
	
	public GaussianFields(_Corpus c, String classifier, double C, double ratio, int k, int kPrime){
		super(c);
		
		m_labelRatio = ratio;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		
		int classNumber = c.getClassSize();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];
		
		m_nodeList = null;
		setClassifier(classifier, C);
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields with matrix inversion [C:%s, kUL:%d, kUU:%d, r:%.3f, alpha:%.3f, beta:%.3f]", 
				m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta);
	}
	
	private void setClassifier(String classifier, double C) {
		if (classifier.equals("NB"))
			m_classifier = new NaiveBayes(m_classNo, m_featureSize);
		else if (classifier.equals("LR"))
			m_classifier = new LogisticRegression(m_classNo, m_featureSize, C);
		else if (classifier.equals("SVM"))
			m_classifier = new SVM(m_classNo, m_featureSize, C);
		else {
			System.out.println("Classifier has not developed yet!");
			System.exit(-1);
		}
	}
	
	@Override
	protected void init() {
		m_labeled.clear();
		Arrays.fill(m_pY, 0);
		Arrays.fill(m_pYSum, 0);
	}
	
	//Train the data set.
	@Override
	public double train(Collection<_Doc> trainSet){
		init();
		
		//using all labeled data for classifier training
		m_classifier.train(trainSet);
		
		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc:trainSet){
			m_pY[doc.getYLabel()]++;
			if(r.nextDouble()<m_labelRatio)
				m_labeled.add(doc);
		}
		
		//estimate the prior of p(y=c)
		Utils.scaleArray(m_pY, 1.0/Utils.sumOfArray(m_pY));
		
		//set up labeled and unlabeled instance size
		m_U = m_testSet.size();
		m_L = m_labeled.size();
		
		return 0; // we can compute the corresponding objective function value
	}
	
	public _Doc getTestDoc(int i) {
		return m_testSet.get(i);
	}
	
	public _Doc getLabeledDoc(int i) {
		return m_labeled.get(i);
	}
	
	protected double getBoWSim(_Doc di, _Doc dj) {
		return Utils.dotProduct(di, dj);
	}
	
	protected double getTopicalSim(_Doc di, _Doc dj) {
		if (di.m_topics == null || dj.m_topics == null)
			return 0;
		int topicSize = di.m_topics.length;
		return Utils.KLsymmetric(di.m_topics, dj.m_topics)/topicSize;
	}
	
	protected double getPOSScore(_Doc di, _Doc dj){
		return Utils.cosine(di.getPOSVct(), dj.getPOSVct());
	}

	protected double getAspectScore(_Doc di, _Doc dj){
		return Utils.cosine(di.getAspVct(), dj.getAspVct());
	}
	
	public double getSimilarity(_Doc di, _Doc dj) {
//		return Math.random();//just for debugging purpose
		return Math.exp(getBoWSim(di, dj) - getTopicalSim(di, dj));
	}
	
	protected void WaitUntilFinish(ActionType atype) {
		int cores = Runtime.getRuntime().availableProcessors();
		int start = 0, end, load = m_U/cores;
		
		for(int i=0; i<cores; i++) {
			if (i==cores-1)
				end = m_U;
			else
				end = start + load;
			
			m_threadpool[i] = new Thread(new PairwiseSimCalculator(this, start, end, atype));
			start = end;
			
			m_threadpool[i].start();
		}
		
		for(int i=0; i<m_threadpool.length; i++){
			try {
				m_threadpool[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	protected void calcSimilarityInThreads(){
		//create the node list for constructing the nearest neighbor graph
		if (m_nodeList==null || m_nodeList.length < m_U+m_L)
			m_nodeList = new _Node[(int)((m_U+m_L)*1.2)];//create sufficient space 			

		//fill in the labeled parts
		for(int i=m_U; i<m_U + m_L; i++) {
			_Doc d = getLabeledDoc(i-m_U);
			m_nodeList[i] = new _Node(i-m_U, d.getYLabel(), d.getYLabel());
		}
		
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];

		System.out.format("Construct nearest neighbor graph nodes in parallel: L: %d, U: %d\n",  m_L, m_U);
		WaitUntilFinish(ActionType.AT_node);
		
		System.out.format("Construct nearest neighbor graph edges in parallel: L: %d, U: %d\n",  m_L, m_U);
		WaitUntilFinish(ActionType.AT_graph);
	}
	
	void SimilarityCheck() {
		_Node node;
		_Edge neighbor;
		
		int y, uPred, lPred, cPred;
		double dMean = 0, dStd = 0;
		double[][][] prec = new double[3][2][2]; // p@5, p@10, p@20; p, n; U, L;
		double[][][] total = new double[3][2][2];
		int[][][] acc = new int[5][2][2]; // combined prediction, classifier's prediction, labeled neighbors prediction, unlabeled neighbors prediction, optimal
		
		for(int i = 0; i < m_U; i++) {
			node = m_nodeList[i];//nearest neighbor graph
			y = (int)node.m_label;
			
			dMean += node.m_pred - node.m_classifierPred;
			dStd += (node.m_pred - node.m_classifierPred) * (node.m_pred - node.m_classifierPred);
			
			/****Check different prediction methods' performance******/
			cPred = (int)(node.m_classifierPred);
			lPred = (int)(node.weightAvgInLabeledNeighbors()+0.5);
			uPred = (int)(node.weightAvgInUnlabeledNeighbors()+0.5);
			acc[0][y][getLabel(node.m_pred)] ++;
			acc[1][y][cPred] ++;				
			acc[2][y][lPred] ++;
			acc[3][y][uPred] ++;
			if (cPred==y || lPred==y || uPred==y)
				acc[4][y][y] ++;//one of these predictions is correct
			else
				acc[4][y][1-y] ++;
			
			/****Check the nearest unlabeled neighbors******/
			double precision = 0;
			for(int pos=0; pos<m_kPrime; pos++){
				neighbor = node.m_unlabeledEdges.get(pos);
				
				if (getLabel(neighbor.getPred()) == y)//neighbor's prediction against the ground-truth
					precision ++;
				
				if (pos==4) {
					prec[0][y][0] += precision/5.0;
					total[0][y][0] ++;
				} else if (pos==9) {
					prec[1][y][0] += precision/10.0;
					total[1][y][0] ++;
				} else if (pos==19) {
					prec[2][y][0] += precision/20.0;
					total[2][y][0] ++;
					break;
				}
			}
			
			/****Check the nearest labeled neighbors******/
			precision = 0;
			for(int pos=0; pos<m_k; pos++){
				neighbor = node.m_labeledEdges.get(pos);
				
				if ((int)neighbor.getLabel() == y)//neighbor's true label against the ground-truth
					precision ++;
				
				if (pos==4) {
					prec[0][y][1] += precision/5.0;
					total[0][y][1] ++;
				} else if (pos==9) {
					prec[1][y][1] += precision/10.0;
					total[1][y][1] ++;
				} else if (pos==19) {
					prec[2][y][1] += precision/20.0;
					total[2][y][1] ++;
					break;
				}
			}
		}
		
		dMean /= m_U;
		dStd = Math.sqrt(dStd/m_U - dMean*dMean);
		
		System.out.println("\nQuery\tDocs\tP@5\tP@10\tP@20");
		System.out.format("Pos\tU\t%.3f\t%.3f\t%.3f\n", prec[0][1][0]/total[0][1][0], prec[1][1][0]/total[1][1][0], prec[2][1][0]/total[2][1][0]);
		System.out.format("Pos\tL\t%.3f\t%.3f\t%.3f\n", prec[0][1][1]/total[0][1][1], prec[1][1][1]/total[1][1][1], prec[2][1][1]/total[2][1][1]);
		System.out.format("Neg\tU\t%.3f\t%.3f\t%.3f\n", prec[0][0][0]/total[0][0][0], prec[1][0][0]/total[1][0][0], prec[2][0][0]/total[2][0][0]);
		System.out.format("Neg\tL\t%.3f\t%.3f\t%.3f\n\n", prec[0][0][1]/total[0][0][1], prec[1][0][1]/total[1][0][1], prec[2][0][1]/total[2][0][1]);
		
		System.out.format("W-C: %.4f/%.4f\n\n", dMean, dStd);
		
		System.out.format("W TN:%d\tFP:%d\tFN:%d\tTP:%d\n", acc[0][0][0], acc[0][0][1], acc[0][1][0], acc[0][1][1]);
		System.out.format("C TN:%d\tFP:%d\tFN:%d\tTP:%d\n", acc[1][0][0], acc[1][0][1], acc[1][1][0], acc[1][1][1]);
		System.out.format("L TN:%d\tFP:%d\tFN:%d\tTP:%d\n", acc[2][0][0], acc[2][0][1], acc[2][1][0], acc[2][1][1]);
		System.out.format("U TN:%d\tFP:%d\tFN:%d\tTP:%d\n", acc[3][0][0], acc[3][0][1], acc[3][1][0], acc[3][1][1]);
		System.out.format("O TN:%d\tFP:%d\tFN:%d\tTP:%d\n", acc[4][0][0], acc[4][0][1], acc[4][1][0], acc[4][1][1]);
	}
	
	protected void constructGraph(boolean createSparseGraph) {		
		/*** pre-compute the full similarity matrix (except the diagonal) in parallel. ****/
		calcSimilarityInThreads();
		
		/***Set up document mapping for debugging purpose***/
		if (m_debugOutput!=null) {
			for (int i = 0; i < m_U; i++) 
				m_testSet.get(i).setID(i);//record the current position
		}
		
		if (!createSparseGraph) {
			System.out.println("Nearest neighbor graph construction finished!");
			return;//stop here if we want to save memory and construct the graph on the fly (space speed trade-off)
		}

//		the following needs to be carefully revised accordingly! Currently, we only support random walk
//		m_graph = new SparseDoubleMatrix2D(m_U+m_L, m_U+m_L);//we have to create this every time with exact dimension
//		
//		/****Construct the C+scale*\Delta matrix and Y vector.****/
//		double scale = -m_alpha / (m_k + m_beta*m_kPrime), sum, value;
//		int nz = 0;
//		_Node node;
//		for(int i = 0; i < m_U; i++) {
//			node = m_nodeList[i];
//			
//			//set the part of unlabeled nodes. U-U			
//			sum = 0;
//			for(_RankItem n:m_kUU) {
//				value = Math.max(m_beta*n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
//				if (value!=0) {
//					m_graph.setQuick(i, n.m_index, scale * value);
//					m_graph.setQuick(n.m_index, i, scale * value);
//					sum += value;
//					nz ++;
//				}
//			}
//			m_kUU.clear();
//			
//			//Set the part of labeled and unlabeled nodes. L-U and U-L
//			for(int j=0; j<m_L; j++) 
//				m_kUL.add(new _RankItem(m_U+j, getCache(i,m_U+j)));
//			
//			for(_RankItem n:m_kUL) {
//				value = Math.max(n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
//				if (value!=0) {
//					m_graph.setQuick(i, n.m_index, scale * value);
//					m_graph.setQuick(n.m_index, i, scale * value);
//					sum += value;
//					nz ++;
//				}
//			}
//			m_graph.setQuick(i, i, 1-scale*sum);
//			m_kUL.clear();
//		}
//		
//		for(int i=m_U; i<m_L+m_U; i++) {
//			sum = 0;
//			for(int j=0; j<m_U; j++) 
//				sum += m_graph.getQuick(i, j);
//			m_graph.setQuick(i, i, m_M-sum); // scale has been already applied in each cell
//		}
//		
//		System.out.format("Nearest neighbor graph (U[%d], L[%d], NZ[%d]) construction finished!\n", m_U, m_L, nz);
	}
	
	//Test the data set.
	@Override
	public double test(){	
		_Node node;
		
		/***Construct the nearest neighbor graph****/
		constructGraph(true);
		
		/***Perform matrix inverse.****/
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D result = alg.inverse(m_graph);
		
		/***setting up the corresponding weight for the true labels***/
		for(int i=m_U; i<m_L+m_U; i++)
			m_nodeList[i].m_classifierPred *= m_M;
		
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			node = m_nodeList[i];
			
			double pred = 0;
			for(int j=0; j<m_U+m_L; j++)
				pred += result.getQuick(i, j) * m_nodeList[j].m_label;			
			node.m_pred = pred;//prediction for the unlabeled based on the labeled data and pseudo labels
			
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-node.m_pred));			
		}
		
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans;
		for(int i = 0; i < m_U; i++) {
			pred = getLabel(m_nodeList[i].m_pred);
			ans = m_testSet.get(i).getYLabel();
			m_TPTable[pred][ans] += 1;
			
			if (pred != ans) {
				if (m_debugOutput!=null)
					debug(m_testSet.get(i));
			} else 
				acc ++;
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		
		return acc/m_U;
	}
	
	/**Different getLabel methods.**/
	//This is the original getLabel: -|c-p(c)|
	int getLabel(double pred) {
		int label = 0;
		double minScore = Math.abs(pred), score;
		for(int i=1; i<m_classNo; i++) {
			score = Math.abs(i-pred);
			if (score<minScore) {
				minScore = score;
				label = i;
			}
		}
		return label;
	}
	
	//p(c) * exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data
	int getLabel3(double pred){
		for(int i = 0; i < m_classNo; i++)			
			m_cProbs[i] = m_pY[i] * Math.exp(-Math.abs(i-pred)) / m_pYSum[i];
		return Utils.argmax(m_cProbs);
	}
	
	//exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data, without class probabilities.
	int getLabel4(double pred) {		
		for (int i = 0; i < m_classNo; i++)
			m_cProbs[i] = Math.exp(-Math.abs(i - pred)) / m_pYSum[i];
		return Utils.argmax(m_cProbs);
	}
	
	@Override
	protected void debug(_Doc d) { }
	
	@Override
	public int predict(_Doc doc) {
		return m_classifier.predict(doc); //we don't support this in transductive learning
	}
	
	@Override
	public double score(_Doc doc, int label) {
		return -1; //we don't support this in transductive learning
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation) {
		
	}
}
