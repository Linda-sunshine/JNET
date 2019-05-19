package Classifier.semisupervised;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures.*;
import structures._Doc.rType;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

public class GaussianFieldsByRandomWalkWithFriends extends GaussianFieldsByRandomWalk{


	// key: user id; value: friends id.
	HashMap<String, String[]> m_neighborsMap;
	// key: user id: value: _AdaptStruct.
	HashMap<String, _AdaptStruct> m_userMap;
	
	public GaussianFieldsByRandomWalkWithFriends(_Corpus c, String classifier,
			double C, double ratio, int k, int kPrime, double alhpa,
			double beta, double delta, double eta, boolean weightedAvg) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta, weightedAvg);
		m_neighborsMap = new HashMap<String, String[]>();
		m_userMap = new HashMap<String, _AdaptStruct>();
	}

	public void constructTrainTestDocs(ArrayList<_User> users){
		for(_User u: users){
			// put the user id and corresponding adaptstruct inside the map
			if(!m_userMap.containsKey(u.getUserID()))
				m_userMap.put(u.getUserID(), new _AdaptStruct(u));
			
			for(_Review r: u.getReviews()){
				if(r.getType() == rType.ADAPTATION)
					m_trainSet.add(r);
				else {
					m_testSet.add(r);
				}
			}
		}
	}
	
	//Train the data set.
	@Override
	public double train(){
		return super.train(m_trainSet);
	}
	
	@Override
	public double test(){
		/***Construct the nearest neighbor graph****/
		constructGraph(false);
		
		if (m_pred_last==null || m_pred_last.length<m_U)
			m_pred_last = new double[m_U]; //otherwise we can reuse the current memory
		
		//record the starting point
		for(int i=0; i<m_U; i++)
			m_pred_last[i] = m_nodeList[i].m_pred;//random walk starts from multiple learner
		
		/***use random walk to solve matrix inverse***/
		System.out.println("Random walk starts:");
		int iter = 0;
		double diff = 0, accuracy;
		do {
			if (m_weightedAvg)
				accuracy = randomWalkByWeightedSum();	
			else
				accuracy = randomWalkByMajorityVote();
			
			diff = updateFu();
			System.out.format("Iteration %d, converge to %.3f with accuracy %.4f...\n", ++iter, diff, accuracy);
		} while(diff > m_delta && iter<50);//maximum 50 iterations 
		
		/***check the purity of newly constructed neighborhood graph after random walk with ground-truth labels***/
		SimilarityCheck();
		//checkSimilarityVariance();
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_nodeList[i].m_pred));			
		}
		
		/***evaluate the performance***/
		double acc = 0;
		int predL, trueL;
		String uid;
		_AdaptStruct user;
		_PerformanceStat userPerfStat;
		
		double[] accStat = new double[10];
		for(int i = 0; i < m_U; i++) {
			//pred = getLabel(m_fu[i]);
			predL = getLabel(m_nodeList[i].m_pred);
			//fetch the review first
			_Review r = (_Review) m_testSet.get(i);
			trueL = m_testSet.get(i).getYLabel();
			r.setPredictLabel(predL);
			uid = r.getUserID();
			user = m_userMap.get(uid);
			userPerfStat = user.getPerfStat();
			userPerfStat.addOnePredResult(predL, trueL);			
		}
		
		ArrayList<ArrayList<Double>> macroF1 = new ArrayList<ArrayList<Double>>();
		// init macroF1
		for(int i=0; i<m_classNo; i++)
			macroF1.add(new ArrayList<Double>());
		// init microF1
		m_microStat.clear();
		
		for(String u: m_userMap.keySet()) {
			// after assigning all testing reviews, calc each user's prf
			userPerfStat = m_userMap.get(u).getPerfStat();
			userPerfStat.calculatePRF();
			for(int i=0; i<m_classNo; i++){
				if(userPerfStat.getTrueClassNo(i) > 0)
					macroF1.get(i).add(userPerfStat.getF1(i));
			}
			m_microStat.accumulateConfusionMat(userPerfStat);
		}
		System.out.print("neg users: " + macroF1.get(0).size());
		System.out.print("\tpos users: " + macroF1.get(1).size()+"\n");

		System.out.println(toString());
		calcMicroPerfStat();
		// macro average and standard deviation.
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++){
			double[] avgStd = calcAvgStd(macroF1.get(i));
			System.out.format("Class %d: %.4f+%.4f\t", i, avgStd[0], avgStd[1]);
		}
		return 0;
	}

	// calculate the average and sd
	public double[] calcAvgStd(ArrayList<Double> fs){
		double avg = 0, std = 0;
		for(double f: fs)
			avg += f;
		avg /= fs.size();
		for(double f: fs)
			std += (f - avg) * (f - avg);
		std = Math.sqrt(std/fs.size());
		return new double[]{avg, std};
	}
	// pass the friendship built somewhere else to random walk
	public void setFriendship(HashMap<String, String[]> neighborsMap){
		m_neighborsMap = neighborsMap;
	}
	
	public boolean isFriend(String ui, String uj){
		String[] neighbors = m_neighborsMap.get(ui);
		for(String nei: neighbors){
			if(nei.equals(uj))
				return true;
		}
		return false;
	}
	
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		_Review ri = (_Review) di;
		_Review rj = (_Review) dj;
		if(isFriend(ri.getUserID(), rj.getUserID()))
			return Math.exp(getBoWSim(di, dj));
		else
			return 0;
	}
	
	// added by Lin for model performance comparison.
	// print out each user's test review's performance.
	public void printUserPerformance(String filename){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			ArrayList<_AdaptStruct> userList = new ArrayList<_AdaptStruct>();
			for(String u: m_userMap.keySet())
				userList.add(m_userMap.get(u));
			Collections.sort(userList, new Comparator<_AdaptStruct>(){
				@Override
				public int compare(_AdaptStruct u1, _AdaptStruct u2){
					return String.CASE_INSENSITIVE_ORDER.compare(u1.getUserID(), u2.getUserID());
				}
			});
			for(_AdaptStruct u: userList){
				writer.write("-----\n");
				writer.write(String.format("%s\t%d\n", u.getUserID(), u.getReviews().size()));
				for(_Review r: u.getReviews()){
					if(r.getType() == rType.ADAPTATION)
						writer.write(String.format("%s\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getSource()));
					if(r.getType() == rType.TEST){
						writer.write(String.format("%s\t%d\t%d\t%s\n", r.getCategory(), r.getYLabel(), r.getPredictLabel(), r.getSource()));
					}
				}
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
