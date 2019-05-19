package Classifier.supervised.modelAdaptation.HDP;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._Doc.rType;
import structures._HDPThetaStar;
import structures._Review;

import java.util.HashMap;

/***
 * In the class, we consider both:
 * 1. The confidence for each review;
 * 2. Multiple sampling of the cluster assignment.
 * @author lin
 */
public class MTCLinAdaptWithHDPDualConfidence extends MTCLinAdaptWithHDPConfidence{

	public MTCLinAdaptWithHDPDualConfidence(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}

	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPConfidenceDual[dim:%d,supDim:%d,lmDim:%d,thinning:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_thinning,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	protected void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		super.sampleOneInstance(user, r);
		// We also put the sampled cluster to the review for later MLE.
		r.updateThetaCountMap(1);
	}
	
	public void clearReviewStats(){
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;//do not touch testing reviews!
				r.clearThetaCountMap();	
			}
		}
	}
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0, ecount = 0;
		
		init(); // clear user performance and init cluster assignment	

		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			calculate_M_step();
		}
		System.out.println("[Info]Burn in period ends, starts iteration...");
		clearReviewStats();

		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			while(ecount++ < m_thinning){
				calculate_E_step();
				assignClusterIndex();		
				sampleGamma();
			}
			
			curLikelihood += estPhi();
			delta = (lastLikelihood - curLikelihood)/curLikelihood;
				
			// After M step, we need to clear the review stats and start collecting again.
			ecount = 0;
			clearReviewStats();
			
			evaluateModel();
				
//			printInfo(i%5==0);//no need to print out the details very often
			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}
	
	// In function logLikelihood, we update the loglikelihood and corresponding gradients.
	// Thus, we only need to update the two functions correspondingly with.
	protected double calcLogLikelihoodY(_Review r){
		int index = -1;
		_HDPThetaStar oldTheta = r.getHDPThetaStar();
		HashMap<_HDPThetaStar, Integer> thetaCountMap = r.getThetaCountMap();
		double likelihood = 0;
		for(_HDPThetaStar theta: thetaCountMap.keySet()){
			index = findHDPThetaStar(theta);
			// some of the cluster may disappear, ignore them.
			if(index >= m_kBar || index < 0)
				continue;
			r.setHDPThetaStar(theta);
			// log(likelihood^k) = k * log likelihood.
			likelihood += thetaCountMap.get(theta) * super.calcLogLikelihoodY(r);
		}
		r.setHDPThetaStar(oldTheta);
		return likelihood;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		int index = -1;
		double confidence = 1;
		_Review r = (_Review) review;
		_HDPThetaStar oldTheta = r.getHDPThetaStar();
		HashMap<_HDPThetaStar, Integer> thetaCountMap = r.getThetaCountMap();
		
		for(_HDPThetaStar theta: thetaCountMap.keySet()){
			index = findHDPThetaStar(theta);
			// some of the cluster may disappear, ignore them.
			if(index >= m_kBar || index < 0)
				continue;
			r.setHDPThetaStar(theta);
			confidence = thetaCountMap.get(theta);
			// confidence plays the role of weight here, how many times the review shows in the cluster.
			super.gradientByFunc(u, review, confidence, g);
		}
		r.setHDPThetaStar(oldTheta);
	}
}
