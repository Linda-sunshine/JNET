package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import cern.jet.random.tdouble.Gamma;

import structures._HDPThetaStar;
import structures._Review;
import structures._WeightedCount;
import utils.Utils;

public class MTCLinAdaptWithHDPConfidence extends MTCLinAdaptWithHDP{

	public MTCLinAdaptWithHDPConfidence(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, lm);
	}
	
	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithHDPConfidence[dim:%d,supDim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),supScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",
											m_dim,m_dimSup,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_eta3,m_eta4,m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	// Write this as an independent function for overriding purpose.
	public void incUserHDPThetaStarMemSize(_HDPAdaptStruct user, _Review r){
		double confidence = calcLogLikelihoodY(r);
		confidence = Math.exp(confidence);
		confidence = 1 - confidence * (1 - confidence);
		r.setConfidence(confidence);
		user.incHDPThetaStarMemSize(r.getHDPThetaStar(), 1);//-->3	
		user.addHDPThetaStarMem(r.getHDPThetaStar(), r);
	}	
	
	// For later overwritten methods.
	public double calcGroupPopularity(_HDPAdaptStruct user, int k, double gamma_k){
		return user.getWeightedHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k;
	}
	
	public void updateDocMembership(_HDPAdaptStruct user, _Review r){
		int index = -1;
		_HDPThetaStar curThetaStar = r.getHDPThetaStar();
		
		//Step 1: remove the current review from the thetaStar and user side.
		user.incHDPThetaStarMemSize(r.getHDPThetaStar(), -1);
		user.rmHDPThetaStarMem(r.getHDPThetaStar(), r);
		curThetaStar.updateMemCount(-1);
		curThetaStar.rmLMStat(r.getLMSparse());

		if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
			m_gamma_e += curThetaStar.getGamma();
			index = findHDPThetaStar(curThetaStar);
			swapTheta(m_kBar-1, index); // move it back to \theta*
			m_kBar --;
		}
	}
}
