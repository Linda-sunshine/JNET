package structures;

import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
/***
 * @author lin
 * The structure is used to represent the edge in mmb model.
 */
public class _MMBNeighbor {
	_HDPAdaptStruct m_uj;
	_HDPThetaStar m_hdpThetaStar;
	int m_eij;
	
	public _MMBNeighbor(_HDPAdaptStruct uj, _HDPThetaStar theta, int eij){
		m_uj = uj;
		m_hdpThetaStar = theta;
		m_eij = eij;
	}
		
	public _HDPThetaStar getHDPThetaStar(){
		return m_hdpThetaStar;
	}
	
	public int getHDPThetaStarIndex(){
		return m_hdpThetaStar.getIndex();
	}
	
	public int getEdge(){
		return m_eij;
	}
}
