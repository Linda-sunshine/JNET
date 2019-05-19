/**
 * 
 */
package structures;

import Classifier.supervised.modelAdaptation._AdaptStruct;

/**
 * @author Hongning Wang
 * Used in online updating of adaptation methods
 */
public class _UserReviewPair implements Comparable<_UserReviewPair> {
	_AdaptStruct m_user;
	_Review m_review;
	
	public _UserReviewPair(_AdaptStruct u, _Review r) {
		m_user = u;
		m_review = r;
	}
	
	public _AdaptStruct getUser() {
		return m_user;
	}
	
	public _Review getReview() {
		return m_review;
	}
	
	@Override
	public int compareTo(_UserReviewPair p) {
		return m_review.compareTo(p.getReview());
	}
}
