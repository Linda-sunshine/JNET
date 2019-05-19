package Classifier.supervised.modelAdaptation;

import java.util.ArrayList;
import java.util.HashMap;

import structures._User;
import structures._PerformanceStat.TestMode;
/**
 * This class provides the base measurement for each user.
 * @author lin
 */
public class Base extends ModelAdaptation {

	public Base(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_testmode = TestMode.TM_batch;
	}
	@Override
	public String toString() {
		return String.format("Base Model.");
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList) {
			m_userList.add(new _AdaptStruct(user));
			user.initModel(m_featureSize+1);
		}
	}

	@Override
	public void setPersonalizedModel() {
		for(_AdaptStruct u: m_userList){
			u.setPersonalizedModel(m_gWeights);
		}
	}
}
