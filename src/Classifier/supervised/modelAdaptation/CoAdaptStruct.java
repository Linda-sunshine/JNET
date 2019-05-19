/**
 * 
 */
package Classifier.supervised.modelAdaptation;

import java.util.ArrayList;
import java.util.Collection;

import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import structures._RankItem;
import structures._Review;
import structures._User;

/**
 * @author Hongning Wang
 * interface for incorporating shared operations for manipulating neighbors
 */
public interface CoAdaptStruct {
	
	//add a neighbor indexed by id to the current user
	public void addNeighbor(int id, double similarity);
	
	//add this current user as neighbor of id
	public void addReverseNeighbor(int id, double similarity);
	
	//get all the neighbors
	public Collection<_RankItem> getNeighbors();
	
	//get all the users who have current user has their neighbors
	public Collection<_RankItem> getReverseNeighbors();
	
	//get similarity between two users
	public double getSimilarity(CoAdaptStruct user, SimType sType);
	
	//get the current user
	public _User getUser();
	
	//get adaptation review size
	public int getAdaptationSize();
	
	//get reviews from this user
	public ArrayList<_Review> getReviews();
}
