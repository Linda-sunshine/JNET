/**
 * 
 */
package structures;

import java.util.PriorityQueue;
import java.util.Random;
import java.util.Vector;


/**
 * @author wang296
 * @param <E> objects to be ranked, large elements in the front
 *
 */
public class MyPriorityQueue<E extends Comparable<? super E>> extends Vector<E> {
	private static final long serialVersionUID = 2620787807800527871L;
	int m_size;
	boolean m_descending;
	
	public MyPriorityQueue(int size){
		super();
		m_size = size;
		m_descending = true;
	}
	
	public MyPriorityQueue(int size, boolean descending){
		super();
		m_size = size;
		m_descending = descending;
	}
	
	private int compare(E o1, E o2){
		if (m_descending)
			return o1.compareTo(o2);
		else
			return o2.compareTo(o1);
	}
	
	public boolean add(E object){
		if (size()<m_size)
			return insert(object);
		else{			
			if (compare(lastElement(), object)>=0)
				return false;//directly discard this
			else{
				insert(object);
				removeElementAt(m_size);//remove the last element
				return true;
			}
		}
	}
	
	private boolean insert(E object){
		if (super.isEmpty())
			return super.add(object);
		else{
			super.add(find(object, 0, super.size()), object);
			return true;
		}
	}
	
	private int find(E object, int startPos, int endPos){
		if (startPos==endPos)
			return startPos;//append to the end
		
		int mid = (startPos + endPos)/2, result = compare(object, get(mid));
		if (result==0)
			return mid;
		else if (result>0)
			return find(object, startPos, mid);
		else
			return find(object, mid+1, endPos);
	}
	
	static public void main(String[] args){
		//efficiency comparison
		Vector<Integer> container = new Vector<Integer>();
		Random rand = new Random(); 
		for(int i=0; i<2000000; i++)
			container.add(rand.nextInt());
		
		//Testing my priority queue
		long time = System.currentTimeMillis();
		MyPriorityQueue<Integer> test = new MyPriorityQueue<Integer>(10, false);
		for(Integer val:container)
			test.add(val);
		for(Integer i : test)
			System.out.println(i);
		System.out.println(System.currentTimeMillis()-time);
		
		//Testing the system one
		time = System.currentTimeMillis();
		PriorityQueue<Integer> test2 = new PriorityQueue<Integer>();
		for(Integer val:container)
			test2.add(val);
		
		for(int i=0; i<10; i++)
			System.out.println(test2.poll());
		System.out.println(System.currentTimeMillis()-time);
	}
}
