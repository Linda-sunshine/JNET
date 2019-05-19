package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Doc.rType;
import structures._Review;
import structures._User;
import structures._stat;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;

/**
 *
 * @author Hongning Wang
 * Loading text file format for amazon and yelp reviews
 */
public class UserAnalyzer extends DocAnalyzer {

	ArrayList<_User> m_users; // Store all users with their reviews.
	double m_trainRatio = 0.25; // by default, the first 25% for training the global model
	double m_adaptRatio = 0.5; // by default, the next 50% for adaptation, and rest 25% for testing
	int m_trainSize = 0, m_adaptSize = 0, m_testSize = 0;
	double m_pCount[] = new double[3]; // to count the positive ratio in train/adapt/test
	boolean m_enforceAdapt = false;

	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, boolean b)
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
		m_users = new ArrayList<_User>();
	}

	public void config(double train, double adapt, boolean enforceAdpt) {
		if (train<0 || train>1) {
			System.err.format("[Error]Incorrect setup of training ratio %.3f, which has to be in [0,1]\n", train);
			return;
		} else if (adapt<0 || adapt>1) {
			System.err.format("[Error]Incorrect setup of adaptation ratio %.3f, which has to be in [0,1]\n", adapt);
			return;
		} else if (train+adapt>1) {
			System.err.format("[Error]Incorrect setup of training and adaptation ratio (%.3f, %.3f), whose sum has to be in (0,1]\n", train, adapt);
			return;
		}

		m_trainRatio = train;
		m_adaptRatio = adapt;
		m_enforceAdapt = enforceAdpt;
	}

	// Load the features from a file and store them in the m_featurNames.@added by Lin.
	@Override
	protected boolean LoadCV(String filename) {
		if(m_newCV){
			return loadNewCV(filename);
		} else
			return loadOldCV(filename);

	}
	protected boolean loadOldCV(String filename){
		if (filename==null || filename.isEmpty())
			return false;

		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, stats[];
			int ngram = 0, DFs[]= {0, 0};
			m_Ngram = 1;//default value of Ngram
			while ((line = reader.readLine()) != null) {
				stats = line.split(",");

				if (stats[1].equals("TOTALDF")) {
					m_TotalDF = (int)(Double.parseDouble(stats[2]));
				} else {
					expandVocabulary(stats[1]);
					DFs[0] = (int)(Double.parseDouble(stats[3]));
					DFs[1] = (int)(Double.parseDouble(stats[4]));
					setVocabStat(stats[1], DFs);

					ngram = 1+Utils.countOccurrencesOf(stats[1], "-");
					if (m_Ngram<ngram)
						m_Ngram = ngram;
				}
			}
			reader.close();

			System.out.format("Load %d %d-gram old features from %s...\n", m_featureNames.size(), m_Ngram, filename);
			m_isCVLoaded = true;
//			m_isCVStatLoaded = true;
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}

	// Load the new cv.
	protected boolean loadNewCV(String filename){
		if (filename==null || filename.isEmpty())
			return false;

		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			m_Ngram = 1;//default value of Ngram

			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#")){//comments
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}
				} else
					expandVocabulary(line);
			}
			reader.close();
			System.out.format("Load %d %d-gram new features from %s...\n", m_featureNames.size(), m_Ngram, filename);
			m_isCVLoaded = true;
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}

	void setVocabStat(String term, int[] DFs) {
		_stat stat = m_featureStat.get(term);
		stat.setRawDF(DFs);
	}

	//Load all the users.
	public void loadUserDir(String folder){
		int count = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile()){// && f.getAbsolutePath().endsWith("txt")){
				loadUser(f.getAbsolutePath());
				count++;
				if(count % 100 == 0)
					System.out.print("*");
				if(count % 5000 == 0)
					System.out.println();
			} else if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
		}
		System.out.format("%d users are loaded from %s...\n", count, folder);
	}

	String extractUserID(String text) {
		int index = text.indexOf('.');
		if (index==-1)
			return text;
		else
			return text.substring(0, index);
	}

	// Load one file as a user here.
	public void loadUser(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.
			// Skip the first line since it is user name.
			reader.readLine();

			String productID, source, category;
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			_Review review;
			int ylabel;
			long timestamp;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());

				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review)) //Create the sparse vector for the review.
						reviews.add(review);
				}
			}

			if(reviews.size() > 1){//at least one for adaptation and one for testing
				allocateReviews(reviews);
				m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
			} else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
				review = reviews.get(0);
				rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	//[0, train) for training purpose
	//[train, adapt) for adaptation purpose
	//[adapt, 1] for testing purpose
	void allocateReviews(ArrayList<_Review> reviews) {
		Collections.sort(reviews);// sort the reviews by timestamp
		int train = (int)(reviews.size() * m_trainRatio), adapt;
		if (m_enforceAdapt)
			adapt = Math.max(1, (int)(reviews.size() * (m_trainRatio + m_adaptRatio)));
		else
			adapt = (int)(reviews.size() * (m_trainRatio + m_adaptRatio));

		_Review r;
		for(int i=0; i<reviews.size(); i++) {
			r = reviews.get(i);
			if (i<train) {
				r.setType(rType.TRAIN);
				if (r.getYLabel()==1)
					m_pCount[0] ++;

				m_trainSize ++;
			} else if (i<adapt) {
				r.setType(rType.ADAPTATION);
				if (r.getYLabel()==1)
					m_pCount[1] ++;

				m_adaptSize ++;
			} else {
				r.setType(rType.TEST);
				if (r.getYLabel()==1)
					m_pCount[2] ++;

				m_testSize ++;
			}
		}
	}

	//Return all the users.
	public ArrayList<_User> getUsers(){
		System.out.format("[Info]Training size: %d(%.2f), adaptation size: %d(%.2f), and testing size: %d(%.2f)\n",
				m_trainSize, m_trainSize>0?m_pCount[0]/m_trainSize:0.0,
				m_adaptSize, m_adaptSize>0?m_pCount[1]/m_adaptSize:0.0,
				m_testSize, m_testSize>0?m_pCount[2]/m_testSize:0.0);
		return m_users;
	}

	// Added by Lin: Load the svd file to get the low dim representation of users.
	public void loadSVDFile(String filename){
		try{
			// Construct the <userID, user> map first.
			int count = 0;
			HashMap<String, double[]> IDLowDimMap = new HashMap<String, double[]>();

			int skip = 3;
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line, userID;
			String[] strs;
			double[] lowDims;
			//Skip the first three lines.
			while(skip-- > 0)
				reader.readLine();
			while((line = reader.readLine()) != null){
				strs = line.split("\\s+");
				userID = strs[0];
				lowDims = new double[strs.length - 1];
				for(int i=1; i<strs.length; i++)
					lowDims[i-1] = Double.valueOf(strs[i]);
				IDLowDimMap.put(userID, lowDims);
				count++;
			}
			// Currently, there are missing low dimension representation of users.
			for(_User u: m_users){
				if(IDLowDimMap.containsKey(u.getUserID()))
					u.setLowDimProfile(IDLowDimMap.get(u.getUserID()));
				else {
					System.out.println("[Warning]" + u.getUserID() + " : low dim profile missing.");
					u.setLowDimProfile(new double[11]);
				}
			}
			reader.close();
			System.out.format("Ther are %d users and %d users' low dimension profile are loaded.\n", m_users.size(), count);
		} catch (IOException e){
			e.printStackTrace();
		}
	}

	public Collection<_Doc> mergeReviews(){
		Collection<_Doc> rvws = new ArrayList<_Doc>();
		for(_User u: m_users){
			rvws.addAll(u.getReviews());
		}
		return rvws;
	}
}