package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer consists analysis based on social connections.
 * The analyzer is also used for link prediction/friend recommendation.
 * The analyzer also contains codes for cf of TUIR by Lu.
 */
public class MultiThreadedLinkPredAnalyzer extends MultiThreadedUserAnalyzer {

    protected HashSet<String> m_userIDs;

    public MultiThreadedLinkPredAnalyzer(String tokenModel, int classNo,
                                     String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
        m_userIDs = new HashSet<>();
    }

    /** Construct user network for analysis****/
    // key: user id; value: friends array.
    HashMap<String, String[]> m_trainMap = new HashMap<String, String[]>();
    HashMap<String, String[]> m_testMap = new HashMap<String, String[]>();

    public HashMap<String, String[]>  getTrainMap(){
        return m_trainMap;
    }
    public HashMap<String, String[]> getTestMap(){
        return m_testMap;
    }

    public void buildTrainFriendship(String filename){
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null){
                analyzerOneLineNetwork(m_trainMap, line);
            }
            reader.close();
            System.out.format("%d users have friends!", m_trainMap.size());
            // map friends to users.
            int count = 0;
            for(_User u: m_users){
                if(m_trainMap.containsKey(u.getUserID())){
                    count++;
                    u.setFriends(m_trainMap.get(u.getUserID()));
                }
            }
            System.out.format("%d users' friends are set!\n", count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // collect the interaction information
    protected void analyzerOneLineNetwork(HashMap<String, String[]> map, String line){
        String[] users = line.trim().split("\t");
        String[] friends = Arrays.copyOfRange(users, 1, users.length);
        m_userIDs.add(users[0]);

        if(friends.length == 0){
            return;
        } else
            map.put(users[0], friends);
    }

    public void buildNonFriendship(String filename){

        System.out.println("[Info]Non-friendship file is loaded from " + filename);
        HashMap<String, String[]> nonFriendMap = new HashMap<String, String[]>();

        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null){
                analyzerOneLineNetwork(nonFriendMap, line);
            }
            reader.close();
            System.out.format("%d users have non-friends!\n", nonFriendMap.size());
            // map friends to users.
            int count = 0;
            for(_User u: m_users){
                if(nonFriendMap.containsKey(u.getUserID())){
                    count++;
                    u.setNonFriends(nonFriendMap.get(u.getUserID()));
                }
            }
            System.out.format("%d users' non-friends are set!\n", count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load the test user friends, for link prediction only
    public void loadTestFriendship(String filename){
        try{
            m_testMap.clear();
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            String[] users, friends;
            while((line = reader.readLine()) != null){
                users = line.trim().split("\t");
                friends = Arrays.copyOfRange(users, 1, users.length);
                if(friends.length == 0){
                    continue;
                }
                m_testMap.put(users[0], friends);
            }
            reader.close();
            // map friends to users.
            for(_User u: m_users){
                if(m_testMap.containsKey(u.getUserID()))
                    u.setTestFriends(m_testMap.get(u.getUserID()));
            }
            checkFriendSize();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // Added by Lin. Load user weights from learned models to construct neighborhood.
    public void loadUserWeights(String folder, String suffix){
        if(folder == null || folder.isEmpty())
            return;
        String userID;
        int userIndex, count = 0;
        double[] weights;
        constructUserIDIndex();
        File dir = new File(folder);

        if(!dir.exists()){
            System.err.print("[Info]Directory doesn't exist!");
        } else{
            for(File f: dir.listFiles()){
                if(f.isFile() && f.getName().endsWith(suffix)){
                    int endIndex = f.getName().lastIndexOf(".");
                    userID = f.getName().substring(0, endIndex);
                    if(m_userIDIndex.containsKey(userID)){
                        userIndex = m_userIDIndex.get(userID);
                        weights = loadOneUserWeight(f.getAbsolutePath());
                        m_users.get(userIndex).setSVMWeights(weights);
                        count++;
                    }
                }
            }
        }
        System.out.format("%d users weights are loaded!\n", count);
    }

    public double[] loadOneUserWeight(String fileName) {
        double[] weights = new double[getFeatureSize()];
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fileName), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] ws = line.split(",");
                if (ws.length != getFeatureSize() + 1)
                    System.out.println("[error]Wrong dimension of the user's weights!");
                else {
                    weights = new double[ws.length];
                    for (int i = 0; i < ws.length; i++) {
                        weights[i] = Double.valueOf(ws[i]);
                    }
                }
            }
            reader.close();
        } catch (IOException e) {
            System.err.format("[Error]Failed to open file %s!!", fileName);
            e.printStackTrace();
        }
        return weights;
    }

    // save the user-user pairs to graphlab for model training.
    public void saveUserUserPairs(String dir){
        int trainUser = 0, testUser = 0, trainPair = 0, testPair = 0;
        try{
            PrintWriter trainWriter = new PrintWriter(new File(dir+"train.csv"));
            PrintWriter testWriter = new PrintWriter(new File(dir+"test.csv"));
            trainWriter.write("user_id,item_id,rating\n");
            testWriter.write("user_id,item_id,rating\n");
            for(_User u: m_users){
                if(u.getFriendSize() != 0){
                    trainUser++;
                    for(String frd: u.getFriends()){
                        trainPair++;
                        trainWriter.write(String.format("%s,%s,%d\n", u.getUserID(), frd, 1));
                        trainWriter.write(String.format("%s,%s,%d\n", frd, u.getUserID(), 1));

                    }
                }
                // for test users, we also need to write out non-friends
                if(u.getTestFriendSize() != 0){
                    testUser++;
                    for(_User nei: m_users){
                        String neiID = nei.getUserID();
                        if(u.hasFriend(neiID) || u.getUserID().equals(neiID))
                            continue;
                        else if(u.hasTestFriend(neiID)){
                            testPair++;
                            testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), neiID, 1));
                            testWriter.write(String.format("%s,%s,%d\n", neiID, u.getUserID(), 1));
                        } else if(m_trainMap.containsKey(neiID)){
                            testPair++;
                            testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), neiID, 0));
                            testWriter.write(String.format("%s,%s,%d\n", neiID, u.getUserID(), 0));
                        }
                    }
                }
            }
            trainWriter.close();
            testWriter.close();
            System.out.format("[Info]Finish writing (%d,%d) training users/pairs, (%d,%d) testing users/pairs.\n", trainUser, trainPair, testUser, testPair);
        } catch(IOException e){
            e.printStackTrace();
        }

    }
    public void checkFriendSize(){
        int train = 0, test = 0;
        for(_User u: m_users){
            if(u.getFriendSize() != 0)
                train++;
            if(u.getTestFriendSize() != 0)
                test++;
        }
        System.out.format("[Check]%d users have train friends, %d users have test friends.\n", train, test);
    }

    // filter the friends who are not in the list and return a neat hashmap
    public HashMap<String, ArrayList<String>> filterFriends(HashMap<String, String[]> neighborsMap){
        double sum = 0;
        HashMap<String, _User> userMap = new HashMap<String, _User>();
        for(_User u: m_users){
            userMap.put(u.getUserID(), u);
        }
        HashMap<String, ArrayList<String>> frdMap = new HashMap<String, ArrayList<String>>();
        for(String uid: neighborsMap.keySet()){
            if(!userMap.containsKey(uid)){
                System.out.println("The user does not exist in user set!");
                continue;
            }
            ArrayList<String> frds = new ArrayList<>();
            for(String frd: neighborsMap.get(uid)){
                if(!neighborsMap.containsKey(frd))
                    continue;
                if(contains(neighborsMap.get(frd), uid)){
                    frds.add(frd);
                } else {
                    System.out.println("asymmetric");
                }
            }
            if(frds.size() > 0){
                frdMap.put(uid, frds);
                sum += frds.size();
            }
        }
        System.out.format("%d users' friends are recorded, avg friends: %.2f.\n", frdMap.size(), sum/frdMap.size());
        return frdMap;
    }

    public boolean contains(String[] strs, String str){
        if(strs == null || strs.length == 0)
            return false;
        for(String s: strs){
            if(str.equals(s))
                return true;
        }
        return false;
    }

    public void rmMultipleReviews4OneItem(){
        Set<String> items = new HashSet<String>();
        ArrayList<Integer> indexes = new ArrayList<Integer>();
        int uCount = 0, rCount = 0;
        boolean flag = false;
        for(_User u: m_users){
            ArrayList<_Review> reviews = u.getReviews();
            items.clear();
            indexes.clear();
            for(int i=0; i<reviews.size(); i++){
                _Review r = reviews.get(i);
                if(items.contains(r.getItemID())){
                    indexes.add(i);
                    rCount++;
                    flag = true;
                } else {
                    items.add(r.getItemID());
                }
            }
            // record the user number
            if(flag){
                uCount++;
                flag = false;
            }
            // remove the reviews.
            Collections.sort(indexes, Collections.reverseOrder());
            for(int idx: indexes){
                reviews.remove(idx);
            }
            u.constructTrainTestReviews();
        }
        System.out.format("%d users have %d duplicate reviews for items.\n", uCount, rCount);
    }

    /***
     * The following codes are used in cf for ETBIR.
     */
    HashMap<String, _User> m_userMap = new HashMap<String, _User>();
    //Load users' test reviews.
    public void loadTestUserDir(String folder){

        // construct the training user map first
        for(_User u: m_users){
            if(!m_userMap.containsKey(u.getUserID()))
                m_userMap.put(u.getUserID(), u);
            else
                System.err.println("[error] The user already exists in map!!");
        }

        if(folder == null || folder.isEmpty())
            return;

        File dir = new File(folder);
        final File[] files=dir.listFiles();
        ArrayList<Thread> threads = new ArrayList<Thread>();
        for(int i=0;i<m_numberOfCores;++i){
            threads.add(  (new Thread() {
                int core;
                @Override
                public void run() {
                    try {
                        for (int j = 0; j + core <files.length; j += m_numberOfCores) {
                            File f = files[j+core];
                            // && f.getAbsolutePath().endsWith("txt")
                            if(f.isFile()){//load the user
                                loadTestUserReview(f.getAbsolutePath(),core);
                            }
                        }
                    } catch(Exception ex) {
                        ex.printStackTrace();
                    }
                }

                private Thread initialize(int core ) {
                    this.core = core;
                    return this;
                }
            }).initialize(i));

            threads.get(i).start();
        }
        for(int i=0;i<m_numberOfCores;++i){
            try {
                threads.get(i).join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // process sub-directories
        int count=0;
        for(File f:files )
            if (f.isDirectory())
                loadUserDir(f.getAbsolutePath());
            else
                count++;

        System.out.format("%d users are loaded from %s...\n", count, folder);
    }

    // Load one file as a user here.
    protected void loadTestUserReview(String filename, int core){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            String userID = extractUserID(file.getName()); //UserId is contained in the filename.

            // Skip the first line since it is user name.
            reader.readLine();

            String productID, source, category="";
            ArrayList<_Review> reviews = new ArrayList<_Review>();

            _Review review;
            int ylabel;
            long timestamp=0;
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
                    if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
                        reviews.add(review);
                    }
                }
            }
            if(reviews.size() > 1){//at least one for adaptation and one for testing
                synchronized (m_allocReviewLock) {
                    if(m_userMap.containsKey(userID)){
                        m_userMap.get(userID).setTestReviews(reviews);
                    }
                }
            } else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
                review = reviews.get(0);
                synchronized (m_rollbackLock) {
                    rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
                }
            }

            reader.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }
}
