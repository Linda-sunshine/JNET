package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer aims at the directional network information.
 */
public class MultiThreadedNetworkAnalyzer extends MultiThreadedLinkPredAnalyzer {

    Random m_rand = new Random();
    HashMap<String, HashSet<String>> m_networkMap = new HashMap<String, HashSet<String>>();

    public MultiThreadedNetworkAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // save one file for indexing users in later use (construct network for baselines)
    public void saveUserIds(String filename){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<m_users.size(); i++){
                writer.format("%s\t%d\n", m_users.get(i).getUserID(), i);
            }
            writer.close();
            System.out.format("Finish saving %d user ids.\n", m_users.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }
    // save the network for later use
    public void saveNetwork(String filename){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_networkMap.keySet()){
                writer.write(uid + '\t');
                for(String it: m_networkMap.get(uid)){
                    writer.write(it + '\t');
                }
                writer.write('\n');
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load the interactions, filter the users who are not in the user
    public void loadInteractions(String filename){
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            double totalEdges = 0;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String uid = users[0];
                if(!m_userIDIndex.containsKey(users[0])){
                    System.err.println("The user does not exist in user set!");
                    continue;
                }
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                if(interactions.length == 0) continue;
                for(String in: interactions){
                    if(in.equals(uid)) continue;
                    if(m_userIDIndex.containsKey(in)){
                        if(!m_networkMap.containsKey(uid))
                            m_networkMap.put(uid, new HashSet<String>());
                        if(!m_networkMap.containsKey(in))
                            m_networkMap.put(in, new HashSet<String>());
                        m_networkMap.get(uid).add(in);
                        m_networkMap.get(in).add(uid);
                    }
                }
            }
            int missing = 0;
            for(String ui: m_networkMap.keySet()){
                for(String frd: m_networkMap.get(ui)){
                    if(!m_networkMap.containsKey(frd))
                        missing++;
                    if(!m_networkMap.get(frd).contains(ui))
                        System.out.println("Asymmetric!!");
                }
            }
            if(missing > 0)
                System.out.println("[error]Some edges are not in the set: " + missing);
            // set the friends for each user
            for(String ui: m_networkMap.keySet()){
                totalEdges += m_networkMap.get(ui).size();
                String[] frds = hashSet2Array(m_networkMap.get(ui));
                m_users.get(m_userIDIndex.get(ui)).setFriends(frds);
            }
            reader.close();
            System.out.format("[Info]Total user size: %d, total doc size: %d, users with friends: %d, total edges: " +
                    "%.3f.\n", m_users.size(), m_corpus.getCollection().size(), m_networkMap.size(), totalEdges);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected String[] hashSet2Array(HashSet<String> strs){
        String[] arr = new String[strs.size()];
        int index = 0;
        for(String str: strs){
            arr[index++] = str;
        }
        return arr;
    }

    // assign cv index for documents in the collection
    public void assignCVIndex4Docs(int k){
        m_corpus.setMasks();

        for(_User u: m_users){
            setMasks4Reviews(u.getReviews(), k);
        }
    }

    // shuffle the document index based on each user
    public void saveCVIndex(String filename){
        ArrayList<_Doc> docs = m_corpus.getCollection();
        int[] stat = new int[5];
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<docs.size(); i++){
                _Review r = (_Review) docs.get(i);
                if(r.getMask4CV() == -1){
                    r.setMask4CV(1);
                }
                writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), r.getMask4CV()));
                stat[r.getMask4CV()]++;
            }
            writer.close();
            System.out.println("[Info]Finish writing cv index! Stat as follow:");
            for(int s: stat)
                System.out.print(s + "\t");
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    // shuffle the document index based on each user
    public void saveCVIndex4TADW(String filename){

        int[] stat = new int[5];
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User u: m_users){
                for(_Review r: u.getReviews()){
                    writer.write(String.format("%s,%s,%d\n", r.getUserID(), r.getItemID(), r.getMask4CV()));
                    stat[r.getMask4CV()]++;
                }
            }
            writer.close();
            System.out.println("[Info]Finish writing cv index for TADW! Stat as follow:");
            for(int s: stat)
                System.out.print(s + "\t");
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    // set masks for one users' all reviews for CV
    public void setMasks4Reviews(ArrayList<_Review> reviews, int k){
        int[] masks = new int[reviews.size()];
        int res = masks.length / k;
        int threshold = res * k;
        for(int i=0; i<masks.length; i++){
            if(i < threshold){
                masks[i] = i % k;
            } else{
                masks[i] = m_rand.nextInt(k);
            }
        }
        shuffle(masks);
        for(int i=0; i< reviews.size(); i++){
            reviews.get(i).setMask4CV(masks[i]);
        }
    }

    // Fisher-Yates shuffle
    public void shuffle(int[] masks){
        int index, tmp;
        for(int i=masks.length-1; i>=0; i--){
            index = m_rand.nextInt(i+1);
            if(index != 1){
                tmp = masks[index];
                masks[index] = masks[i];
                masks[i] = tmp;
            }
        }
    }

    // load cv index for all the documents
    public void loadCVIndex(String filename, int kFold){
        try {
            File file = new File(filename);
            int[] stat = new int[kFold];
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                int id = Integer.valueOf(strs[1]);
                int mask = Integer.valueOf(strs[2]);
                stat[mask]++;
                if(!m_userIDIndex.containsKey(uId))
                    System.out.println("No such user!");
                else {
                    int uIndex = m_userIDIndex.get(uId);
                    if (uIndex > m_users.size())
                        System.out.println("Exceeds the array size!");
                    else {
                        m_users.get(m_userIDIndex.get(uId)).getReviewByID(id).setMask4CV(mask);
                    }
                }
            }
            System.out.println("[Stat]Stat as follow:");
            for(int s: stat)
                System.out.print(s + "\t");
            System.out.println();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    /*** The data structure is used for cv index ***/
    class _Edge4CV{

        protected String m_userId = "";
        protected int m_cvIdx = -1; // -1: not assigned; 0-x: fold index

        public _Edge4CV(String uid, int cvIdx){
            m_userId = uid;
            m_cvIdx = cvIdx;
        }
        public String getUserId(){
            return m_userId;
        }
        public int getCVIndex(){
            return m_cvIdx;
        }

    }

    HashMap<String, ArrayList<_Edge4CV>> m_uidInteractionsMap = new HashMap<String, ArrayList<_Edge4CV>>();
    HashMap<String, ArrayList<_Edge4CV>> m_uidNonInteractionsMap = new HashMap<String, ArrayList<_Edge4CV>>();

    // Assign interactions to different folds for CV, try to balance different folds.
    public void assignCVIndex4Network(int kFold, int time){
        m_uidInteractionsMap.clear();
        m_uidNonInteractionsMap.clear();

        System.out.println("[Info]Start CV Index assignment for network....");

        ArrayList<Integer> interactions = new ArrayList<Integer>();
        ArrayList<Integer> nonInteractions = new ArrayList<Integer>();

        int orgTotal = 0, realTotal = 0;
        for(int i=0; i<m_users.size(); i++){
            _User ui = m_users.get(i);
            String uiId = ui.getUserID();
            String[] friends = ui.getFriends();

            interactions.clear();
            nonInteractions.clear();

            // ignore the users without any interactions
            if(friends != null && friends.length > 0) {
                if(!m_uidInteractionsMap.containsKey(uiId))
                    m_uidInteractionsMap.put(uiId, new ArrayList<_Edge4CV>());
                if(!m_uidNonInteractionsMap.containsKey(uiId))
                    m_uidNonInteractionsMap.put(uiId, new ArrayList<_Edge4CV>());

                orgTotal += friends.length;
                // construct the friend indexes
                for(String frd: friends){
                    int frdIdx = m_userIDIndex.get(frd);
                    if(frdIdx > i)
                        interactions.add(frdIdx);
                }

                for(int j=i+1; j<m_users.size(); j++){
                    if(!interactions.contains(j))
                        nonInteractions.add(j);
                }
                // sample masks for interactions: assign fold number to interactiosn
                int[] masks4Interactions = generateMasks(interactions.size(), kFold);
                // collect the interactions in the hashmap
                for(int m=0; m<interactions.size(); m++){
                    String ujId = m_users.get(interactions.get(m)).getUserID();

                    if(!m_uidInteractionsMap.containsKey(ujId))
                        m_uidInteractionsMap.put(ujId, new ArrayList<_Edge4CV>());
                    m_uidInteractionsMap.get(uiId).add(new _Edge4CV(ujId, masks4Interactions[m]));
                    m_uidInteractionsMap.get(ujId).add(new _Edge4CV(uiId, masks4Interactions[m]));
                }

                // sample non-interactions: select non-interactions for each fold, might be repetitive
                HashMap<Integer, HashSet<Integer>> foldNonInteractions = new HashMap<Integer, HashSet<Integer>>();
                for(int k=0; k<kFold; k++){
                    int number = time * interactions.size() / 5;
                    foldNonInteractions.put(k, sampleNonInteractions(nonInteractions, number));
                }
                // collect the non-interactions in the hashmap
                for(int k: foldNonInteractions.keySet()){
                    for(int ujIdx: foldNonInteractions.get(k)){
                        String ujId = m_users.get(ujIdx).getUserID();
                        if(!m_uidNonInteractionsMap.containsKey(ujId))
                            m_uidNonInteractionsMap.put(ujId, new ArrayList<_Edge4CV>());
                        m_uidNonInteractionsMap.get(uiId).add(new _Edge4CV(ujId, k));
                        m_uidNonInteractionsMap.get(ujId).add(new _Edge4CV(uiId, k));
                    }
                }
            }
        }
        System.out.println("Interaction user size: " + m_uidInteractionsMap.size());
        System.out.println("Non-interaction user size: " + m_uidNonInteractionsMap.size());

        for(String uid: m_uidInteractionsMap.keySet()){
            realTotal += m_uidInteractionsMap.get(uid).size();
        }
        System.out.format("Org Total: %d, real Total: %d\n", orgTotal, realTotal);
    }

    public void sanityCheck4CVIndex4Network(boolean interactionFlag){
        HashMap<String, ArrayList<_Edge4CV>> map = interactionFlag ? m_uidInteractionsMap: m_uidNonInteractionsMap;
        if(interactionFlag)
            System.out.println("=====Stat for users' interactions======");
        else
            System.out.println("=====Stat for users' non-interactions======");

        double total = 0, avg = 0;
        double[] stat = new double[5];

        // we only care about users who have interactions, thus use their idx for indexing
        int count = 0;
        for(String uid: m_uidInteractionsMap.keySet()){
            if(map.get(uid).size() == 0) {
                count++;
                continue;
            }

            for(_Edge4CV eg: map.get(uid)){
                total++;
                stat[eg.getCVIndex()]++;
            }
        }
        System.out.format("%d users don't have non-interactions!\n", count);
        avg = total / m_uidInteractionsMap.size();
        System.out.format("[Stat]Total user size: %d, total edge: %.1f, avg interaction/non-interaction size: %.3f\nEach fold's " +
                "interaction/non-interaction size is as follows:\n", m_uidInteractionsMap.size(), total, avg);
        for(double s: stat){
            System.out.print(s+"\t");
        }
        System.out.println("\n");
    }

    public void saveCVIndex4Network(String filename, boolean interactionFlag){
        HashMap<String, ArrayList<_Edge4CV>> map = interactionFlag ? m_uidInteractionsMap : m_uidNonInteractionsMap;
        try{
            // we only care about users who have interactions
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_uidInteractionsMap.keySet()){
                for(_Edge4CV eg: map.get(uid)){
                    writer.format("%s,%s,%d\n", uid, eg.getUserId(), eg.getCVIndex());
                }
            }
            writer.close();
            System.out.format("[Info]Finish writing %d users in %s.\n", m_uidInteractionsMap.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    // set masks for one users' all reviews for CV
    public int[] generateMasks(int len, int k){
        int[] masks = new int[len];

        int res = masks.length / k;
        int threshold = res * k;
        for(int i=0; i<masks.length; i++){
            if(i < threshold){
                masks[i] = i % k;
            } else{
                masks[i] = m_rand.nextInt(k);
            }
        }
        shuffle(masks);
        return masks;
    }

    public HashSet<Integer> sampleNonInteractions(ArrayList<Integer> nonInteractions, int nu){
        HashSet<Integer> sampledNonInteractions = new HashSet<Integer>();
        for(int i=0; i<nu; i++){
            int idx = m_rand.nextInt(nonInteractions.size());
            sampledNonInteractions.add(nonInteractions.get(idx));
        }
        return sampledNonInteractions;
    }

    public HashSet<String> sampleNonInteractionsWithUserIds(ArrayList<String> nonInteractions, int nu){
        HashSet<String> sampledNonInteractions = new HashSet<String>();
        for(int i=0; i<nu; i++){
            int idx = m_rand.nextInt(nonInteractions.size());
            sampledNonInteractions.add(nonInteractions.get(idx));
        }
        return sampledNonInteractions;
    }

    public void findUserWithMaxDocSize(){
        int max = -1;
        int count = 0;
        for(_User u: m_users){
            if(u.getReviewSize() > 1000) {
                System.out.println(u.getUserID());
                count++;
            }
            max = Math.max(max, u.getReviewSize());
        }
        System.out.println("Max doc size: " + max);
    }

    /****We need to output some files for running baselines
     * TADW, PLANE etc.
     * ****/
    public void printDocs4Plane(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User user: m_users){
                ArrayList<_SparseFeature[]> vectors = new ArrayList<>();
                for(_Review r: user.getReviews()){
                    vectors.add(r.getSparse());
                }
                _SparseFeature[] fvs = Utils.mergeSpVcts(vectors);
                for(_SparseFeature fv: fvs){
                    int index = fv.getIndex();
                    double val = fv.getValue();
                    for(int i=0; i<val; i++){
                        writer.write(index+" ");
                    }
                }
                writer.write("\n");
            }
            writer.close();
            System.out.println("Finish writing docs for PLANE!!");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void writeAggregatedUsers(String filename){
        try{
            int count = 0;
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User user: m_users){
                if(user.getUserID().equals("-dF9A2Q3L8C0d2ZyEIgDSQ"))
                    System.out.println("!!!!The user exists in the dataset!!!!");
                writer.write(user.getUserID()+"\n");
                for(_Review r: user.getReviews()){
                   writer.write(r.getSource()+" ");
                }
                writer.write("\n");
                count++;
            }
            writer.close();
            System.out.format("%d/%d users' data are writen in %s.\n", count, m_users.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // reserve docs for perplexity calculation, group users based on connectivity
    // e1 and e2 are thresholds for splitting users into light, medium and heavy users
    public void sampleUsers4ColdStart4Docs(String filename, int e1, int e2, int sampleSize){
        Random rand = new Random();
        ArrayList<String> light = new ArrayList<>();
        ArrayList<String> medium = new ArrayList<>();
        ArrayList<String> heavy = new ArrayList<>();

        // step 1: collect all the user ids in different groups
        for(_User user: m_users){
            if(user.getFriends() == null) continue;
            String userId = user.getUserID();
            int frdSize = user.getFriends().length;
            if(frdSize > e2){
                heavy.add(userId);
            } else if(frdSize > e1){
                medium.add(userId);
            } else
                light.add(userId);
        }
        // step 2: sample specified number of users from each group
        HashSet<String> sampledLight = sample(light, sampleSize);
        HashSet<String> sampledMedium = sample(medium, sampleSize);
        HashSet<String> sampledHeavey = sample(heavy, sampleSize);

        // step 3: save the sampled users and their documenets
        writeCVIndex4Docs(filename, sampledLight, sampledMedium, sampledHeavey);
    }

    // reserve edges for link prediction, group users based on document size
    // d1 and d2 are thresholds for splitting users into light, medium and heavy users
    public void sampleUsers4ColdStart4Edges(String dir, int d1, int d2, int sampleSize){
        Random rand = new Random();
        ArrayList<String> light = new ArrayList<>();
        ArrayList<String> medium = new ArrayList<>();
        ArrayList<String> heavy = new ArrayList<>();

        // step 1: collect all the user ids in different groups
        for(_User user: m_users){
            if(user.getReviews() == null) continue;
            String userId = user.getUserID();
            int rvwSize = user.getReviews().size();
            if(rvwSize > d2){
                heavy.add(userId);
            } else if(rvwSize > d1){
                medium.add(userId);
            } else
                light.add(userId);
        }
        // step 2: sample specified number of users from each group
        HashSet<String> sampledLight = sample(light, sampleSize);
        HashSet<String> sampledMedium = sample(medium, sampleSize);
        HashSet<String> sampledHeavy = sample(heavy, sampleSize);

        // step 3: since edges are symmetric, remove the associated edges
        removeSymmetricEdges(sampledLight);
        removeSymmetricEdges(sampledMedium);
        removeSymmetricEdges(sampledHeavy);

        // step 4: save the sampled users and their interactions
        writeCVIndex4Edges(dir+"_interactions.txt", sampledLight, sampledMedium, sampledHeavy);

        // step 5: sample non-interactions for different groups of users
        for(int time: new int[]{2, 3, 4, 5, 6, 7, 8}) {
            String filename = String.format("%s_noninteractions_time_%d_light.txt", dir, time);
            sampleNonInteractions4OneGroup(filename, sampledLight, time);
            filename = String.format("%s_noninteractions_time_%d_medium.txt", dir, time);
            sampleNonInteractions4OneGroup(filename, sampledMedium, time);
            filename = String.format("%s_noninteractions_time_%d_heavy.txt", dir, time);
            sampleNonInteractions4OneGroup(filename, sampledHeavy, time);
        }
    }

    public void sampleNonInteractions4OneGroup(String filename, HashSet<String> uids, int time){
        ArrayList<Integer> interactions = new ArrayList<Integer>();
        ArrayList<Integer> nonInteractions = new ArrayList<Integer>();

        HashMap<String, HashSet<Integer>> userNonInteractionMap = new HashMap<>();
        for(String uid: uids){

            int i = m_userIDIndex.get(uid);
            _User ui = m_users.get(i);
            interactions.clear();
            nonInteractions.clear();

            for(String frd: ui.getFriends()){
                interactions.add(m_userIDIndex.get(frd));
            }
            for(int j=0; j<m_users.size(); j++){
                if(i == j) continue;
                if(interactions.contains(j)) continue;
                nonInteractions.add(j);
            }

            int number = time * interactions.size();
            userNonInteractionMap.put(uid, sampleNonInteractions(nonInteractions, number));
        }

        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: userNonInteractionMap.keySet()){
                writer.write(uid + "\t");
                for(int nonIdx: userNonInteractionMap.get(uid)){
                    String nonId = m_users.get(nonIdx).getUserID();
                    writer.write(nonId + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' non-interactions are written in %s.\n", uids.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void removeSymmetricEdges(HashSet<String> uids){
        for(String uid: uids){
            _User ui = m_users.get(m_userIDIndex.get(uid));
            if(ui.getFriends() == null)
                System.out.println("The user does not have any friends!!");
            for(String frd: ui.getFriends()){
                _User uj = m_users.get(m_userIDIndex.get(frd));
                uj.removeOneFriend(uid);
            }
        }
    }

    public HashSet<String> sample(ArrayList<String> candidates, int size){
        Random rand = new Random();
        HashSet<String> sampledUsers = new HashSet<>();
        while(sampledUsers.size() < size){
            int index = rand.nextInt(candidates.size());
            _User user = m_users.get(m_userIDIndex.get(candidates.get(index)));
            if(user.getFriends() != null) {
                sampledUsers.add(candidates.get(index));
            }
        }
        return sampledUsers;
    }

    public void writeCVIndex4Docs(String filename, HashSet<String> light, HashSet<String> medium, HashSet<String> heavy){

        try{
            int[] stat = new int[4];
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_userIDIndex.keySet()){
                _User user = m_users.get(m_userIDIndex.get(uid));
                if(light.contains(uid)){
                    writeOneUserReviews(writer, user, 0, stat);
                } else if(medium.contains(uid)){
                    writeOneUserReviews(writer, user, 1, stat);
                } else if(heavy.contains(uid)){
                    writeOneUserReviews(writer, user, 2, stat);
                } else{
                    writeOneUserReviews(writer, user, 3, stat);
                }
            }
            writer.close();
            System.out.println("[Info]Finish writing cv index! Stat as follow:");
            for(int s: stat)
                System.out.print(s + "\t");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // write out the training interactions for users
    public void writeCVIndex4Edges(String filename, HashSet<String> light, HashSet<String> medium, HashSet<String> heavy){
        try{
            int count = 0;
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_networkMap.keySet()){
                _User user = m_users.get(m_userIDIndex.get(uid));
                if(user.getFriends() == null || user.getFriends().length == 0)
                    continue;
                if(!light.contains(uid) && !medium.contains(uid) && !heavy.contains(uid)){
                    count++;
                    writer.write(uid+"\t");
                    for(String frd: user.getFriends())
                        writer.write(frd + "\t");
                    writer.write("\n");
                }
            }
            writer.close();
            System.out.format("[stat]%d/%d users' interactions are written in filname.\n", count, m_networkMap.keySet().size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void writeOneUserReviews(PrintWriter writer, _User user, int mask, int[] stat){
        for(_Review r: user.getReviews()){
            writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), mask));
        }
        stat[mask]++;
    }

    public void calcDocStat4LightMediumHeavy(int k1, int k2){
        int light = 0, medium = 0, heavy = 0, max = 0;
        for(_User user: m_users) {
            int reviewSize = user.getReviews().size();
            max = Math.max(max, reviewSize);
            if(reviewSize > k2){
                heavy++;
            } else if(reviewSize > k1){
                medium++;
            } else
                light++;
        }
        System.out.format("[Stat]Dos-Light: %d, Medium: %d, Heavy: %d, max: %d.\n", light, medium, heavy, max);
    }

    public void calcEdgeStat4LightMediumHeavy(int k1, int k2){
        int light = 0, medium = 0, heavy = 0, max = 0;
        for(_User user: m_users) {
            if(user.getFriends() == null) continue;
            int frdSize = user.getFriends().length;
            max = Math.max(frdSize, max);
            if(frdSize > k2){
                heavy++;
            } else if(frdSize > k1){
                medium++;
            } else
                light++;
        }
        System.out.format("[Stat]Edges-Light: %d, Medium: %d, Heavy: %d, max: %d.\n", light, medium, heavy, max);
    }

    public void calcDataStat(){
        double avgDocLen = 0;
        double docSize = 0;
        for(_User u: m_users){
            for(_Review r: u.getReviews()){
                docSize++;
                avgDocLen += r.getTotalDocLength();
            }
        }
        avgDocLen /= docSize;
        System.out.format("[Info]Avg doc length: %.2f\n", avgDocLen);
    }

    HashMap<String, HashSet<String>> m_groups = new HashMap<>();

    public void initGroups(){
        m_groups.put("light", new HashSet<>());
        m_groups.put("medium", new HashSet<>());
        m_groups.put("heavy", new HashSet<>());
    }

    /******The codes are used in cold start setting*****/
    public void loadTestUsers(String prefix){

        initGroups();
        for(String group: m_groups.keySet()){
            try {
                HashSet<String> curGroup = m_groups.get(group);
                curGroup.clear();
                String filename = String.format("%s_%s.txt", prefix, group);
                File file = new File(filename);
                BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] strs = line.trim().split("\t");
                    String uid = strs[0];
                    curGroup.add(uid);
                }
                reader.close();
                System.out.format("[Info]%d user ids from group-%s are loaded !!\n", curGroup.size(), filename);
            } catch(IOException e){
                e.printStackTrace();
            }
        }
    }

    // print out the connections for each group of users
    public void printTestInteractions(String prefix) {
        for(String group: m_groups.keySet()){
            try{
                String filename = String.format("%s_%s.txt", prefix, group);
                PrintWriter writer = new PrintWriter(new File(filename));
                for(String uid: m_groups.get(group)){
                    writer.write(uid + "\t");
                    for(String frd: m_networkMap.get(uid)){
                        writer.write(frd+"\t");
                    }
                    writer.write("\n");
                }
                writer.close();
                System.out.format("[Info]Finish writing %d users' friends from group-%s!!\n", m_groups.get(group).size(), filename);
            } catch(IOException e){
                e.printStackTrace();
            }
        }
    }

    // analyze the restaurant information for recommendation
    ArrayList<_Review> m_selectedReviews = new ArrayList<>();
    HashMap<String, HashSet<String>> m_testInteractions = new HashMap<>();
    HashMap<String, HashSet<String>> m_testNonInteractions = new HashMap<>();

    public void selectReviews4Recommendation(){
        HashMap<String, ArrayList<_Review>> itemMap = new HashMap<>();
        for(_User u: m_users){
            for(_Review r: u.getReviews()){
                String itemId = r.getItemID();
                if(!itemMap.containsKey(itemId)) {
                    itemMap.put(itemId, new ArrayList<_Review>());
                }
                itemMap.get(itemId).add(r);
            }
        }
        for(String itemId: itemMap.keySet()){
            ArrayList<_Review> reviews = itemMap.get(itemId);
            if(reviews.size() > 5 && reviews.size() <= 10){
                Collections.sort(reviews, new Comparator<_Review>() {
                    @Override
                    public int compare(_Review o1, _Review o2) {
                        return (int)(o1.getTimeStamp() - o2.getTimeStamp());
                    }
                });
                String ui = reviews.get(0).getUserID();
                if(m_testInteractions.containsKey(ui))
                    continue;
                m_selectedReviews.add(reviews.get(0));
                m_testInteractions.put(ui, new HashSet<>());
                for(int i=0; i<reviews.size(); i++){
                    if(i != 0) {
                        // assign the rest as the testing data
                        reviews.get(i).setMask4CV(0);
                        m_testInteractions.get(ui).add(reviews.get(i).getUserID());
                    }
                }
            } else{
                for(_Review r: reviews){
                    r.setMask4CV(1);
                }
            }
        }
        System.out.format("Total item: %d\n", itemMap.size());
        System.out.format("%d reviews are selected for recommendation!", m_selectedReviews.size());
    }

    public void sampleNonInteractions(int time) {
        m_testNonInteractions.clear();
        HashSet<String> interactions = new HashSet<>();
        ArrayList<String> nonInteractions = new ArrayList<>();

        for (String ui : m_testInteractions.keySet()) {

            interactions = m_testInteractions.get(ui);
            nonInteractions.clear();

            for (int j = 0; j < m_users.size(); j++) {
                String uj = m_users.get(j).getUserID();
                if (uj.equals(ui)) continue;
                if (interactions.contains(uj)) continue;
                nonInteractions.add(uj);
            }

            int number = time * m_testInteractions.get(ui).size();
            m_testNonInteractions.put(ui, sampleNonInteractionsWithUserIds(nonInteractions, number));
        }
    }

    public void saveInteractions(String prefix){
        try{
            String filename = prefix + "Interactions4Recommendations_test.txt";
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_testInteractions.keySet()){
                writer.write(uid + "\t");
                for(String uj: m_testInteractions.get(uid)){
                    writer.write(uj + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' interactions are written in %s.\n", m_testInteractions.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void saveNonInteractions(String prefix, int time){
        try{
            String filename = String.format("%sNonInteractions_time_%d_Recommendations.txt", prefix, time);
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_testNonInteractions.keySet()){
                writer.write(uid + "\t");
                for(String uj: m_testNonInteractions.get(uid)){
                    writer.write(uj + "\t");
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Stat]%d users' non-interactions are written in %s.\n", m_testNonInteractions.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void printSelectedQuestionIds(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_Review r: m_selectedReviews){

                String uId = r.getUserID();
                writer.format("%s\t%d\n", uId, r.getID());
            }
            writer.close();
            System.out.format("Finish writing %d selected questions!\n", m_selectedReviews.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    //In the main function, we generate the data for recommendation for yelp
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {


        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "YelpNew"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        analyzer.selectReviews4Recommendation();

        String questionFile = String.format("%s/%s/AnswerRecommendation/%SelectedQuestions.txt", prefix, dataset, dataset);
        String cvIndexFile4Rec = String.format("%s/%s/AnswerRecommendation/%sCVIndex4Recommendation.txt", prefix, dataset, dataset);
        analyzer.saveCVIndex(cvIndexFile4Rec);

        String prefix4Rec = String.format("%s/%s/AnswerRecommendation/%s", prefix, dataset, dataset);
        analyzer.saveInteractions(prefix4Rec);
        analyzer.printSelectedQuestionIds(questionFile);

        for(int time: new int[]{5, 10}) {
            analyzer.sampleNonInteractions(time);
            analyzer.saveNonInteractions(prefix4Rec, time);
        }
    }
}
