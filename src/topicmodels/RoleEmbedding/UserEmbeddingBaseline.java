package topicmodels.RoleEmbedding;

import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The modeling of user embedding with ui * uj as the affinity
 * The algorithm does not distinguish the input and output of users
 */


public class UserEmbeddingBaseline {

    protected double m_converge, m_alpha, m_stepSize; //parameter for regularization
    protected int m_dim, m_numberOfIteration;
    protected double[][] m_usersInput; // U*M

    protected double[][] m_inputG; // the gradient for the update
    protected ArrayList<String> m_uIds;
    protected HashMap<String, Integer> m_uId2IndexMap;
    protected HashMap<Integer, HashSet<Integer>> m_oneEdges;
    protected HashMap<Integer, HashSet<Integer>> m_zeroEdges;
    protected HashMap<Integer, HashSet<Integer>> m_oneEdgesTest;
    protected HashMap<Integer, HashSet<Integer>> m_zeroEdgesTest;

    // default is l2 normalization of user vectors
    protected boolean m_L1 = false;

    public UserEmbeddingBaseline(int m, int nuIter, double converge, double alpha, double stepSize){
        m_dim = m;
        m_numberOfIteration = nuIter;
        m_converge = converge;
        m_alpha = alpha;
        m_stepSize = stepSize;

        m_uIds = new ArrayList<>();
        m_uId2IndexMap = new HashMap<>();
        m_oneEdges = new HashMap<>();
        m_zeroEdges = new HashMap<>();
        m_oneEdgesTest = new HashMap<>();
        m_zeroEdgesTest = new HashMap<>();
    }

    @Override
    public String toString() {
        return String.format("UserEmbedding_Baseline[dim:%d, alpha:%.4f, #Iter:%d]", m_dim, m_alpha, m_numberOfIteration);
    }

    public void setL1Regularization(boolean b){
        m_L1 = b;
        if(m_L1)
            System.out.println("[Info]L1 regularization over user vectors!");
        else
            System.out.println("[Info]L2 regularization over user vectors!");
    }

    // load user ids from file
    public void loadUsers(String filename){
        try {
            // load beta for the whole corpus first
            File userFile = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(userFile),
                    "UTF-8"));
            String line;
            int count = 0;
            while ((line = reader.readLine()) != null){
                // start reading one user's id
                String uid = line.trim();
                m_uIds.add(uid);
                m_uId2IndexMap.put(uid, count++);
            }
            System.out.format("[Info]Finish loading %d user ids from %s\n", m_uIds.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // load connections/nonconnections from files
    public void loadEdges(String filename, int eij){

        HashMap<Integer, HashSet<Integer>> edgeMap;
        if(eij == 1 )
            edgeMap = m_oneEdges;
        else if(eij == 0)
            edgeMap = m_zeroEdges;
        else if(eij == -1)
            edgeMap = m_oneEdgesTest;
        else
            edgeMap = m_zeroEdgesTest;
        try {
            // load beta for the whole corpus first
            File linkFile = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(linkFile),
                    "UTF-8"));
            String line, uiId, ujId, strs[];
            int uiIdx, ujIdx, count = 0;
            while ((line = reader.readLine()) != null){
                // start reading one user's id
                strs = line.trim().split("\\s+");
                if(strs.length < 2){
                    System.out.println("Invalid pair!");
                    continue;
                }
                uiId = strs[0];
                uiIdx = m_uId2IndexMap.get(uiId);

                for(int j=1; j<strs.length; j++) {
                    ujId = strs[j];
                    ujIdx = m_uId2IndexMap.get(ujId);

                    if (!m_uId2IndexMap.containsKey(uiId)) {
                        System.out.println("The user does not exist in the user set!");
                        continue;
                    }
                    if (!edgeMap.containsKey(uiIdx)) {
                        edgeMap.put(uiIdx, new HashSet<Integer>());
                    }
                    edgeMap.get(uiIdx).add(ujIdx);
                    count++;
                    if (count % 10000 == 0)
                        System.out.print(".");
                    if (count % 1000000 == 0)
                        System.out.println();
                }
            }
            double avg = 0;
            HashMap<Integer, HashSet<Integer>> map = eij == 1 ? m_oneEdges : m_zeroEdges;
            for(int ui: map.keySet()){
                avg += map.get(ui).size();
            }
            avg /= map.size();
            System.out.format("\n[Info]Finish loading %d edges of %d users' %d links (avg: %.4f), from %s\n", eij, edgeMap.size(),
                    count, avg, filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeUserIds(String input, String output){
        try {
            // load beta for the whole corpus first
            HashSet<String> uids = new HashSet<String>();
            File file = new File(input);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file),
                    "UTF-8"));
            String line, strs[];
            while ((line = reader.readLine()) != null){
                // start reading one user's id
                strs = line.trim().split("\\s+");
                if(strs.length < 2){
                    System.out.println("Invalid pair!");
                    continue;
                }
                uids.add(strs[0]);
                uids.add(strs[1]);
            }
            reader.close();
            System.out.format("Finish loading %d user ids!!", uids.size());
            PrintWriter writer = new PrintWriter(new File(output));
            for(String uid: uids)
                writer.write(uid+"\n");
            writer.close();
            System.out.format("Finish writing %d user ids!!", uids.size());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    HashMap<Integer, HashSet<Integer>> m_oneEdge2ndOrder = new HashMap<>();
    public void generate2ndConnections(){
        // for one user
        for(int uiId: m_oneEdges.keySet()){
            // add friends' friends
            if(!m_oneEdge2ndOrder.containsKey(uiId))
                m_oneEdge2ndOrder.put(uiId, new HashSet<Integer>());
            for(int ujId: m_oneEdges.get(uiId)){
                m_oneEdge2ndOrder.get(uiId).add(ujId);
                if(!m_oneEdge2ndOrder.containsKey(ujId))
                    m_oneEdge2ndOrder.put(ujId, new HashSet<Integer>());
                m_oneEdge2ndOrder.get(ujId).add(uiId);
                for(int ujFrdId: m_oneEdges.get(ujId)){
                    m_oneEdge2ndOrder.get(uiId).add(ujFrdId);
                    if(!m_oneEdge2ndOrder.containsKey(ujFrdId))
                        m_oneEdge2ndOrder.put(ujFrdId, new HashSet<Integer>());
                    m_oneEdge2ndOrder.get(ujFrdId).add(uiId);

                }
            }
        }
        m_oneEdges = m_oneEdge2ndOrder;
        double avg = 0;
        for(int ui: m_oneEdges.keySet()){
            avg += m_oneEdges.get(ui).size();
        }
        avg /= m_oneEdges.size();
        System.out.format("Finish generating 2nd order connections, avg connection: %.4f.\n", avg);
    }

    HashMap<Integer, HashSet<Integer>> m_oneEdge3rdOrder = new HashMap<>();
    public void generate3rdConnections(){
        // for one user
        int count = 0;
        for(int uiId: m_oneEdges.keySet()){
            count++;
            if(count % 200 == 0)
                System.out.print('.');
            // add friends' friends
            if(!m_oneEdge3rdOrder.containsKey(uiId))
                m_oneEdge3rdOrder.put(uiId, new HashSet<Integer>());
            for(int ujId: m_oneEdges.get(uiId)){
                m_oneEdge3rdOrder.get(uiId).add(ujId);
                if(!m_oneEdge3rdOrder.containsKey(ujId))
                    m_oneEdge3rdOrder.put(ujId, new HashSet<Integer>());
                m_oneEdge3rdOrder.get(ujId).add(uiId);
                for(int ujFrdId: m_oneEdges.get(ujId)){
                    m_oneEdge3rdOrder.get(uiId).add(ujFrdId);
                    if(!m_oneEdge3rdOrder.containsKey(ujFrdId))
                        m_oneEdge3rdOrder.put(ujFrdId, new HashSet<Integer>());
                    m_oneEdge3rdOrder.get(ujFrdId).add(uiId);

                }
            }
        }
        m_oneEdges = m_oneEdge3rdOrder;
        double avg = 0;
        for(int ui: m_oneEdges.keySet()){
            avg += m_oneEdges.get(ui).size();
        }
        avg /= m_oneEdges.size();
        System.out.format("Finish generating 3rd order connections, avg connection: %.4f.\n", avg);
    }


    public void sampleZeroEdges() {
        for(int i=0; i<m_uIds.size(); i++){
            if(i % 10000 == 0)
                System.out.print(".");
            if(i % 1000000 == 0)
                System.out.println();
            String uiId = m_uIds.get(i);
            int uiIdx = m_uId2IndexMap.get(uiId);
            if(!m_zeroEdges.containsKey(uiIdx)){
                m_zeroEdges.put(uiIdx, new HashSet<Integer>());
            }
            HashSet<Integer> zeroEdges = m_zeroEdges.get(uiIdx);
            HashSet<Integer> oneEdges = m_oneEdges.containsKey(uiIdx) ? m_oneEdges.get(uiIdx) : null;
            int number = m_oneEdges.containsKey(uiIdx) ? m_oneEdges.get(uiIdx).size() * 3 : 0;

            while(zeroEdges.size() < number) {
                String ujId = m_uIds.get((int) (Math.random() * m_uIds.size()));
                int ujIdx = m_uId2IndexMap.get(ujId);
                if (oneEdges == null || !oneEdges.contains(ujIdx)) {
                    zeroEdges.add(ujIdx);
                    if(!m_zeroEdges.containsKey(ujIdx))
                        m_zeroEdges.put(ujIdx, new HashSet<Integer>());
                    m_zeroEdges.get(ujIdx).add(uiIdx);
                }
            }

        }
        double[] avg = new double[2];
        for(int uid: m_zeroEdges.keySet()){
            avg[0] += m_zeroEdges.get(uid).size();
        }
        for(int uid: m_oneEdges.keySet()){
            avg[1] += m_oneEdges.get(uid).size();
        }
        avg[0] /= m_zeroEdges.size();
        avg[1] /= m_oneEdges.size();
        System.out.format("avg one edge: %.2f, avg zero edge: %.2f.\n", avg[1], avg[0]);
    }
    public void saveZeroEdges(String filename){

        try{
            int count = 0;
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int uiIdx: m_zeroEdges.keySet()){
                String uiId = m_uIds.get(uiIdx);
                HashSet<Integer> zeroEdges = m_zeroEdges.get(uiIdx);
                for(int ujIdx: zeroEdges){
                    writer.format("%s\t%s\n", uiId, m_uIds.get(ujIdx));
                    count++;
                }
            }
            writer.close();
            System.out.format("Finish writing %d zero edges.\n", count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void init(){
        m_usersInput = new double[m_uIds.size()][m_dim];
        m_inputG = new double[m_uIds.size()][m_dim];

        for(double[] user: m_usersInput){
            initOneVector(user);
        }
    }

    // initialize each user vector
    public void initOneVector(double[] vct){
        for(int i=0; i<vct.length; i++){
            vct[i] = Math.random();
        }
        Utils.normalize(vct);
    }

    // update user vectors;
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;
        double[] ui, uj;

//        double testLoss = 0;
//        ArrayList<Double> testLossArray = new ArrayList<>();

        do{
            fValue = 0;
//            testLoss = 0;
            for(double[] g: m_inputG){
                Arrays.fill(g, 0);
            }
            // updates based on one edges
            for(int uiIdx: m_oneEdges.keySet()){
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    // for each edge
                    ui = m_usersInput[uiIdx];
                    uj = m_usersInput[ujIdx];
                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        m_inputG[uiIdx][g] += gTermOne * calcUserGradientTermTwo(g, uj);
                        m_inputG[ujIdx][g] += gTermOne * calcUserGradientTermTwo(g, ui);
                    }
                }
            }
            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateUserVectorsWithSampledZeroEdges();
            }

            // add the gradient from regularization
            for(int i=0; i<m_usersInput.length; i++){
                for(int m=0; m<m_dim; m++){
                    if(m_L1){
                        if(m_usersInput[i][m] > 0)
                            m_inputG[i][m] += m_alpha;
                        else if(m_usersInput[i][m] < 0)
                            m_inputG[i][m] -= m_alpha;
                        else {
                            // if it is 0, map it to one value in the range [-1,1]
                            // Math.random * 2 - 1
                            m_inputG[i][m] -= m_alpha * (Math.random() * 2 - 1);
                            System.err.println("[error]Zero point reached in L1!!");
                        }
                    }
                    else
                        m_inputG[i][m] -= m_alpha * 2 * m_usersInput[i][m];
                }
            }
            // update the user vectors based on the gradients
            for(int i=0; i<m_usersInput.length; i++){
                for(int j=0; j<m_dim; j++){
                    if(m_L1)
                        fValue -= m_alpha * Math.abs(m_usersInput[i][j]);
                    else
                        fValue -= m_alpha * m_usersInput[i][j] * m_usersInput[i][j];
                    m_usersInput[i][j] += m_stepSize * m_inputG[i][j];
                }
            }

//            for (int uiIdx : m_oneEdgesTest.keySet()) {
//                for (int ujIdx : m_oneEdgesTest.get(uiIdx)) {
//                    if (ujIdx <= uiIdx) continue;
//                    affinity = calcAffinity(uiIdx, ujIdx);
//                    testLoss += Math.log(sigmod(affinity));
//                }
//            }
//            // calculate the loss on testing non-links
//            for (int uiIdx : m_zeroEdgesTest.keySet()) {
//                for(int ujIdx: m_zeroEdgesTest.get(uiIdx)){
//                    if(ujIdx <= uiIdx) continue;
//                    affinity = calcAffinity(uiIdx, ujIdx);
//                    testLoss += Math.log(sigmod(-affinity));
//                }
//            }
//            testLossArray.add(testLoss);
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
//        System.out.println("-------Loss on testing links--------");
//        for(double v: testLossArray)
//            System.out.println(v);
        return fValue;
    }

    public double calcAffinity(int i, int j){
        return Utils.dotProduct(m_usersInput[i], m_usersInput[j]);
    }

    // if sampled zero edges are load, user sampled zero edges for update
    public double updateUserVectorsWithSampledZeroEdges(){
        double fValue = 0, affinity, gTermOne, ui[], uj[];
        for(int uiIdx: m_zeroEdges.keySet()){
            for(int ujIdx: m_zeroEdges.get(uiIdx)){
                if(ujIdx <= uiIdx) continue;
                // for each edge
                ui = m_usersInput[uiIdx];
                uj = m_usersInput[ujIdx];
                affinity = calcAffinity(uiIdx, ujIdx);
                fValue += Math.log(sigmod(-affinity));
                gTermOne = sigmod(affinity);
                // each dimension of user vectors ui and uj
                for(int g=0; g<m_dim; g++){
                    m_inputG[uiIdx][g] -= gTermOne * calcUserGradientTermTwo(g, uj);
                    m_inputG[ujIdx][g] -= gTermOne * calcUserGradientTermTwo(g, ui);
                }
            }
        }
        return fValue;
    }

    public double calcUserGradientTermTwo(int g, double[] uj){
        return uj[g];
    }

    public void train(){

        System.out.println(toString());

        init();
        int iter = 0;
        double lastFunctionValue = -1.0;
        double currentFunctionValue;
        double converge;
        // iteratively update user vectors and role vectors until converge
        do {
            System.out.format(String.format("\n----------Start EM %d iteraction----------\n", iter));

            // update user vectors;
            currentFunctionValue = updateUserVectors();

            if (iter++ > 0)
                converge = Math.abs((lastFunctionValue - currentFunctionValue) / lastFunctionValue);
            else
                converge = 1.0;

            lastFunctionValue = currentFunctionValue;

        } while (iter < m_numberOfIteration && converge > m_converge);
    }


    public double[] getOneColumn(double[][] mtx, int j){
        double[] col = new double[mtx.length];
        for(int i=0; i<mtx.length; i++){
            col[i] = mtx[i][j];
        }
        return col;
    }

    public double sigmod(double v){
        return 1/(1 + Math.exp(-v));
    }

    // print out learned user embedding
    public void printUserEmbedding(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", m_usersInput.length, m_dim);
            for (int i = 0; i < m_usersInput.length; i++) {
                writer.format("%s\t", m_uIds.get(i));
                for (double v : m_usersInput[i]) {
                    writer.format("%.4f\t", v);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("Finish writing %d user embeddings!", m_usersInput.length);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void preprocessYelpData(String filename, String uidFilename, String linkFilename){
        HashSet<String> uids = new HashSet<>();
        HashSet<String> ujds = new HashSet<>();
        int nuOfLinks = 0;
        try {
            // load beta for the whole corpus first
            File linkFile = new File(filename);

            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(linkFile),
                    "UTF-8"));
            PrintWriter idWriter = new PrintWriter(new File(uidFilename));
            PrintWriter linkWriter = new PrintWriter(new File(linkFilename));
            String line, strs[];
            while ((line = reader.readLine()) != null) {
                // start reading one user's id
                strs = line.trim().split("\\s+");
                String uid = strs[0];
                idWriter.write(uid + "\n");
                for (int i = 1; i < strs.length; i++) {
                    nuOfLinks++;
                    linkWriter.format("%s\t%s\n", uid, strs[i]);
                }
            }
            idWriter.close();
            linkWriter.close();
            System.out.format("\n[Info]Number of links: %d\n", nuOfLinks);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    HashMap<Integer, Integer> m_circles;

    public void loadCircles(String filename){
        try {
            m_circles = new HashMap<>();
            // load beta for the whole corpus first
            HashSet<Integer> uids = new HashSet<>();
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file),
                    "UTF-8"));
            String line, strs[];
            int nuOfCircles = 0;
            while ((line = reader.readLine()) != null){
                // start reading one user's id
                strs = line.trim().split("\\s+");

                if(strs.length < 2){
                    System.out.println("Invalid pair!");
                    continue;
                }
                for(int i=1; i<strs.length; i++){
                    m_circles.put(Integer.valueOf(strs[i]), nuOfCircles);
                }
                nuOfCircles++;
            }
            reader.close();
            System.out.format("Finish loading %d circles and %d user ids!!", nuOfCircles, uids.size());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void assignCircleIndex(String input, String output){
        try {
            File file = new File(input);
            ArrayList<Integer> circleIndexes = new ArrayList<>();
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file),
                    "UTF-8"));
            String line, strs[];
            while ((line = reader.readLine()) != null){
                strs = line.trim().split("\\s+");
                if(strs.length <= 2){
                    continue;
                } else{
                    if(m_circles.containsKey(Integer.valueOf(strs[0])))
                        circleIndexes.add(m_circles.get(Integer.valueOf(strs[0])));
                    else
                        circleIndexes.add(-1);
                }
            }
            System.out.format("%d users's circles are collected!", circleIndexes.size());
            reader.close();
            PrintWriter writer = new PrintWriter(new File(output));
            for(int cIndex: circleIndexes)
                writer.write(cIndex + "\t");
            System.out.format("Finish writing %d users's circles!", circleIndexes.size());
            writer.close();

        } catch (IOException e){
            e.printStackTrace();
        }
    }
    //The main function for general link pred
    public static void main(String[] args){

        String dataset = "YelpNew"; // "release-youtube"
        int fold = 0, nuIter = 500, order = 1;

        for(int m: new int[]{10}){
            String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
            String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
            String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);
            String oneEdgeTestFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_test.txt", dataset, fold);
            String zeroEdgeTestFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_test.txt", dataset, fold);

            String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_user_embedding_order_%d_dim_%d_fold_%d.txt", order, dataset, order, m, fold);

            double converge = 1e-6, alpha = 1, stepSize = 0.001;
            UserEmbeddingBaseline base = new UserEmbeddingBaseline(m, nuIter, converge, alpha, stepSize);
            String circleFile = String.format("./data/RoleEmbedding/%sCircles.txt", dataset);
            String userCircleIndexFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_user_circle_index.txt", dataset);

//          base.loadCircles(circleFile);
//          base.writeUserIds(oneEdgeFile, userFile);

            base.loadUsers(userFile);
            if(order >= 1)
                base.loadEdges(oneEdgeFile, 1);
            if(order >= 2)
                base.generate2ndConnections();
            if(order >= 3)
                base.generate3rdConnections();

            base.loadEdges(zeroEdgeFile, 0); // load zero edges
//            base.loadEdges(oneEdgeTestFile, -1);
//            base.loadEdges(zeroEdgeTestFile, -2);

//          base.sampleZeroEdges();
//          base.saveZeroEdges(zeroEdgeFile);
//
//            base.setL1Regularization(true);
            base.train();
            base.printUserEmbedding(userEmbeddingFile);

//          base.loadCircles(circleFile);
//          base.assignCircleIndex(userEmbeddingFile, userCircleIndexFile);
        }
    }
}
