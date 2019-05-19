package topicmodels.RoleEmbedding;

import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class RoleEmbeddingFixU extends RoleEmbeddingBaseline {
    public RoleEmbeddingFixU(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize) {
        super(m, L, nuIter, converge, alpha, beta, stepSize);
    }

    public void init(String filename){

        m_usersInput = new double[m_uIds.size()][m_dim];
        m_inputG = new double[m_uIds.size()][m_dim];
        loadUserEmbedding(filename);
//        for(double[] user: m_usersInput){
//            initOneVector(user);
//        }

        m_roles = new double[m_nuOfRoles][m_dim];
        m_rolesG = new double[m_nuOfRoles][m_dim];
        for(double[] role: m_roles){
            initOneVector(role);
        }
    }


    public void loadUserEmbedding(String filename){
        m_usersInput = new double[m_uIds.size()][m_dim];

        try {
            // load beta for the whole corpus first
            File userFile = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(userFile),
                    "UTF-8"));
            int count = 0;
            String line = reader.readLine(); // skip the first line
            while ((line = reader.readLine()) != null){
                count++;
                String[] strs = line.trim().split("\t");
                String uid = strs[0];
                int uIdx = m_uId2IndexMap.get(uid);
                double[] embedding = new double[strs.length - 1];
                for(int i=1; i<strs.length; i++)
                    embedding[i-1] = Double.valueOf(strs[i]);
                m_usersInput[uIdx] = embedding;
            }
            System.out.format("[Info]Finish loading %d user embeddings from %s\n", count, filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // update role vectors;
    public double updateRoleVectorsByElement(){

        System.out.println("Start optimizing role vectors...");
        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;

        double testLoss;
        ArrayList<Double> testLossArray = new ArrayList<>();

        do {
            fValue = 0; testLoss = 0;
            for (double[] g : m_rolesG) {
                Arrays.fill(g, 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;

                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each element of role embedding B_{gh}
                    for(int g=0; g<m_nuOfRoles; g++){
                        for(int h=0; h<m_dim; h++){
                            m_rolesG[g][h] += gTermOne * calcRoleGradientTermTwo(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                        }
                    }
                }
            }

            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateRoleVectorsWithSampledZeroEdgesByElement();
            }

            // calculate the loss on testing links
            for (int uiIdx : m_oneEdgesTest.keySet()) {
                for (int ujIdx : m_oneEdgesTest.get(uiIdx)) {
                    if (ujIdx <= uiIdx) continue;
                    affinity = calcAffinity(uiIdx, ujIdx);
                    testLoss += Math.log(sigmod(affinity));
                }
            }
            // calculate the loss on testing non-links
            for (int uiIdx : m_zeroEdgesTest.keySet()) {
                for(int ujIdx: m_zeroEdgesTest.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    affinity = calcAffinity(uiIdx, ujIdx);
                    testLoss += Math.log(sigmod(-affinity));

                }
            }

            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    m_rolesG[l][m] -= 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                    testLoss -= m_beta * m_roles[l][m] * m_roles[l][m];
                }
            }
            testLossArray.add(testLoss);

            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.01 * m_rolesG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        System.out.println("-------Loss on testing links--------");
        for(double v: testLossArray)
            System.out.println(v);
        return fValue;
    }
    @Override
    public void train(){
        System.out.println(toString());

        int iter = 0;
        double lastFunctionValue = -1.0;
        double currentFunctionValue;
        double converge;
        // iteratively update user vectors and role vectors until converge
        do {
            System.out.format(String.format("\n----------Start EM %d iteraction----------\n", iter));

            // update user vectors;
            currentFunctionValue = updateUserVectors();

            // update role vectors;
            currentFunctionValue += updateRoleVectors();

            if (iter++ > 0)
                converge = Math.abs((lastFunctionValue - currentFunctionValue) / lastFunctionValue);
            else
                converge = 1.0;

            lastFunctionValue = currentFunctionValue;

        } while (iter < m_numberOfIteration && converge > m_converge);
    }


    // update user vectors;
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;

        do{
            fValue = 0;

            // updates based on one edges
            for(int uiIdx: m_oneEdges.keySet()){
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));

                }
            }
            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateUserVectorsWithSampledZeroEdges();
            }

            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
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
            }
        }
        return fValue;
    }

    //The main function for general link pred
    public static void main(String[] args) {

        String dataset = "YelpNew"; //
        int fold = 0, dim = 10, nuOfRoles = 10, nuIter = 100, order = 1;

        String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);
        String oneEdgeTestFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_test.txt", dataset, fold);
        String zeroEdgeTestFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_test.txt", dataset, fold);

        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_user_embedding_order_%d_dim_%d_fold_%d_init.txt", order, dataset, order, dim, fold);
        String userEmbeddingOutputFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_user_embedding_order_%d_dim_%d_fold_%d.txt", order, dataset, order, dim, fold);

        String roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_embedding_fixU_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);

        double converge = 1e-6, alpha = 1, beta = 1, stepSize = 0.0001;
        RoleEmbeddingFixU roleBase = new RoleEmbeddingFixU(dim, nuOfRoles, nuIter, converge, alpha, beta, stepSize);

        roleBase.loadUsers(userFile);
        roleBase.init(userEmbeddingFile);

        if (order >= 1)
            roleBase.loadEdges(oneEdgeFile, 1);
        if (order >= 2)
            roleBase.generate2ndConnections();
        if (order >= 3)
            roleBase.generate3rdConnections();

        roleBase.loadEdges(zeroEdgeFile, 0); // load zero edges
        roleBase.loadEdges(oneEdgeTestFile, -1); // one edges for testing
        roleBase.loadEdges(zeroEdgeTestFile, -2);

        roleBase.train();
//        roleBase.printUserEmbedding(userEmbeddingOutputFile);
        roleBase.printRoleEmbedding(roleEmbeddingFile, roleBase.getRoleEmbeddings());
    }
}
