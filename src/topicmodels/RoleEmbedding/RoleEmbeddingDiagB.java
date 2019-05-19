package topicmodels.RoleEmbedding;

import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;

public class RoleEmbeddingDiagB extends RoleEmbeddingFixU {

    public RoleEmbeddingDiagB(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize) {
        super(m, L, nuIter, converge, alpha, beta, stepSize);
    }

    public void init(String filename){
        m_usersInput = new double[m_uIds.size()][m_dim];
        m_inputG = new double[m_uIds.size()][m_dim];
        loadUserEmbedding(filename);

//        for(double[] user: m_usersInput){
//            initOneVector(user);
//        }
        m_roles = new double[m_dim][m_dim];
        m_rolesG = new double[m_dim][m_dim];
        // fix role to be identity matrix
        for(int i=0; i<m_roles.length; i++)
            m_roles[i][i] = Math.random();
    }

    // u_i^T B u_j
    @Override
    public double calcAffinity(int i, int j){

        double res = 0;
        double[] ui = m_usersInput[i];
        double[] uj = m_usersInput[j];
        for(int p=0; p<m_dim; p++){
            res += ui[p] * m_roles[p][p] * uj[p];
        }
        return res;
    }

    @Override
    // calculate the second part of the gradient for user vector
    public double calcUserGradientTermTwo(int g, double[] uj){
        return m_roles[g][g] * uj[g];
    }


    // update role vectors;
    public double updateRoleVectorsByElement(){

        System.out.println("Start optimizing role vectors...");
        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;

        double testLoss, testLossBase;
        ArrayList<Double> testLossArray = new ArrayList<>();

        do {
            fValue = 0; testLoss = 0;
            for(double[] g: m_rolesG)
                Arrays.fill(g, 0);

            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;

                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));
                    // each element of role embedding B_{gh}
                    for(int g=0; g<m_dim; g++){
                        m_rolesG[g][g] += sigmod(-affinity) * m_usersInput[uiIdx][g] * m_usersInput[ujIdx][g];

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

            for(int l=0; l<m_dim; l++){
                m_rolesG[l][l] -= 2 * m_beta * m_roles[l][l];
                fValue -= m_beta * m_roles[l][l] * m_roles[l][l];
                testLoss -= m_beta * m_roles[l][l] * m_roles[l][l];
            }
            testLossArray.add(testLoss);

            // update the role vectors based on the gradients
            for(int l=0; l<m_dim; l++){
                m_roles[l][l] += m_stepSize * m_rolesG[l][l];

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

    public double updateRoleVectorsWithSampledZeroEdgesByElement(){
        double affinity, gTermOne, fValueZero = 0;
        // updates of gradient from zero edges
        for (int uiIdx : m_zeroEdges.keySet()) {
            for(int ujIdx: m_zeroEdges.get(uiIdx)){
                if(ujIdx <= uiIdx) continue;
                affinity = calcAffinity(uiIdx, ujIdx);
                fValueZero += Math.log(sigmod(-affinity));
                // each element of role embedding B_{gh}
                for(int g=0; g<m_dim; g++){
                    m_rolesG[g][g] -= sigmod(affinity) * m_usersInput[uiIdx][g] * m_usersInput[ujIdx][g];
                }
            }
        }
        return fValueZero;
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
        String userEmbeddingOutputFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_multirole_diagB_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d_rand.txt", order, dataset, order, nuOfRoles, dim, fold);
        String roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_embedding_diagB_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);

        double converge = 1e-6, alpha = 1, beta = 1, stepSize = 0.00001;
        RoleEmbeddingDiagB roleBase = new RoleEmbeddingDiagB(dim, nuOfRoles, nuIter, converge, alpha, beta, stepSize);

        roleBase.loadUsers(userFile);
        roleBase.init(userEmbeddingFile);

        if (order >= 1)
            roleBase.loadEdges(oneEdgeFile, 1);
        if (order >= 2)
            roleBase.generate2ndConnections();
        if (order >= 3)
            roleBase.generate3rdConnections();
        roleBase.loadEdges(zeroEdgeFile, 0); // load zero edges
        roleBase.loadEdges(oneEdgeTestFile, -1);
        roleBase.loadEdges(zeroEdgeTestFile, -2);

//        roleBase.sampleZeroEdges();
//        roleBase.saveZeroEdges(zeroEdgeFile);

        roleBase.train();
//        roleBase.printUserEmbedding(userEmbeddingOutputFile);
        roleBase.printRoleEmbedding(roleEmbeddingFile, roleBase.getRoleEmbeddings());
    }

}
