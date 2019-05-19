package topicmodels.RoleEmbedding;

import java.util.ArrayList;
import java.util.Arrays;

public class RoleEmbeddingFixB extends RoleEmbeddingBaseline {

    public RoleEmbeddingFixB(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize) {
        super(m, L, nuIter, converge, alpha, beta, stepSize);
    }

    public void init(){
        m_usersInput = new double[m_uIds.size()][m_dim];
        m_inputG = new double[m_uIds.size()][m_dim];
        for(double[] user: m_usersInput){
            initOneVector(user);
        }
        m_roles = new double[m_nuOfRoles][m_dim];
        m_rolesG = new double[m_nuOfRoles][m_dim];
        // fix role to be identity matrix
        for(int i=0; i<m_roles.length; i++)
            m_roles[i][i] = 1;
    }

    // u_i^T B^T B u_j
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
        double[] ui, uj;

        double testLoss = 0;
        ArrayList<Double> testLossArray = new ArrayList<>();

        do{
            fValue = 0;
            testLoss = 0;
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
                        else
                            System.err.println("[error]Zero point reached in L1!!");
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
            for(int m=0; m<m_dim; m++){
                testLoss -= m_beta * m_roles[m][m] * m_roles[m][m];
            }
            testLossArray.add(testLoss);
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        System.out.println("-------Loss on testing links--------");
        for(double v: testLossArray)
            System.out.println(v);
        return fValue;
    }


    // update role vectors;
    public double updateRoleVectors(){

        System.out.println("Start optimizing role vectors...");
        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;

        do {
            fValue = 0;

            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;

                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));

                }
            }

            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateRoleVectorsWithSampledZeroEdgesByElement();
            }

//            for(int l=0; l<m_nuOfRoles; l++){
//                for(int m=0; m<m_dim; m++){
//                    m_rolesG[l][m] -= 2 * m_beta * m_roles[l][m];
//                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
//                }
//            }

            diff = fValue - lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
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

        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_multirole_fixB_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);
        String roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_embedding_fixB_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);

        double converge = 1e-6, alpha = 1, beta = 0.5, stepSize = 0.02;
        RoleEmbeddingFixB roleBase = new RoleEmbeddingFixB(dim, nuOfRoles, nuIter, converge, alpha, beta, stepSize);

        roleBase.loadUsers(userFile);
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
        roleBase.printUserEmbedding(userEmbeddingFile);
        roleBase.printRoleEmbedding(roleEmbeddingFile, roleBase.getRoleEmbeddings());
    }
}
