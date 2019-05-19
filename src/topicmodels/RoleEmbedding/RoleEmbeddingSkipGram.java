package topicmodels.RoleEmbedding;

import java.util.Arrays;

public class RoleEmbeddingSkipGram extends RoleEmbeddingBaseline {

    // m_roles: A; m_roleContext: B
    protected double[][] m_rolesContext, m_rolesContextG; // L*M, i.e., B in the derivation
    protected double m_gamma; // parameter for regularization of A

    public RoleEmbeddingSkipGram(int m, int L, int nuIter, double converge, double alpha, double beta, double gamma, double stepSize){
        super(m, L, nuIter, converge, alpha, beta, stepSize);
        m_gamma = gamma;
    }

    @Override
    public String toString() {
        return String.format("MultiRoleEmbedding_SkipGram[dim:%d, #Roles:%d, alpha:%.4f, beta:%.4f, gamma:%.4f, #Iter:%d]", m_dim,
                m_nuOfRoles, m_alpha, m_beta, m_gamma, m_numberOfIteration);

    }

    public void init(){
        super.init();
        m_rolesContext = new double[m_nuOfRoles][m_dim];
        m_rolesContextG = new double[m_nuOfRoles][m_dim];
        for(double[] role: m_rolesContext){
            initOneVector(role);
        }
    }

    // u_i^T B^T B u_j
    @Override
    public double calcAffinity(int i, int j){

        double res = 0;
        double[] ui = m_usersInput[i];
        double[] uj = m_usersInput[j];
        for(int p=0; p<m_dim; p++){
            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    res += uj[m] * m_rolesContext[l][m] * m_roles[l][p] * ui[p];
                }
            }
        }
        return res;
    }

    // update user vectors;
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;
        double[] ui, uj;

        do{
            fValue = 0;
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
                        m_inputG[uiIdx][g] += gTermOne * calcUserGradientTermTwoSource(g, uj);
                        m_inputG[ujIdx][g] += gTermOne * calcUserGradientTermTwoTarget(g, ui);
                    }
                }
            }

            // fix me!!!
            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateUserVectorsWithSampledZeroEdges();
            }

            // add the gradient from regularization
            for(int i=0; i<m_usersInput.length; i++){
                for(int m=0; m<m_dim; m++){
                    m_inputG[i][m] -= m_alpha * 2 * m_usersInput[i][m];
                }
            }
            // update the user vectors based on the gradients
            for(int i=0; i<m_usersInput.length; i++){
                for(int j=0; j<m_dim; j++){
                    fValue -= m_alpha * m_usersInput[i][j] * m_usersInput[i][j];
                    m_usersInput[i][j] += m_stepSize * m_inputG[i][j];
                }
            }
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

    // calculate the second part of the gradient for user vector
    public double calcUserGradientTermTwoSource(int g, double[] uj){
        double val = 0;
        for(int l=0; l<m_nuOfRoles; l++){
            for(int m=0; m<m_dim; m++){
                val += uj[m] * m_rolesContext[l][m] * m_roles[l][g];
            }
        }
        return val;
    }

    // calculate the second part of the gradient for user vector
    public double calcUserGradientTermTwoTarget(int g, double[] ui){
        double val = 0;
        for(int l=0; l<m_nuOfRoles; l++){
            for(int p=0; p<m_dim; p++){
                val += m_rolesContext[l][g] * m_roles[l][p] * ui[p];
            }
        }
        return val;
    }

    @Override
    public double updateRoleVectors() {
        System.out.println("Start optimizing role vectors...");
        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;
        do {
            fValue = 0;
            for (int i = 0; i < m_rolesG.length; i++) {
                Arrays.fill(m_rolesG[i], 0);
                Arrays.fill(m_rolesContextG[i], 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for (int ujIdx : m_oneEdges.get(uiIdx)) {
                    if (ujIdx <= uiIdx) continue;

                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each element of role embedding B_{gh}
                    for (int g = 0; g < m_nuOfRoles; g++) {
                        for (int h = 0; h < m_dim; h++) {
                            m_rolesG[g][h] += gTermOne * calcRoleGradientTermTwoSource(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                            m_rolesContextG[g][h] += gTermOne * calcRoleGradientTermTwoTarget(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                        }
                    }
                }
            }

            // updates of gradient from zero edges
            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateRoleVectorsWithSampledZeroEdges();
            }

            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    m_rolesG[l][m] -= 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                    m_rolesContextG[l][m] -= 2 * m_gamma * m_rolesContext[l][m];
                    fValue -= m_gamma * m_rolesContext[l][m] * m_rolesContext[l][m];
                }
            }

            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.01 * m_rolesG[l][m];
                    m_rolesContext[l][m] += m_stepSize * 0.01 * m_rolesContextG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);


        } while (iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

    public double updateRoleVectorsWithSampledZeroEdges(){
        double affinity, gTermOne, fValueZero = 0;
        // updates of gradient from zero edges
        for (int uiIdx : m_zeroEdges.keySet()) {
            for(int ujIdx: m_zeroEdges.get(uiIdx)){
                if(ujIdx <= uiIdx) continue;
                affinity = calcAffinity(uiIdx, ujIdx);
                fValueZero += Math.log(sigmod(-affinity));
                gTermOne = sigmod(affinity);
                // each element of role embedding B_{gh}
                for(int g=0; g<m_nuOfRoles; g++){
                    for(int h=0; h<m_dim; h++){
                        m_rolesG[g][h] -= gTermOne * calcRoleGradientTermTwoSource(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                        m_rolesContextG[g][h] -= gTermOne * calcRoleGradientTermTwoTarget(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                    }
                }
            }
        }
        return fValueZero;
    }

    // calculate the second part of the gradient for user vector
    public double calcRoleGradientTermTwoTarget(int g, int h, double[] ui, double[] uj){
        double val = 0;
        for(int p=0; p<m_dim; p++){
            val += uj[h] * m_roles[g][p] * ui[p];
        }
        return val;
    }

    public double calcRoleGradientTermTwoSource(int g, int h, double[] ui, double[] uj){
        double val = 0;
        for(int m=0; m<m_dim; m++){
            val += uj[m] * m_rolesContext[g][m] * ui[h];
        }
        return val;
    }


    public double[][] getContextRoleEmbeddings(){
        return m_rolesContext;
    }
    //The main function for general link pred
    public static void main(String[] args){

        String dataset = "FB"; // "release-youtube"
        int fold = 0, dim = 10, nuOfRoles = 10, nuIter = 100, order = 1;

        String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);
        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_multirole_skipgram_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);
        String roleSourceEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_source_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);
        String roleTargetEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_target_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);

        double converge = 1e-6, alpha = 0.5, beta = 0.5, gamma = 0.5, stepSize = 0.001;
        RoleEmbeddingSkipGram roleSkipGram = new RoleEmbeddingSkipGram(dim, nuOfRoles, nuIter, converge, alpha, beta, gamma, stepSize);

        roleSkipGram.loadUsers(userFile);
        if(order >= 1)
            roleSkipGram.loadEdges(oneEdgeFile, 1);
        if(order >= 2)
            roleSkipGram.generate2ndConnections();
        if(order >= 3)
            roleSkipGram.generate3rdConnections();
        roleSkipGram.loadEdges(zeroEdgeFile, 0); // load zero edges

        roleSkipGram.train();
        roleSkipGram.printUserEmbedding(userEmbeddingFile);
        roleSkipGram.printRoleEmbedding(roleSourceEmbeddingFile, roleSkipGram.getRoleEmbeddings());
        roleSkipGram.printRoleEmbedding(roleTargetEmbeddingFile, roleSkipGram.getContextRoleEmbeddings());
    }

}
