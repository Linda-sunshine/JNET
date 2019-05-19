package topicmodels.RoleEmbedding;

import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;

public class RoleEmbeddingMaxRoleDiff extends RoleEmbeddingFixU {

    public RoleEmbeddingMaxRoleDiff(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize) {
        super(m, L, nuIter, converge, alpha, beta, stepSize);
    }

    // update role vectors;
    public double updateRoleVectorsByElement(){

        System.out.println("Start optimizing role vectors...");
        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;

        do {
            fValue = 0;
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

            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    m_rolesG[l][m] -= 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                }
            }

            // add the difference between each pair of roles
            for(int g=0; g<m_roles.length; g++){
                for(int h=g+1; h<m_roles.length; h++){
                    fValue += m_beta * Utils.euclideanDistance(m_roles[g], m_roles[h]);
                    double[] tmp =  minus2Array(m_roles[g], m_roles[h]); // m_roles[g]-m_roles[h]
                    Utils.add2Array(m_rolesG[g], tmp, m_beta * 2); // gradient = gradient + beta * 2 * (m_roles[g]-m_roles[h])
                    Utils.add2Array(m_rolesG[h], tmp, -m_beta * 2); // gradient = gradient + beta * 2 * (m_roles[g]-m_roles[h])

                }
            }

            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.001 * m_rolesG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);

        return fValue;
    }

    public double[] minus2Array(double[] a, double[] b){
        if(a.length != b.length)
            System.out.println("Different dimension!!");
        double[] tmp = new double[a.length];
        for(int i=0; i<a.length; i++){
            tmp[i] = a[i] - b[i];
        }
        return tmp;
    }


    //The main function for general link pred
    public static void main(String[] args) {

        String dataset = "YelpNew"; //
        int fold = 0, dim = 10, nuOfRoles = 10, nuIter = 100, order = 1;

        String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);

        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_user_embedding_order_%d_dim_%d_fold_%d_init.txt", order, dataset, order, dim, fold);
        String roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_embedding_maxD_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, order, nuOfRoles, dim, fold);

        double converge = 1e-6, alpha = 1, beta = 5, stepSize = 0.001;
        RoleEmbeddingMaxRoleDiff roleBase = new RoleEmbeddingMaxRoleDiff(dim, nuOfRoles, nuIter, converge, alpha, beta, stepSize);

        roleBase.loadUsers(userFile);
        roleBase.init(userEmbeddingFile);

        if (order >= 1)
            roleBase.loadEdges(oneEdgeFile, 1);
        if (order >= 2)
            roleBase.generate2ndConnections();
        if (order >= 3)
            roleBase.generate3rdConnections();
        roleBase.loadEdges(zeroEdgeFile, 0); // load zero edges

        roleBase.train();
        roleBase.printRoleEmbedding(roleEmbeddingFile, roleBase.getRoleEmbeddings());
    }
}

