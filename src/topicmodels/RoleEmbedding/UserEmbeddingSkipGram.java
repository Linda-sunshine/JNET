package topicmodels.RoleEmbedding;


import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The modeling of user embedding with uj * vi as the affinity
 * The algorithm distinguish the input and output of users:
 * uj is the output matrix and ui is the input matrix
 */

public class UserEmbeddingSkipGram extends UserEmbeddingBaseline {
    protected double[][] m_usersOutput; // two sets of representations of users u_i
    protected double[][] m_outputG; // the gradient for the outputs of users
    protected double m_eta; // //parameter for regularization of output user vectors

    public UserEmbeddingSkipGram(int m, int nuIter, double converge, double alpha, double eta, double stepSize){
        super(m, nuIter, converge, alpha, stepSize);
        m_eta = eta;
    }

    @Override
    public String toString() {
        return String.format("UserEmbedding_SkipGram[dim:%d, alpha:%.4f, #Iter:%d]", m_dim, m_alpha, m_numberOfIteration);
    }

    @Override
    public void init(){

        super.init();
        m_usersOutput = new double[m_uIds.size()][m_dim];
        m_outputG = new double[m_uIds.size()][m_dim];
        for(double[] user: m_usersOutput){
            initOneVector(user);
        }
    }

    // update user vectors;
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;
        double[] vi, vj, ui, uj;
//        double[][] userInputG = new double[m_usersInput.length][m_dim];
//        double[][] userOutputG = new double[m_usersInput.length][m_dim];

        do{
            fValue = 0;
            for(int i=0; i<m_inputG.length; i++){
                Arrays.fill(m_inputG[i], 0);
                Arrays.fill(m_outputG[i], 0);
            }
            // updates based on one edges
            for(int uiIdx: m_oneEdges.keySet()){
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    // for each edge
                    vi = m_usersInput[uiIdx];
                    vj = m_usersInput[ujIdx];
                    ui = m_usersOutput[uiIdx];
                    uj = m_usersOutput[ujIdx];
                    affinity = Utils.dotProduct(vi, uj);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        m_inputG[uiIdx][g] += gTermOne * uj[g];
                        m_inputG[ujIdx][g] += gTermOne * ui[g];
                        m_outputG[uiIdx][g] += gTermOne * vj[g];
                        m_outputG[ujIdx][g] += gTermOne * vi[g];
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
                    m_inputG[i][m] -= m_alpha * 2 * m_usersInput[i][m];
                    m_outputG[i][m] -= m_eta * 2 * m_usersOutput[i][m];
                }
            }
            // update the user vectors based on the gradients
            for(int i=0; i<m_usersInput.length; i++){
                for(int j=0; j<m_dim; j++){
                    fValue -= m_alpha * m_usersInput[i][j] * m_usersInput[i][j];
                    fValue -= m_eta * m_usersOutput[i][j] * m_usersOutput[i][j];
                    m_usersInput[i][j] += m_stepSize * m_inputG[i][j];
                    m_usersOutput[i][j] += m_stepSize * m_outputG[i][j];
                }
            }
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

    @Override
    // if sampled zero edges are load, user sampled zero edges for update
    public double updateUserVectorsWithSampledZeroEdges(){
        double fValue = 0, affinity, gTermOne, vi[], vj[], ui[], uj[];
        for(int i: m_zeroEdges.keySet()){
            for(int j: m_zeroEdges.get(i)){
                if(j <= i) continue;
                // for each edge
                vi = m_usersInput[i];
                vj = m_usersInput[j];
                ui = m_usersOutput[i];
                uj = m_usersOutput[j];
                affinity = calcAffinity(i, j); // uj^T * vi
                fValue += Math.log(sigmod(-affinity));
                gTermOne = sigmod(affinity);
                // each dimension of user vectors ui and uj
                updateGradients(i, j, gTermOne, ui, uj, vi, vj);
            }
        }
        return fValue;
    }

    // uj^T * vi
    public double calcAffinity(int i, int j){
        return Utils.dotProduct(m_usersInput[i], m_usersOutput[j]);
    }

    public void updateGradients(int i, int j, double gTermOne, double[] ui, double[] uj, double[] vi, double[] vj){
        for(int g=0; g<m_dim; g++){
            m_inputG[i][g] -= gTermOne * calcUserGradientTermTwo(g, uj);
            m_inputG[j][g] -= gTermOne * calcUserGradientTermTwo(g, ui);
            m_outputG[i][g] -= gTermOne * calcUserGradientTermTwo(g, vj);
            m_outputG[j][g] -= gTermOne * calcUserGradientTermTwo(g, vi);
        }
    }

    // print out learned user embedding-output representation
    public void printUserEmbeddingOutput(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", m_usersOutput.length, m_dim);
            for (int i = 0; i < m_usersOutput.length; i++) {
                writer.format("%s\t", m_uIds.get(i));
                for (double v : m_usersOutput[i]) {
                    writer.format("%.4f\t", v);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("Finish writing %d user embeddings!", m_usersOutput.length);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args){

        String dataset = "YelpNew"; // "release-youtube"
        String model = "user_skipgram";
        int fold = 0, nuIter = 500, order = 1;

        for(int m: new int[]{20, 50, 100, 200, 300, 500}){
            String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
            String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
            String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);
            String userEmbeddingInputFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_%s_input_embedding_order_%d_dim_%d_fold_%d.txt", order, dataset, model, order, m, fold);
            String userEmbeddingOutputFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_%s_output_embedding_order_%d_dim_%d_fold_%d.txt", order, dataset, model, order, m, fold);

            double converge = 1e-6, alpha = 1, eta = 1, stepSize = 0.001;
            UserEmbeddingSkipGram skipGram = new UserEmbeddingSkipGram(m, nuIter, converge, alpha, eta, stepSize);

            skipGram.loadUsers(userFile);
            if(order >= 1)
                skipGram.loadEdges(oneEdgeFile, 1);
            if(order >= 2)
                skipGram.generate2ndConnections();
            if(order >= 3)
                skipGram.generate3rdConnections();
            skipGram.loadEdges(zeroEdgeFile, 0); // load zero edges

            skipGram.train();
            skipGram.printUserEmbedding(userEmbeddingInputFile);
            skipGram.printUserEmbeddingOutput(userEmbeddingOutputFile);
        }
    }
}
