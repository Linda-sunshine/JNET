package structures;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EmbeddingParameter {

    public String m_prefix = "/zf8/lg5bt/DataSigir";//"./data/CoLinAdapt"
    public String m_data = "YelpNew";

    public int m_emIter = 50;
    public int m_number_of_topics = 10;
    public int m_varIter = 10;
    public int m_trainInferIter = 1;
    public int m_testInferIter = 1500;
    public int m_paramIter = 20;

    public int m_embeddingDim = 10;
    public int m_kFold = 0;

    public boolean m_multiFlag = true;
    public double m_stepSize = 1e-3;

    public double m_gamma = 0.1;
    public boolean m_alphaFlag = true;
    public boolean m_gammaFlag = false;
    public boolean m_betaFlag = true;
    public boolean m_tauFlag = false;
    public boolean m_xiFlag = false;
    public boolean m_rhoFlag = false;
    public boolean m_ada = false;
    public boolean m_coldStartFlag = false;

    public String m_saveDir = "";
    public String m_mode = "cv4doc";

    public EmbeddingParameter(String argv[]) {

        int i;

        //parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-')
                break;
            else if (++i >= argv.length)
                exit_with_help();
            else if (argv[i - 1].equals("-prefix"))
                m_prefix = argv[i];
            else if (argv[i - 1].equals("-data"))
                m_data = argv[i];
            else if (argv[i - 1].equals("-emIter"))
                m_emIter = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-nuTopics"))
                m_number_of_topics = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-varIter"))
                m_varIter = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-paramIter"))
                m_paramIter = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-dim"))
                m_embeddingDim = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-multi"))
                m_multiFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-stepSize"))
                m_stepSize = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-gamma"))
                m_gamma = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-saveDir"))
                m_saveDir = argv[i];
            else if (argv[i - 1].equals("-alphaFlag"))
                m_alphaFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-gammaFlag"))
                m_gammaFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-betaFlag"))
                m_betaFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-tauFlag"))
                m_tauFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-xiFlag"))
                m_xiFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-rhoFlag"))
                m_rhoFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-kFold"))
                m_kFold = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-ada"))
                m_ada = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-mode"))
                m_mode = argv[i];
            else if (argv[i - 1].equals("-trainInferIter"))
                m_trainInferIter = Integer.valueOf(argv[i]);
            else if(argv[i - 1].equals("-testInferIter"))
                m_testInferIter = Integer.valueOf(argv[i]);
            else if(argv[i - 1].equals("-coldStart"))
                m_coldStartFlag = Boolean.valueOf(argv[i]);

        }
        // must specify the save directory for data
        if(m_saveDir.length() == 0){
            System.out.println("[Error]Please specify the save directory!!");
            exit_with_help();
        }
    }

    private void exit_with_help() {

        System.out.print("Usage: java execution [options] training_folder\n" +"options:\n"
                +"-data: specific the dataset used for training (default YelpNew)\noption: Yelp, StackOverflow\n"
                +"-: the ratio of data for training: batch-0.5; online-1 (default 0.5, online must be 1)\n"
                +"-eta1: coefficient for the scaling in each user group's regularization (default 1)\n"
                +"-eta2: coefficient for the shifting in each user group's regularization (default 0.5)\n"
                +"-eta3: coefficient for the scaling in super user's regularization (default 0.1)\n"
                +"-eta4: coefficient for the shifting in super user's regularization (default 0.3)\n"
                +"-model: specific training model,\noption: Base-base, GlobalSVM-gsvm, IndividualSVM-indsvm, RegLR-reglr, LinAdapt-linadapt, MultiTaskSVM-mtsvm, MTLinAdapt_Batch-mtlinbatch, MTLinAdapt_Online-mtlinonline\n"
                +"-fv: feature groups for user groups (default 800),\noption: 400, 800, 1600, 5000\n"
                +"-fvSup: feature groups for super user(default 5000)\n"
                +"-M : the size of the auxiliary variables in the posterior inference of the group indicator (default 6)\n"
                +"-alpha : concentraction parameter for DP (default 1.0)\n"
                +"-nuI : number of iterations for sampling (default 30)\n"
                +"-sdA : variance for the normal distribution for the prior of shifting parameters (default 0.2)\n"
                +"-sdB : variance for the normal distribution for the prior of sacling parameters (default 0.2)\n"
                +"-burn : number of iterations in burn-in period (default 5)\n"
                +"-thin : thinning of sampling chain (default 5)\n"
                +"--------------------------------------------------------------------------------\n"
        );

        System.exit(1);
    }
}
