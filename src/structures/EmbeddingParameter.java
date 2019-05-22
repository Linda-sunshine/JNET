package structures;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EmbeddingParameter {

    public String m_prefix = "./data";//"./data/CoLinAdapt"
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

    public String m_saveDir = "./data/output/";
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
            else
                exit_with_help();
        }
    }

    private void exit_with_help() {

        System.out.print("Usage: java execution [options] training_folder\noptions:\n"
                +"-prefix: the directory of the data (default ./data/)\n"
                +"-data: specific dataset used for training (default YelpNew)\noption: YelpNew, StackOverflow\n"
                +"-emIter: the number of iterations in variationla EM algorithm (default 50)\n"
                +"-nuTopics: the number of topics (default 30)\n"
                +"-dim: the dimension of user embeddings and topic embeddings (default 10)\n"
                +"-multi: run the algorithm in multi-threading or not (default true)\n"
                +"-saveDir: directory for saving the learned user embeddings and topic embeddings (default ./data/output/)\n"
                +"-kFold: speicfy which fold to train (default 0), option: 0, 1, 2, 3, 4\n"
                +"-mode: specify the experimental setting (default cv4doc)option: cv4doc, cv4edge\n"
                +"-coldStart: whether we perform experiments in cold-start setting or not (default: false)\n"
                +"--------------------------------------------------------------------------------\n"
        );
        System.exit(1);
    }
}
