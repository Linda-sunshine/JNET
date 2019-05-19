package structures;

public class RoleParameter {

    public String m_prefix = "/zf8/lg5bt";//"./data/CoLinAdapt"
    public String m_data = "YelpNew";
    public String m_model = "user"; // "user", "user_skipgram", "multirole"

    public int m_fold = 0;
    public int m_dim = 10;
    public int m_iter = 150;
    public int m_nuOfRoles = 10;

    public double m_converge = 1e-6;
    public double m_alpha = 0.5;
    public double m_eta = 0.5;
    public double m_beta = 0.5;
    public double m_gamma = 0.5;
    public double m_stepSize = 0.001;

    // whether we model the second order connections
    public int m_order = 1;

    public RoleParameter(String argv[]) {

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
            else if(argv[i - 1].equals("-model"))
                m_model = argv[i];
            else if(argv[i - 1].equals("-fold"))
                m_fold = Integer.valueOf(argv[i]);
            else if(argv[i - 1].equals("-dim"))
                m_dim = Integer.valueOf(argv[i]);
            else if(argv[i - 1].equals("-iter"))
                m_iter = Integer.valueOf(argv[i]);
            else if(argv[i - 1].equals("-nuOfRoles"))
                m_nuOfRoles = Integer.valueOf(argv[i]);
            else if(argv[i - 1].equals("-alpha"))
                m_alpha = Double.valueOf(argv[i]);
            else if(argv[i - 1].equals("-eta"))
                m_eta = Double.valueOf(argv[i]);
            else if(argv[i - 1].equals("-beta"))
                m_beta = Double.valueOf(argv[i]);
            else if(argv[i - 1].equals("-gamma"))
                m_gamma = Double.valueOf(argv[i]);
            else if(argv[i - 1].equals("-stepSize"))
                m_stepSize = Double.valueOf(argv[i]);
            else if(argv[i - 1].equals("-order"))
                m_order = Integer.valueOf(argv[i]);

        }
    }
    private void exit_with_help(){
        System.exit(1);
    }
}
