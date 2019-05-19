package structures;


public class TopicModelParameter {
	
	public String m_prefix = "/zf18/ll5fy/lab/dataset";//"./data/CoLinAdapt"
	public String m_source = "yelp"; // "amazon_movie"
	public String m_set = "byUser_4k_review";
	public String m_topicmodel = "ETBIR";
	public int m_crossV = 5;

	public double m_beta = 1 + 1e-3;
	public double m_alpha = 1 + 1e-2;
	public double m_lambda = 1 + 1e-3;
	
	// model parameters for ETBIR
	public double m_sigma = 1.0 + 1e-2;
	public double m_rho = 1.0 + 1e-2;
	
	public int m_topk = 30;
	public int m_emIter = 50;
	public int m_number_of_topics = 30;
	public int m_varMaxIter = 20; // variational inference max iter number
	
	public double m_varConverge = 1e-6;
	public double m_emConverge = 1e-10;
	
	public String m_output = String.format("%s/%s/%s/output", m_prefix, m_source, m_set);// output directory

	public boolean m_flag_gd = false;
	public boolean m_flag_fix_lambda = false;
	public boolean m_flag_diagonal = false;

	public boolean m_flag_tune = false;

	public boolean m_flag_coldstart = false;

	//item tagging
	public String m_mode;
		
	public TopicModelParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				System.exit(1);
			else if (argv[i-1].equals("-prefix"))
				m_prefix = argv[i];
			else if (argv[i-1].equals("-source"))
				m_source = argv[i];
            else if (argv[i-1].equals("-set"))
                m_set = argv[i];
			else if(argv[i-1].equals("-topicmodel"))
				m_topicmodel = argv[i];
            else if(argv[i-1].equals("-crossV")){
                m_crossV = Integer.valueOf(argv[i]);
            }

			else if(argv[i-1].equals("-alpha"))
				m_alpha = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-beta"))
				m_beta = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-lambda"))
				m_lambda = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-sigma"))
				m_sigma = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-rho"))
				m_rho = Double.valueOf(argv[i]);
			
			else if(argv[i-1].equals("-topk"))
				m_topk = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-emIter"))
				m_emIter = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-nuOfTopics"))
				m_number_of_topics = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-varMaxIter"))
				m_varMaxIter = Integer.valueOf(argv[i]);
			
			else if(argv[i-1].equals("-varConverge"))
				m_varConverge = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-emConverge"))
				m_emConverge = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-output"))
				m_output = argv[i];

			else if (argv[i-1].equals("-flagGd"))
				m_flag_gd = argv[i].equals("true");
			else if (argv[i-1].equals("-flagFixLambda"))
				m_flag_fix_lambda = argv[i].equals("true");
			else if (argv[i-1].equals("-flagDiagonal"))
				m_flag_diagonal = argv[i].equals("true");
			else if (argv[i-1].equals("-flagTune"))
				m_flag_tune = argv[i].equals("true");
			else if (argv[i-1].equals("-flagColdstart"))
				m_flag_coldstart = argv[i].equals("true");

			else if(argv[i-1].equals("-mode"))
				m_mode = argv[i];
			else
				System.exit(1);
		}
	}
}