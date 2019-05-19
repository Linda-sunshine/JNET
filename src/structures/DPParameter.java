package structures;


public class DPParameter {
	
	public String m_prefix = "/zf8/lg5bt/DataSigir";//"./data/CoLinAdapt"
	public String m_data = "YelpNew";
	public String m_model = "mtclinmmb";
	public double m_adaptRatio = 0.5;
	
	public int m_nuOfIterations = 30;
	public int m_M = 6;
	public int m_burnin = 10;
	public int m_thinning = 3;
	public double m_sdA = 0.0425;
	public double m_sdB = 0.0425;
	
	// Concentration parameter
	public double m_alpha = 0.01;
	public double m_eta = 0.05;
	public double m_beta = 0.01;
	
	public double m_eta1 = 0.05;
	public double m_eta2 = 0.05;
	public double m_eta3 = 0.05;
	public double m_eta4 = 0.05;
	
	// MTCLRWithDP, MTCLRWithHDP
	public double m_q = 0.1; // global parameter.
	public double m_c = 1;// coefficient in front of language model weights.
	
	// paramters for feature groups
	public int m_fv = 800;
	public int m_fvSup = 5000;
	
	// parameters for language models
	public String m_fs = "DF";
	public int m_lmTopK = 1000;
	public boolean m_post = false;
	
	// used in the sanity check of dp + x in testing.
	public int m_base = 30;
	public double m_th = 0.05;
	
	// used in the mixed model tuning.
	public int m_threshold = 15;
	
	// used in mmb model, sparsity parameter
	public double m_rho = 0.05;
	
	public boolean m_saveModel = false;
	public String m_saveDir = "/zf8/lg5bt/hdpExp/mmb";
	
	// used in testing a separate set of users
	public int m_testSize = 2000;
	public int m_trainSize = 8000;
	
	public boolean m_jointAll = false;
	public boolean m_trace = false;
	public int m_multipleE = 1;
	
	// time*friend is used as non-friend in link prediction
	public int m_time = 3;
	
	public DPParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				exit_with_help();
			else if (argv[i-1].equals("-data"))
				m_data = argv[i];
			else if(argv[i-1].equals("-adaptRatio"))
				m_adaptRatio = Double.parseDouble(argv[i]);
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta2"))
				m_eta2 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta3"))
				m_eta3 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta4"))
				m_eta4 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-model"))
				m_model = argv[i];
		
			else if (argv[i-1].equals("-fv"))
				m_fv = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-fvSup"))
				m_fvSup= Integer.valueOf(argv[i]);
			
			else if (argv[i-1].equals("-M"))
				m_M = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-alpha"))
				m_alpha = Double.valueOf(argv[i]);
			
			else if (argv[i-1].equals("-nuI"))
				m_nuOfIterations = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-sdA"))
				m_sdA = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-sdB"))
				m_sdB = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-burn"))
				m_burnin = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-thin"))
				m_thinning = Integer.valueOf(argv[i]);
			
			else if (argv[i-1].equals("-eta"))
				m_eta = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-beta"))
				m_beta = Double.valueOf(argv[i]);
		
			else if (argv[i-1].equals("-q"))
				m_q = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-lmc"))
				m_c = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-fs"))
				m_fs = argv[i];
			else if(argv[i-1].equals("-lmtopk"))
				m_lmTopK = Integer.parseInt(argv[i]);
			else if(argv[i-1].equals("-post"))
				m_post = Boolean.parseBoolean(argv[i]);
		
			// sanity check parameters.
			else if(argv[i-1].equals("-base"))
				m_base = Integer.parseInt(argv[i]);
			else if(argv[i-1].equals("-th"))
				m_th = Double.parseDouble(argv[i]);
			else if(argv[i-1].equals("-threshold"))
				m_threshold = Integer.parseInt(argv[i]);
			else if(argv[i-1].equals("-prefix"))
				m_prefix = argv[i];
			else if(argv[i-1].equals("-rho"))
				m_rho = Double.parseDouble(argv[i]);
			else if(argv[i-1].equals("-saveModel"))
				m_saveModel = Boolean.valueOf(argv[i]);
			else if(argv[i-1].equals("-trainSize")){
				m_trainSize = Integer.parseInt(argv[i]);
			} else if(argv[i-1].equals("-jointAll")){
				m_jointAll = Boolean.valueOf(argv[i]);
			} else if(argv[i-1].equals("-trace")){
				m_trace = Boolean.valueOf(argv[i]);
			} else if(argv[i-1].equals("-e")){
				m_multipleE = Integer.valueOf(argv[i]);
			} else if(argv[i-1].equals("-t")){
				m_time = Integer.valueOf(argv[i]);
			} else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
//		System.out.print(
//				 "Usage: java execution [options] training_folder\n"
//				+"options:\n"
//				+"-data: specific the dataset used for training (default Amazon)\noption: Amazon, Yelp\n"
//				+"-adaptRatio: the ratio of data for training: batch-0.5; online-1 (default 0.5, online must be 1)\n"
//				+"-eta1: coefficient for the scaling in each user group's regularization (default 1)\n"
//				+"-eta2: coefficient for the shifting in each user group's regularization (default 0.5)\n"
//				+"-eta3: coefficient for the scaling in super user's regularization (default 0.1)\n"
//				+"-eta4: coefficient for the shifting in super user's regularization (default 0.3)\n"
//				+"-model: specific training model,\noption: Base-base, GlobalSVM-gsvm, IndividualSVM-indsvm, RegLR-reglr, LinAdapt-linadapt, MultiTaskSVM-mtsvm, MTLinAdapt_Batch-mtlinbatch, MTLinAdapt_Online-mtlinonline\n"
//				+"-fv: feature groups for user groups (default 800),\noption: 400, 800, 1600, 5000\n"
//				+"-fvSup: feature groups for super user(default 5000)\n"
//				+"-M : the size of the auxiliary variables in the posterior inference of the group indicator (default 6)\n"
//				+"-alpha : concentraction parameter for DP (default 1.0)\n"
//				+"-nuI : number of iterations for sampling (default 30)\n"
//				+"-sdA : variance for the normal distribution for the prior of shifting parameters (default 0.2)\n"
//				+"-sdB : variance for the normal distribution for the prior of sacling parameters (default 0.2)\n"
//				+"-burn : number of iterations in burn-in period (default 5)\n"
//				+"-thin : thinning of sampling chain (default 5)\n"
//				+"--------------------------------------------------------------------------------\n"
//		);
		System.exit(1);
	}
}
