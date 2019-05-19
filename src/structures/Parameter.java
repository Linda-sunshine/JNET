/**
 * 
 */
package structures;

/**
 * @author hongning
 * To support command line input arguments
 */
public class Parameter {
	/***** Default setting for these parameters *****/
	public int m_classNumber = -1; //has to be specified by user now!
	public int m_Ngram = 2; //The default value is unigram. 
	public int m_lengthThreshold = 10; //Document length threshold
	
	//"TF", "TFIDF", "BM25", "PLN"
	public String m_featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
	public int m_norm = 2;//The way of normalization.(only 1 and 2)
	public int m_CVFold = 10; //k fold-cross validation
	
	//Supervised classification models: "NB", "LR", "PR-LR", "SVM"
	//Semi-supervised classification models: "GF", "GF-RW", "GF-RW-ML"
	//Topic Models: "2topic", "pLSA", "HTMM", "LRHTMM"
	public String m_model = "SVM"; //Which model to use.
	
	//"PR"
	public String m_weightScheme = "NONE"; // weather we will use computed weighting
	
	//"SUP", "TRANS", "TM", "FV"
	public String m_style = "SUP"; // FV means save vector representation of documents to file
	public double m_sampleRate = 0.25; // sampling rate for Gaussian Fields when constructing the graph
	public int m_kUL = 100; // k nearest labeled neighbors
	public int m_kUU = 50; // k' nearest unlabeled neighbors
	public boolean m_weightedAvg = true; // True: weighted sum in random walk; False: majority vote in random walk
	public boolean m_simFlag = false; // True: use similarity weight in majority vote;
	public int m_bound = 3; // rating difference for generating pairwise constraints
	public String m_classifier = "SVM"; // base classifier for Gaussian Field
	public double m_eta = 0.1; //random start ratio
	public double m_cSampleRate = 0.001; // sampling rate of constraints for metric learning
	
	/*****The parameters used in loading files.*****/
	public String m_folder = null;
	public String m_suffix = ".json";
	public String m_tokenModel = "./data/Model/en-token.bin"; //Token model.
	public String m_stnModel = "./data/Model/en-sent.bin"; //Sentence model.
	public String m_posModel = "./data/Model/en-pos-maxent.bin"; //Sentence model.
	public String m_stopwords = "./data/Model/stopwords.dat";
	public String m_featureFile = null;//list of controlled vocabulary
	public String m_featureStat = "./data/Features/fv_stat.dat";//detailed statistics of the selected features
	public String m_fvFile = null; // vector representation of documents
	public String m_debugOutput = null; // debug output file

	/*****Parameters in feature selection.*****/
	public String m_featureSelection = "CHI"; //Feature selection method.
	public double m_startProb = 0.4; // Used in feature selection, the starting point of the features.
	public double m_endProb = 0.999; // Used in feature selection, the ending point of the features.
	public int m_maxDF = -1; // Filter the features with DFs larger than this threshold.
	public int m_minDF = 5; // Filter the features with DFs smaller than this threshold.
	
	/*****Parameters specified for classifiers.*****/
	public double m_C = 1.0; // trade-off parameter in LR and SVM
	
	/*****Parameters specified for classifiers.*****/
	public int m_numTopics = 50; // number of topics
	public double m_alpha = 1.05; // dirichlet prior for p(z|d)
	public double m_beta = 1.01; // dirichlet prior for p(w|z)
	public double m_lambda = 0.8; // p(B) in pLSA and 2topic model, L2 regularization in LRHTMM
	public double m_gamma = 5.0; // strength of prior words 
	public double m_converge = 1e-5; // EM convergence rate
	public int m_maxmIterations = 100; // maximum number of iterations
	public boolean m_multithread = false; // using multi-thread for topic models
	public int m_maxVarIterations = 10; // maximum number of iterations in variational inference
	public double m_varConverge = 1e-7; // convergence rate for variational inference
	public double m_burnIn = 0.4; // burn in period in Gibbs sampling
	public int m_lag = 10; // lag in accumulating Gibbs samples
	public String m_priorFile = null; // prior seed word list
	
	public Parameter(String argv[])
	{
		int i;
		
		// parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				exit_with_help();
			else if (argv[i-1].equals("-suf"))
				m_suffix = argv[i];
			else if (argv[i-1].equals("-st"))
				m_stopwords = argv[i];
			else if (argv[i-1].equals("-fpath"))
				m_featureFile = argv[i];
			else if (argv[i-1].equals("-fstat"))
				m_featureStat = argv[i];
			else if (argv[i-1].equals("-vf"))
				m_fvFile = argv[i];
			else if (argv[i-1].equals("-dbf"))
				m_debugOutput = argv[i];
			else if (argv[i-1].equals("-fs"))
				m_featureSelection = argv[i];
			else if (argv[i-1].equals("-sp"))
				m_startProb = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-ep"))
				m_endProb = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-df_min"))
				m_minDF = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-df_max"))
				m_maxDF = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-cs"))
				m_classNumber = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-ngram"))
				m_Ngram = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-lcut"))
				m_lengthThreshold = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-fv"))
				m_featureValue = argv[i];
			else if (argv[i-1].equals("-norm"))
				m_norm = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-cv"))
				m_CVFold = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-c"))
				m_model = argv[i];
			else if (argv[i-1].equals("-C"))
				m_C = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-w"))
				m_weightScheme = argv[i];
			else if (argv[i-1].equals("-s"))
				m_style = argv[i];
			else if (argv[i-1].equals("-cf"))
				m_classifier = argv[i];
			else if (argv[i-1].equals("-sr"))
				m_sampleRate = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-wm"))
				m_weightedAvg = argv[i].equals("1");
			else if (argv[i-1].equals("-sf"))
				m_simFlag = argv[i].equals("1");
			else if (argv[i-1].equals("-kUL"))
				m_kUL = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-kUU"))
				m_kUU = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-bd"))
				m_bound = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-csr"))
				m_cSampleRate = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-k"))
				m_numTopics = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-alpha"))
				m_alpha = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-gamma"))
				m_gamma = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-beta"))
				m_beta = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-lambda"))
				m_lambda = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta"))
				m_eta = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-iter"))
				m_maxmIterations = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-con"))
				m_converge = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-viter"))
				m_maxVarIterations = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-vcon"))
				m_varConverge = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-burnin"))
				m_burnIn = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-lag"))
				m_lag = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-mt"))
				m_multithread = argv[i].equals("1");
			else if (argv[i-1].equals("-fprior"))
				m_priorFile = argv[i];
			else
				exit_with_help();
		}
		
		if(i==argv.length-1)
			m_folder = argv[i];
		
		if (m_classNumber<=0){
			System.err.println("Class number has to be manually set!");
			System.exit(-1);
		}
	}
	
	private void exit_with_help()
	{
		System.out.print(
		 "Usage: java execution [options] training_folder\n"
		+"options:\n"
		+"-suf suffix : type of files to be loaded (default .json)\n"
		+"-st stopword_file : list of files that will be excluded in feature generation\n"
		+"-fpath cv_file : list of controlled vocabular to be used in feature generation (default null)\n"
		+"-fstat fv_file : statistics of the features collected from the corpus (default null)\n"
		+"-vf vct_file : vector representation file (default null)\n"
		+"-dbf debug_file : classifier's debug output file (default null)\n"
		+"-fs type : feature selection method (default CHI)\n"
		+"	DF -- Document frequency\n"
		+"	CHI -- Chi-Square test statistics\n"
		+"	IG -- Informatoin gain\n"
		+"	MI -- Mutual information\n"
		+"-sp float : ignore the bottom proportion of ranked features (default 0.4)\n"
		+"-ep float : ignore the top proportion of ranked features (default 0.999)\n"
		+"-df int : ignore the features occurring less than c documents (default 10) \n"
		+"-cs int : total number of classes (has to be manually specified!)\n"
		+"-ngram int : n-gram for feature generation (default 2)\n"
		+"-lcut int : ignore the documents with length less than c (default 5)\n"
		+"-window int : window size in time series based sentiment analysis (default 0)\n"
		+"-cv int : cross validation fold (default 10)\n"
		+"-fv type : feature value generation method (default TFIDF)\n"
		+"	TF -- Term frequency\n"
		+"	TFIDF -- Term frequency times inverse document frequence\n"
		+"	BM25 -- Term frequency times BM25 IDF with document length normalization\n"
		+"	PLN -- Pivoted length normalization\n"
		+"-norm int : feature value normalization method (default L2)\n"
		+"	1 -- L1 normalization\n"
		+"	2 -- L2 normalization\n"
		+"	0 -- No normalization\n"
		+"-c type : classification method (default SVM)\n"
		+"	NB -- Naive Bayes\n"
		+"	LR -- Logistic Regression\n"
		+"	PR-LR -- Posterior Regularized Logistic Regression\n"
		+"	SVM -- Support Vector Machine (liblinear)\n"
		+"	GF -- Gaussian Fields by matrix inversion\n"
		+"	GF-RW -- Gaussian Fields by random walk\n"
		+"	GF-RW-ML -- Gaussian Fields by random walk with distance metric learning (by libliner)\n"
		+"	2topic -- Two-Topic Topic Model\n"
		+"	pLSA -- Probabilistic Latent Semantic Analysis\n"
		+"	gLDA -- Latent Dirichlet Allocation with Gibbs sampling\n"
		+"	vLDA -- Latent Dirichlet Allocation with variational inference\n"
		+"	HTMM -- Hidden Topic Markov Model\n"
		+"	LRHTMM -- MaxEnt Hidden Topic Markov Model\n"
		+"-cf type : multiple learning in Gaussian Fields (default SVM)\n"
		+"	NB -- Naive Bayes\n"
		+"	LR -- Logistic Regression\n"
		+"	PR-LR -- Posterior Regularized Logistic Regression\n"
		+"	SVM -- Support Vector Machine (liblinear)\n"
		+"-w type : instance weighting scheme (default None)\n"
		+"	PR -- Content similarity based PageRank\n"
		+"-s type : learning paradigm (default SUP)\n"
		+"	SUP -- Supervised learning\n"
		+"	SEMI -- Semi-supervised learning\n"
		+"	TM -- Topic Models\n"
		+"-C float -- trade-off parameter in LR and SVM (default 0.1)\n"
		+"-sr float : Sample rate for transductive learning (default 0.25)\n"
		+"-kUL int : k nearest labeled neighbors (default 100)\n"
		+"-kUU int : kP nearest unlabeled neighbors (default 50)\n"
		+"-wm 0/1 : weighted sum or majority vote in random walk\n"
		+"-sf 0/1 : use similiarity as weight in majority vote\n"
		+"-bd int : rating difference bound in generating pairwise constraint (default 3)\n"
		+"-csr double : constrain sampling rate for metric learning (default 1e-3)\n"
		+"-k int : number of topics (default 50)\n"
		+"-fprior prior_file : prior seed word list (default null)\n"
		+"-alpha float : dirichlet prior for p(z|d) (default 1.05)\n"
		+"-beta float : dirichlet prior for p(w|z) (default 1.01)\n"
		+"-lambda float : manual background proportion setting p(B) (default 0.8)\n"
		+"-eta float : random restart probability eta in randowm walk (default 0.1)\n"
		+"-gamma float : strength of prior (default 5.0)\n"
		+"-iter int : maximum number of EM iteration (default 100)\n"
		+"-con float : convergency limit for EM iterations (default 1e-5)\n"
		+"-viter int : maximum number of variational inference iteration (default 10)\n"
		+"-vcon float : convergency limit for variational inference iterations (default 1e-7)\n"
		+"-burn float : burn in period of sampling method (default 0.4)\n"
		+"-lag int : sampling lag when accumulating samples (default 10)\n"
		+"-mt 0/1 : using multi-thread for topic models (default 0)\n"	
		);
		System.exit(1);
	}
	
	public String toString() {
		StringBuffer buffer = new StringBuffer(512);
		buffer.append("\n--------------------------------------------------------------------------------------");
		buffer.append("\nParameters of learning procedure:");
		buffer.append("\n#Class: " + m_classNumber + "\tNgram: " + m_Ngram + "\tFeature value: " + m_featureValue + "\tNormalization: " + m_norm);
		buffer.append("\nDoc length cut: " + m_lengthThreshold);
		
		if (m_style.equals("TM")) {
			buffer.append("\nTopic Model: " + m_model + "\t#Topics: " + m_numTopics + "\tCross validation: " + m_CVFold);
			buffer.append("\nalpha: " + m_alpha + "\tbeta: " + m_beta + "\tlambda: " + m_lambda 
					+ "\n#Iterations: " + m_maxmIterations + "\tConvergency: " + m_converge + "\tMulti-thread: " + m_multithread);
			
			if (m_model.equals("vLDA"))
				buffer.append("\n#VarIter: " + m_maxVarIterations + "\tVarConverge: " + m_varConverge);
			else if (m_model.equals("gLDA"))
				buffer.append("\nBurn in: " + m_burnIn + "\tLag: " + m_lag);
			
			if (m_priorFile!=null && m_priorFile.isEmpty()==false) 
				buffer.append("\nPrior file: " + m_priorFile + "\tStrength: " + m_gamma);
		} else {
			if (m_style.equals("SEMI")) {
				buffer.append("\nLearning paradigm: SEMI\tSampling rate: " + m_sampleRate + "\tkUL: " + m_kUL + "\tkUU: " + m_kUU
						+ "\nalpha: " + m_alpha + "\tbeta: " + m_beta);
				
				if (m_model.contains("-RW"))
					buffer.append("\teta: " + m_eta + "\tconverge: " + m_converge);
				if (m_model.contains("-ML"))
					buffer.append("\nDiff bound: " + m_bound + "\tcSample Rate: " + m_cSampleRate);
				
				buffer.append("\nSolver: " + (m_model.equals("GF")?"Matrix Inversion":"Random Walk") + "\tBase Classifer: " + m_classifier + "\tWeighted Sum: " + m_weightedAvg + "\tSimilarity Weight: " + m_simFlag);
			} else
				buffer.append("\nLearning paradigm: SUP");
			
			if (m_model.equals("LR") || m_model.equals("SVM"))
				buffer.append("\nClassifier: " + m_model + "\tInstance weighting: " + m_weightScheme + "\tTrade-off Parameter: " + m_C+ "\tCross validation: " + m_CVFold);
			else
				buffer.append("\nClassifier: " + m_model + "\tInstance weighting: " + m_weightScheme + "\tCross validation: " + m_CVFold);
		}
		
		if (m_folder!=null)
			buffer.append("\nData directory: " + m_folder);
		else if (m_fvFile!=null)
			buffer.append("\nVector file: " + m_fvFile);
		buffer.append("\n--------------------------------------------------------------------------------------");
		return buffer.toString();
	}
	
	public String printFeatureSelectionConfiguration() {
		StringBuffer buffer = new StringBuffer(512);
		buffer.append("--------------------------------------------------------------------------------------");
		buffer.append("\nParameters of feature selection:");
		buffer.append("\nSelection method: " + m_featureSelection + "\tDF max cut: " + m_maxDF + "\tDF min cut: " + m_minDF + "\tRange: [" + m_startProb + "," + m_endProb + "]");
		buffer.append("\nFeature file: " + m_featureFile + "\tStatistics file: " + m_featureStat);
		buffer.append("\n--------------------------------------------------------------------------------------");
		return buffer.toString();
	}
}
