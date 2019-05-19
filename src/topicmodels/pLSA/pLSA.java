package topicmodels.pLSA;

/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * Probabilistic Latent Semantic Analysis Topic Modeling 
 */

import structures.*;
import topicmodels.twoTopic;
import utils.Utils;

import java.io.*;
import java.util.*;


public class pLSA extends twoTopic {
	// Dirichlet prior for p(\theta|d)
	protected double d_alpha; // smoothing of p(z|d)
	
	protected double[][] topic_term_probabilty ; /* p(w|z) */	
	protected double[][] word_topic_prior; /* prior distribution of words under a set of topics, by default it is null */
	
	protected boolean m_sentiAspectPrior = false; // symmetric sentiment aspect prior
	
	public pLSA(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			double lambda, //arguments for 2topic topic model
			int number_of_topics, double alpha) { //arguments for pLSA			
		super(number_of_iteration, converge, beta, c, lambda);
		
		this.d_alpha = alpha;
		this.number_of_topics = number_of_topics;		
		m_logSpace = false;
		
		createSpace();
	}
	
	protected void createSpace() {
		topic_term_probabilty = new double[this.number_of_topics][this.vocabulary_size];
		word_topic_sstat = new double[this.number_of_topics][this.vocabulary_size];		
		background_probability = new double[vocabulary_size];//to be initialized during EM
	}
	
	public void setSentiAspectPrior(boolean senti) {
		m_sentiAspectPrior = senti; 
		if (senti && number_of_topics%2==1) {
			System.err.format("The topic size (%d) specified is not even!", number_of_topics);
			System.exit(-1);
		}
	}
	
	public void LoadPrior(String filename, double eta) {		
		if (filename == null || filename.isEmpty())
			return;
		
		try {
			String tmpTxt;
			String[] container;
			
			HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
			for(int i=0; i<m_corpus.getFeatureSize(); i++)
				featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());
			
			int wid, wCount = 0;
			
			double[] prior;
			ArrayList<double[]> priorWords = new ArrayList<double[]>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			while( (tmpTxt=reader.readLine()) != null ){
				tmpTxt = tmpTxt.trim();
				if (tmpTxt.isEmpty())
					continue;
				
				container = tmpTxt.split(" ");
				wCount = 0;
				prior = new double[vocabulary_size];
				for(int i=1; i<container.length; i++) {
					if (featureNameIndex.containsKey(container[i])) {
						wid = featureNameIndex.get(container[i]); // map it to a controlled vocabulary term
						prior[wid] = eta;
						wCount++;
					}
				}
				
				System.out.format("Prior keywords for Topic %d (%s): %d/%d\n", priorWords.size(), container[0], wCount, container.length-1);
				priorWords.add(prior);
			}
			reader.close();
			
			word_topic_prior = priorWords.toArray(new double[priorWords.size()][]);
			
			if (m_sentiAspectPrior && word_topic_prior.length%2==1) {
				System.err.format("The topic size (%d) specified in the sentiment-aspect seed words is not even!", word_topic_prior.length);
				System.exit(-1);
			} else if (word_topic_prior.length > number_of_topics) {
				System.err.format("More topics specified in seed words (%d) than topic model's configuration(%d)!\n", word_topic_prior.length, number_of_topics);
				System.err.format("Reset the topic size to %d!\n", word_topic_prior.length);
				
				this.number_of_topics = word_topic_prior.length;
				createSpace();
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public String toString() {
		return String.format("pLSA[k:%d, lambda:%.2f]", number_of_topics, m_lambda);
	}

	@Override
	protected void initialize_probability(Collection<_Doc> collection) {	
		// initialize topic document proportion, p(z|d)
		// initialize background topic
		Arrays.fill(background_probability, d_beta-1.0);
		for(_Doc d:collection) {
			d.setTopics(number_of_topics, d_alpha-1.0);//allocate memory and randomize it
			for(_SparseFeature fv:d.getSparse()) 
				background_probability[fv.getIndex()] += fv.getValue();
		}
		Utils.L1Normalization(background_probability);
		
		// initialize term topic matrix p(w|z,\phi)
		for(int i=0;i<number_of_topics;i++)
			Utils.randomize(word_topic_sstat[i], d_beta-1.0);
		imposePrior();
		
		calculate_M_step(0);
	}

	@Override
	public void initial(){}

	protected void imposePrior() {		
		if (word_topic_prior!=null) {//we have enforced that the topic size is at least as many as prior seed words
			if (m_sentiAspectPrior) {
				int size = word_topic_prior.length/2, shift = number_of_topics/2;//if it is sentiment aspect prior, the size must be even
				for(int k=0; k<size; k++) {
					for(int n=0; n<vocabulary_size; n++) {
						word_topic_sstat[k][n] += word_topic_prior[k][n];
						word_topic_sstat[k + shift][n] += word_topic_prior[k + size][n];
					}
				}
			} else {//no symmetric property
				for(int k=0; k<word_topic_prior.length; k++) {
					for(int n=0; n<vocabulary_size; n++)
						word_topic_sstat[k][n] += word_topic_prior[k][n];
				}
			}
		}
	}
	
	@Override
	protected void init() { // clear up for next iteration during EM
		for(int k=0;k<this.number_of_topics;k++)
			Arrays.fill(word_topic_sstat[k], d_beta-1.0);//pseudo counts for p(w|z)
		imposePrior();
		
		//initiate sufficient statistics
		for(_Doc d:m_trainSet)
			Arrays.fill(d.m_sstat, d_alpha-1.0);//pseudo counts for p(\theta|d)
	}
	
	@Override
	protected void initTestDoc(_Doc d) {
		//allocate memory and randomize it
		d.setTopics(number_of_topics, d_alpha-1.0);//in real space
		estThetaInDoc(d);
	}
	
	@Override
	public double calculate_E_step(_Doc d) {	
		double propB; // background proportion
		double exp; // expectation of each term under topic assignment
		for(_SparseFeature fv:d.getSparse()) {
			int j = fv.getIndex(); // jth word in doc
			double v = fv.getValue();
			
			//-----------------compute posterior----------- 
			double sum = 0;
			for(int k=0;k<this.number_of_topics;k++)
				sum += d.m_topics[k]*topic_term_probabilty[k][j];//shall we compute it in log space?
			
			propB = m_lambda * background_probability[j];
			propB /= propB + (1-m_lambda) * sum;//posterior of background probability
			
			//-----------------compute and accumulate expectations----------- 
			for(int k=0;k<this.number_of_topics;k++) {
				exp = v * (1-propB)*d.m_topics[k]*topic_term_probabilty[k][j]/sum;
				d.m_sstat[k] += exp;
				
				if (m_collectCorpusStats)//when testing, we don't need to collect sufficient statistics
					word_topic_sstat[k][j] += exp;
			}
		}
		
		if (m_collectCorpusStats==false || m_converge>0)
			return calculate_log_likelihood(d);
		else
			return 1;//no need to compute likelihood
	}
	
	@Override
	public void calculate_M_step(int iter) {
		// update topic-term matrix
		double sum = 0;
		for(int k=0;k<this.number_of_topics;k++) {
			sum = Utils.sumOfArray(word_topic_sstat[k]);
			for(int i=0;i<this.vocabulary_size;i++)
				topic_term_probabilty[k][i] = word_topic_sstat[k][i] / sum;
		}
		
		// update per-document topic distribution vectors
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
	}
	
	protected double docThetaLikelihood(_Doc d) {
		double logLikelihood = 0; //Utils.lgamma(number_of_topics * d_alpha) - number_of_topics*Utils.lgamma(d_alpha);
		for(int i=0; i<this.number_of_topics; i++) {
			if (m_logSpace)
				logLikelihood += (d_alpha-1) * d.m_topics[i];
			else
				logLikelihood += (d_alpha-1) * Math.log(d.m_topics[i]);
		}
		return logLikelihood;
	}
	
	@Override
	protected void estThetaInDoc(_Doc d) {
		double sum = Utils.sumOfArray(d.m_sstat);//estimate the expectation of \theta
		for(int k=0;k<this.number_of_topics;k++) {
			if (m_logSpace)
				d.m_topics[k] = Math.log(d.m_sstat[k]/sum);
			else
				d.m_topics[k] = d.m_sstat[k]/sum;
		}
	}
	
	/*likelihod calculation */
	/* M is number of doc
	 * N is number of word in corpus
	 */
	/* p(w,d) = sum_1_M sum_1_N count(d_i, w_j) * log[ lambda*p(w|theta_B) + [lambda * sum_1_k (p(w|z) * p(z|d)) */ 
	//NOTE: cannot be used for unseen documents!
	@Override
	protected double calculate_log_likelihood(_Doc d) {		
		double logLikelihood = docThetaLikelihood(d), prob;
		for(_SparseFeature fv:d.getSparse()) {
			int j = fv.getIndex();	
			prob = 0.0;
			for(int k=0;k<this.number_of_topics;k++)//\sum_z p(w|z,\theta)p(z|d)
				prob += d.m_topics[k]*topic_term_probabilty[k][j];
			prob = prob*(1-m_lambda) + this.background_probability[j]*m_lambda;//(1-\lambda)p(w|d) * \lambda p(w|theta_b)
			logLikelihood += fv.getValue() * Math.log(prob);
		}
		return logLikelihood;
	}
	
	//corpus-level parameters will be only called during training
	@Override
	protected double calculate_log_likelihood() {		
		//prior from Dirichlet distributions
		double logLikelihood = number_of_topics * (Utils.lgamma(vocabulary_size*d_beta) - vocabulary_size*Utils.lgamma(d_beta));;
		for(int i=0; i<this.number_of_topics; i++) {
			for(int v=0; v<this.vocabulary_size; v++) {
				if (m_logSpace)
					logLikelihood += (d_beta-1) * topic_term_probabilty[i][v];
				else
					logLikelihood += (d_beta-1) * Math.log(topic_term_probabilty[i][v]);
			}
		}
		
		return logLikelihood;
	}

	public void printParameterAggregation(int k, String folderName, String topicmodel){
		String phiPathByUser = String.format("%s%s_phiByUser_%d.txt", folderName, topicmodel, number_of_topics);
		String phiPathByItem = String.format("%s%s_phiByItem_%d.txt", folderName, topicmodel, number_of_topics);
		String phiPath = String.format("%s%s_phi_%d.txt", folderName, topicmodel, number_of_topics);
		String betaPath = String.format("%s%s_beta_%d.txt", folderName, topicmodel, number_of_topics);

		//print out phi per doc, and beta
		printPhi(phiPath);
		printBeta(betaPath);

		//aggregate parameter \gamma by user/item
		printTopWords(k, phiPathByUser, getDocByUser());
		printTopWords(k, phiPathByItem, getDocByItem());

		//overall topic words
		printTopWords(k, String.format("%s%s_topWords_%d.txt", folderName, topicmodel, number_of_topics));
	}
	
	//print all the quantities in real space
	@Override
	public void printTopWords(int k, String topWordPath) {
		System.out.println("TopWord FilePath:" + topWordPath);
		Arrays.fill(m_sstat, 0);
		for(_Doc d:m_trainSet) {
			for(int i=0; i<number_of_topics; i++)
				m_sstat[i] += m_logSpace?Math.exp(d.m_topics[i]):d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);			
	
		try{
			PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));
		
			for(int i=0; i<topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
				for(int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));
				
				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for(_RankItem it:fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name, m_logSpace?Math.exp(it.m_value):it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch(Exception ex){
			System.err.print("File Not Found");
		}
	}

	//print all the quantities in real space
	@Override
	public void printTopWords(int k) {
		Arrays.fill(m_sstat, 0);
		for(_Doc d:m_trainSet) {
			for(int i=0; i<number_of_topics; i++)
				m_sstat[i] += m_logSpace?Math.exp(d.m_topics[i]):d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		for(int i=0; i<topic_term_probabilty.length; i++) {
			MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
			for(int j = 0; j < vocabulary_size; j++)
				fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));
			System.out.format("Topic %d(%.5f):\t", i, m_sstat[i]);
			for(_RankItem it:fVector)
				System.out.format("%s(%.5f)\t", it.m_name, m_logSpace?Math.exp(it.m_value):it.m_value);
			System.out.println();
		}
	}

	public void printPhi(String topWordPath){
		File file = new File(topWordPath);
		try{
			file.getParentFile().mkdirs();
			file.createNewFile();
		} catch(IOException e){
			e.printStackTrace();
		}

		try{
			PrintWriter topWordWriter = new PrintWriter(file);

			for(int i=0; i < m_trainSet.size(); i++) {
				_Review doc = (_Review) m_trainSet.get(i);
				String itemID = doc.getItemID();
				String userID = doc.getUserID();

				topWordWriter.format("No. %d Doc(user: %s, item: %s) ***************\n", i,
						userID, itemID);
				for (int k = 0; k < number_of_topics; k++) {
					topWordWriter.format("%.5f\t", m_logSpace?Math.exp(doc.m_topics[k]):doc.m_topics[k]);
				}
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch(FileNotFoundException ex){
			System.err.println("File Not Found: " + topWordPath);
		}
	}

	public void printBeta(String betaFile){
		//print out prior parameter of topic-word distribution: beta
		File file = new File(betaFile);
		try{
			file.getParentFile().mkdirs();
			file.createNewFile();
		} catch(IOException e){
			e.printStackTrace();
		}
		try{
			PrintWriter betaWriter = new PrintWriter(file);
			for(int i = 0; i < m_corpus.getFeatureSize(); i++) //first line is vocabulary
				betaWriter.format("%s\t", m_corpus.getFeature(i));
			betaWriter.println();

			for (int i = 0; i < topic_term_probabilty.length; i++){//next is beta
				for(int j = 0; j < vocabulary_size; j++)
					betaWriter.format("%.10f\t",  m_logSpace ? Math.exp(topic_term_probabilty[i][j]) : topic_term_probabilty[i][j]);
				betaWriter.println();
			}
			betaWriter.close();
		} catch(FileNotFoundException ex){
			System.err.format("[Error]Failed to open file %s\n", betaFile);
		}
	}

	//_Doc has no userID, _Review has
	public HashMap<String, List<_Doc>> getDocByUser(){
		HashMap<String, List<_Doc>> docByUser = new HashMap<>();
		for(_Doc d:m_trainSet) {
            _Review doc = (_Review) d;
			String userName = doc.getUserID();
			if(!docByUser.containsKey(userName)){
				docByUser.put(userName, new ArrayList<_Doc>());
			}
			docByUser.get(userName).add(d);
		}
		return docByUser;
	}

	public HashMap<String, List<_Doc>> getDocByItem(){
		HashMap<String, List<_Doc>> docByItem = new HashMap<>();
		for(_Doc d:m_trainSet) {
			String itemName = d.getItemID();
			if(!docByItem.containsKey(itemName)){
				docByItem.put(itemName, new ArrayList<_Doc>());
			}
			docByItem.get(itemName).add(d);
		}
		return docByItem;
	}

	public void printTopWords(int k, String topWordPath, HashMap<String, List<_Doc>> docCluster) {
		File file = new File(topWordPath);
		try{
			file.getParentFile().mkdirs();
			file.createNewFile();
		} catch(IOException e){
			e.printStackTrace();
		}

		try{
			PrintWriter topWordWriter = new PrintWriter(file);

			for(Map.Entry<String, List<_Doc>> entryU : docCluster.entrySet()) {
				double[] gamma = new double[number_of_topics];
				Arrays.fill(gamma, 0);
				for(_Doc d:entryU.getValue()) {
					for (int i = 0; i < number_of_topics; i++)
						gamma[i] += m_logSpace ? Math.exp(d.m_topics[i]):d.m_topics[i];
				}
				Utils.L1Normalization(gamma);

				topWordWriter.format("ID %s(%d reviews)\n", entryU.getKey(), entryU.getValue().size());
				for (int i = 0; i < topic_term_probabilty.length; i++) {
					MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
					for (int j = 0; j < vocabulary_size; j++)
						fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));

					topWordWriter.format("-- Topic %d(%.5f):\t", i, gamma[i]);
					for (_RankItem it : fVector)
						topWordWriter.format("%s(%.5f)\t", it.m_name, m_logSpace ? Math.exp(it.m_value) : it.m_value);
					topWordWriter.write("\n");
				}
			}
			topWordWriter.close();
		} catch(FileNotFoundException ex){
			System.err.println("File Not Found: " + topWordPath);
		}
	}

}
