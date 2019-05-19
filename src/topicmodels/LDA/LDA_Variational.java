/**
 * 
 */
package topicmodels.LDA;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import structures.*;
import topicmodels.pLSA.pLSA;
import utils.Utils;

/**
 * @author hongning
 * Variational sampling for Latent Dirichlet Allocation model
 * Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." 
 */
public class LDA_Variational extends pLSA {

	// parameters to control variational inference
	protected int m_varMaxIter;
	protected double m_varConverge;
	
	protected double[] m_alpha; // we can estimate a vector of alphas as in p(\theta|\alpha)
	protected double[] m_alphaStat; // statistics for alpha estimation
	protected double[] m_alphaG; // gradient for alpha
	protected double[] m_alphaH; // Hessian for alpha
	
	public LDA_Variational(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, 
			int number_of_topics, double alpha, int varMaxIter, double varConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha);
		
		m_varConverge = varConverge;
		m_varMaxIter = varMaxIter;		
		
		m_logSpace = true;
	}
	
	@Override
	protected void createSpace() {
		super.createSpace();
		
		m_alpha = new double[number_of_topics];
		m_alphaStat = new double[number_of_topics];
		m_alphaG = new double[number_of_topics];
		m_alphaH = new double[number_of_topics];
		
		Arrays.fill(m_alpha, d_alpha);
	}
	
	@Override
	public String toString() {
		return String.format("LDA[k:%d, alpha:%.2f, beta:%.2f, Variational]", number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		// initialize with all smoothing terms
		init();
		Arrays.fill(m_alpha, d_alpha);
		
		// initialize topic-word allocation, p(w|z)
		for(_Doc d:collection) {
			d.setTopics4Variational(number_of_topics, d_alpha);//allocate memory and randomize it
			collectStats(d);
		}
		
		calculate_M_step(0);
	}
	
	@Override
	protected void init() {//will be called at the beginning of each EM iteration
		// initialize alpha statistics
		Arrays.fill(m_alphaStat, 0);
		
		// initialize with all smoothing terms
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta-1.0);
		imposePrior();
	}
	
	protected void collectStats(_Doc d) {
		_SparseFeature[] fv = d.getSparse();
		int wid;
		double v; 
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++)
				word_topic_sstat[i][wid] += v*d.m_phi[n][i];
		}
		
		//if we need to use maximum likelihood to estimate alpha
		double diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
		for(int i=0; i<number_of_topics; i++)
			m_alphaStat[i] += Utils.digamma(d.m_sstat[i]) - diGammaSum;
	}

	@Override
	protected void initTestDoc(_Doc d) {
		d.setTopics4Variational(number_of_topics, d_alpha);
	}
	
	@Override
	public double calculate_E_step(_Doc d) {	
		double last = 1;		
		if (m_varConverge>0)
			last = calculate_log_likelihood(d);
		
		double current = last, converge, logSum, v;
		int iter = 0, wid;
		_SparseFeature[] fv = d.getSparse();
		
		do {
			//variational inference for p(z|w,\phi)
			for(int n=0; n<fv.length; n++) {
				wid = fv[n].getIndex();
				v = fv[n].getValue();
				for(int i=0; i<number_of_topics; i++)
					d.m_phi[n][i] = topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]);
				
				logSum = Utils.logSumOfExponentials(d.m_phi[n]);
				for(int i=0; i<number_of_topics; i++)
					d.m_phi[n][i] = Math.exp(d.m_phi[n][i] - logSum);
			}
			
			//variational inference for p(\theta|\gamma)
			System.arraycopy(m_alpha, 0, d.m_sstat, 0, m_alpha.length);
			for(int n=0; n<fv.length; n++) {
				v = fv[n].getValue();
				for(int i=0; i<number_of_topics; i++)
					d.m_sstat[i] += d.m_phi[n][i] * v;
			}
			
			if (m_varConverge>0) {
				current = calculate_log_likelihood(d);			
				converge = Math.abs((current - last)/last);
				last = current;
				
				if (converge<m_varConverge)
					break;
			}
		} while(++iter<m_varMaxIter);		
		
		if (m_collectCorpusStats) {
			collectStats(d);//collect the sufficient statistics after convergence
		 	return current;
		} else if (m_varConverge>0)
			return current;//to avoid computing this again
		else
			return calculate_log_likelihood(d);//in testing, we need to compute log-likelihood
	}
	
	@Override
	public void calculate_M_step(int iter) {	
		//maximum likelihood estimation of p(w|z,\beta)
		for(int i=0; i<number_of_topics; i++) {
			double sum = Utils.sumOfArray(word_topic_sstat[i]);
			for(int v=0; v<vocabulary_size; v++) //will be in the log scale!!
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]/sum);
		}

		//we need to estimate p(\theta|\alpha) as well later on
		int docSize = getCorpusSize(), i = 0;
		double alphaSum, diAlphaSum, z, c, c1, c2, diff, deltaAlpha;
		do {
			alphaSum = Utils.sumOfArray(m_alpha);
			diAlphaSum = Utils.digamma(alphaSum);
			z = docSize * Utils.trigamma(alphaSum);
			
			c1 = 0; c2 = 0;
			for(int k=0; k<number_of_topics; k++) {
				m_alphaG[k] = docSize * (diAlphaSum - Utils.digamma(m_alpha[k])) + m_alphaStat[k];
				m_alphaH[k] = -docSize * Utils.trigamma(m_alpha[k]);
				
				c1 +=  m_alphaG[k] / m_alphaH[k];
				c2 += 1.0 / m_alphaH[k];
			}			
			c = c1 / (1.0/z + c2);
			
			diff = 0;
			for(int k=0; k<number_of_topics; k++) {
				deltaAlpha = (m_alphaG[k]-c) / m_alphaH[k];
				m_alpha[k] -= 0.001 * deltaAlpha; // set small stepsize, so the value won't jump too much
				diff += deltaAlpha * deltaAlpha;
			}
			diff /= number_of_topics;
		} while(++i<m_varMaxIter && diff>m_varConverge);

		// update per-document topic distribution vectors
		finalEst();
	}
	
	protected int getCorpusSize() {
		return m_trainSet.size();
	}
	
	@Override
	protected void finalEst() {	
		//estimate p(z|d) from all the collected samples
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
	}
	
	@Override
	public double calculate_log_likelihood(_Doc d) {
		int wid;
		double[] diGamma = new double[this.number_of_topics];
		double logLikelihood = Utils.lgamma(Utils.sumOfArray(m_alpha)) - Utils.lgamma(Utils.sumOfArray(d.m_sstat)), v;
		double diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
		for(int i=0; i<number_of_topics; i++) {
			diGamma[i] = Utils.digamma(d.m_sstat[i]) - diGammaSum;
			logLikelihood += Utils.lgamma(d.m_sstat[i]) - Utils.lgamma(m_alpha[i])
					+ (m_alpha[i] - d.m_sstat[i]) * diGamma[i];
		}
		
		//collect the sufficient statistics
		_SparseFeature[] fv = d.getSparse();
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++) 
				logLikelihood += v * d.m_phi[n][i] * (diGamma[i] + topic_term_probabilty[i][wid] - Math.log(d.m_phi[n][i]));
		}

		return logLikelihood;
	}


	@Override
	protected void estThetaInDoc(_Doc doc) {
		double sum = 0;
		Arrays.fill(doc.m_topics, 0);

		_SparseFeature[] fv = doc.getSparse();
		for (int n = 0; n < fv.length; n++) {
			double v = fv[n].getValue();
			for(int i=0; i < number_of_topics; i++){
				doc.m_topics[i] += v*doc.m_phi[n][i];//here should multiply v
			}
		}

		sum = Utils.sumOfArray(doc.m_topics);
		for(int i=0; i < number_of_topics; i++){
			if (m_logSpace){
				doc.m_topics[i] = Math.log(doc.m_topics[i]/sum);
			}else{
				doc.m_topics[i] = doc.m_topics[i]/sum;
			}
		}
	}
	
	// perform inference of topic distribution in the document
	@Override
	public double inference(_Doc d) {
		initTestDoc(d);		
		double likelihood = calculate_E_step(d);
		estThetaInDoc(d);
		return calc_term_log_likelihood(d);
	}


	public double calc_term_log_likelihood(_Doc d) {
		int wid;
		double v, logLikelihood = 0;

		//collect the sufficient statistics
		_SparseFeature[] fv = d.getSparse();
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++)
				logLikelihood += v * d.m_phi[n][i] * topic_term_probabilty[i][wid];
		}

		return logLikelihood;
	}

	@Override
	public void printParameterAggregation(int k, String folderName, String topicmodel) {
		super.printParameterAggregation(k, folderName, topicmodel);

        String gammaPathByUser = String.format("%s%s_postByUser_%d.txt", folderName, topicmodel, number_of_topics);
        String gammaPathByItem = String.format("%s%s_postByItem_%d.txt", folderName, topicmodel, number_of_topics);
        printAggreTopWords(k, gammaPathByUser, getDocByUser());
        printAggreTopWords(k, gammaPathByItem, getDocByItem());

		printParam(folderName, topicmodel);
	}

    public void printAggreTopWords(int k, String topWordPath, HashMap<String, List<_Doc>> docCluster) {
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
                    for (int i = 0; i < number_of_topics; i++) {
                        gamma[i] += d.m_sstat[i];
                    }
                }
                for(int i = 0; i < number_of_topics; i++){
                	gamma[i] /= entryU.getValue().size();
				}

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
			System.err.format("[Error]Failed to open file %s\n", topWordPath);
        }
    }

	public void printParam(String folderName, String topicmodel){
		String priorAlphaPath = String.format("%s%s_priorAlpha_%d.txt", folderName, topicmodel, number_of_topics);
		String postGammaPath = String.format("%s%s_postGamma_%d.txt", folderName, topicmodel, number_of_topics);

		//print out prior parameter of dirichlet: alpha
		File file = new File(priorAlphaPath);
		try{
			file.getParentFile().mkdirs();
			file.createNewFile();
		} catch(IOException e){
			e.printStackTrace();
		}
		try{
			PrintWriter alphaWriter = new PrintWriter(file);
			for (int i = 0; i < number_of_topics; i++)
				alphaWriter.format("%.5f\t", this.m_alpha[i]);
			alphaWriter.close();
		} catch(FileNotFoundException ex){
			System.err.format("[Error]Failed to open file %s\n", priorAlphaPath);
		}

		//print out posterior parameter of dirichlet for each document: gamma
		file = new File(postGammaPath);
		try{
			file.getParentFile().mkdirs();
			file.createNewFile();
		} catch(IOException e){
			e.printStackTrace();
		}
		try{
			PrintWriter gammaWriter = new PrintWriter(file);

			for(int idx = 0; idx < m_trainSet.size(); idx++) {
				gammaWriter.write(String.format("No. %d Doc(user: %s, item: %s) ***************\n", idx,
                        ((_Doc4ETBIR) m_trainSet.get(idx)).getUserID(),
                        ((_Doc4ETBIR) m_trainSet.get(idx)).getItemID()));
				for (int i = 0; i < number_of_topics; i++)
					gammaWriter.format("%.5f\t", ((_Doc4ETBIR) m_trainSet.get(idx)).m_sstat[i]);
				gammaWriter.println();
			}
			gammaWriter.close();
		} catch(FileNotFoundException ex){
			System.err.format("[Error]Failed to open file %s\n", postGammaPath);
		}
	}

	// save each document's phi and the beta learned from all the documents
	public void printPhi(String dir){
		try{
			String phiFileName = String.format("%s/Phi.txt", dir);
			PrintWriter writer = new PrintWriter(new File(phiFileName));
			for(_Doc d: m_trainSet){
				_Review r = (_Review) d;
				writer.write("-----\n");
				writer.format("%s\t%s\t%d\t%d\n", r.getUserID(), r.getID(), r.m_phi.length, number_of_topics);
				for(double[] phi: r.m_phi){
					for(double v: phi){
						writer.format("%.5f\t", v);
					}
					writer.write("\n");
				}
			}
			writer.close();

		} catch(IOException e){
			e.printStackTrace();
		}
	}

	// save each document's phi and the beta learned from all the documents
	public void printBeta(String dir){
		try{
			String betaFileName = String.format("%s/Beta.txt", dir);
			PrintWriter writer = new PrintWriter(new File(betaFileName));
			writer = new PrintWriter(new File(betaFileName));
			writer.format("%d\t%d\n", number_of_topics, vocabulary_size);
			for(double[] topic: topic_term_probabilty){
				for(double v: topic){
					writer.format("%.2f\t", v);
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	public void printGamma(String dir){
		try{
			String gammaFileName = String.format("%s_gamma.txt", dir);
			PrintWriter writer = new PrintWriter(new File(gammaFileName));
			for(_Doc d: m_trainSet){
				_Review r = (_Review) d;
				writer.format("%s\t%s\t%d\t", r.getUserID(), r.getID(), r.m_sstat.length);
				for(double stat: r.m_sstat){
					writer.format("%.5f\t", stat);
				}
				writer.write("\n");
				writer.write(r.getSource()+"\n");
			}
			writer.close();
			String sourceFileName = String.format("%s_source.txt", dir);
			writer = new PrintWriter(new File(sourceFileName));
			for(_Doc d: m_trainSet){
				_Review r = (_Review) d;
				writer.format("%s\t%s\t%d\t", r.getUserID(), r.getID(), r.m_sstat.length);
				writer.write(r.getSource()+"\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
