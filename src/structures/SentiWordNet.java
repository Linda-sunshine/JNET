package structures;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.Normalizer;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import utils.Utils;

public class SentiWordNet {

	Map<String, Double> dictionary;
	private SnowballStemmer m_stemmer;
	
	public boolean contains(String term) {
		return dictionary.containsKey(term);
	}
	
	public double score(String term) {
		return dictionary.get(term);
	}

	public String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return token;
	}
	
	public String Normalize(String token){
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		
		if (Utils.isNumber(token))
			return "NUM";
		else
			return token;
	}
	
	public SentiWordNet(String pathToSWN) throws IOException {
		
		m_stemmer = new englishStemmer();
		// This is our main dictionary representation
		dictionary = new HashMap<String, Double>();

		// From String to list of doubles.
		HashMap<String, HashMap<Integer, Double>> tempDictionary = new HashMap<String, HashMap<Integer, Double>>();

		BufferedReader csv = null;
		try {
			csv = new BufferedReader(new FileReader(pathToSWN));
			int lineNumber = 0;

			String line;
			while ((line = csv.readLine()) != null) {
				lineNumber++;

				// If it's a comment, skip this line.
				if (!line.trim().startsWith("#")) {
					// We use tab separation
					String[] data = line.split("\t");
					String wordTypeMarker = data[0];

					// Example line:
					// POS ID PosS NegS SynsetTerm#sensenumber Desc
					// a 00009618 0.5 0.25 spartan#4 austere#3 ascetical#2
					// ascetic#2 practicing great self-denial;...etc

					// Is it a valid line? Otherwise, through exception.
					if (data.length != 6) {
						throw new IllegalArgumentException("Incorrect tabulation format in file, line: "+ lineNumber);
					}

					// Calculate synset score as score = PosS - NegS
					Double synsetScore = Double.parseDouble(data[2])
							- Double.parseDouble(data[3]);

					// Get all Synset terms
					String[] synTermsSplit = data[4].split(" ");

					// Go through all terms of current synset.
					for (String synTermSplit : synTermsSplit) {
						// Get synterm and synterm rank
						String[] synTermAndRank = synTermSplit.split("#");
						String synTerm = synTermAndRank[0] + "#"
								+ wordTypeMarker;

						int synTermRank = Integer.parseInt(synTermAndRank[1]);
						// What we get here is a map of the type:
						// term -> {score of synset#1, score of synset#2...}

						// Add map to term if it doesn't have one
						if (!tempDictionary.containsKey(synTerm)) {
							tempDictionary.put(synTerm,
									new HashMap<Integer, Double>());
						}

						// Add synset link to synterm
						tempDictionary.get(synTerm).put(synTermRank,
								synsetScore);
					}
				}
			}

			// Go through all the terms.
			Set<String> synTerms = tempDictionary.keySet();
			for (String synTerm : synTerms) {
				double score = 0;
				int count = 0;
				HashMap<Integer, Double> synSetScoreMap = tempDictionary.get(synTerm);
				Collection<Double> scores = synSetScoreMap.values();
				for (double s : scores) {
					if (s != 0) {
						score += s;
						count++;
					}
					if (score != 0)
						score = (double) score / count;
				}
				String[] termMarker = synTerm.split("#");
				dictionary.put(SnowballStemming(Normalize(termMarker[0])) + "#" + termMarker[1], score);
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (csv != null) {
				csv.close();
			}
		}
	}

	
	public double extract(String word, String pos) {
		
		if(dictionary.containsKey(word + "#" + pos) )
			return dictionary.get(word + "#" + pos);
		else 
			return -2; // -2 means no score basically
	}
	
	public static void main(String [] args) throws IOException {
		String pathToSWN = "./data/Model/SentiWordNet_3.0.0_20130122.txt";
		SentiWordNet sentiwordnet = new SentiWordNet(pathToSWN);
		System.out.println("work#n "+sentiwordnet.extract("opaque", "v"));
		System.out.println("bad#n "+sentiwordnet.extract("bad", "n"));
		System.out.println("blue#a "+sentiwordnet.extract("blue", "a"));
		System.out.println("blue#n "+sentiwordnet.extract("blue", "n"));
	}
}