/**
 * 
 */
package structures;

/**
 * @author hongning
 *
 */
public class TokenizeResult {
	
	String[] m_rawTokens; // original raw unigrams
	String[] m_tokens;
	int m_stopwords;
	int m_originLength;
	
	public TokenizeResult(String[] rawTokens) {
		m_rawTokens = rawTokens;
		m_originLength = rawTokens.length;
		m_tokens = null;
		m_stopwords = 0;
	}
	
	public void setTokens(String[] tokens) {
		m_tokens = tokens;
	}
	
	public String[] getTokens() {
		return m_tokens;
	}
	
	public String[] getRawTokens() {
		return m_rawTokens;
	}

	public void incStopwords() {
		m_stopwords ++;
	}
	
	public int getStopwordCnt() {
		return m_stopwords;
	}
	
	public int getRawCnt() {
		return m_originLength;
	}
	
	public double getStopwordProportion() {
		return (double)m_stopwords / m_originLength;
	}
}
