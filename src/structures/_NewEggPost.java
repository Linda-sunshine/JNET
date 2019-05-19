package structures;

import json.JSONObject;
import utils.Utils;

public class _NewEggPost extends _Post {
	String m_prodId;
	public String getProdId() {
		return m_prodId;
	}

	public void setProdId(String prodId) {
		this.m_prodId = prodId;
	}

	String m_proContent;
	public String getProContent() {
		if (m_proContent==null || m_proContent.isEmpty())
			return null;
		return m_proContent;
	}

	public void setProContent(String proContent) {
		this.m_proContent = proContent;
	}

	String m_conContent;
	public String getConContent() {
		if (m_conContent==null || m_conContent.isEmpty())
			return null;
		return m_conContent;
	}

	public void setConContent(String conContent) {
		this.m_conContent = conContent;
	}
	
	String m_comments;	
	public String getComments() {
		if (m_comments==null || m_comments.isEmpty())
			return null;
		return m_comments;
	}

	public void setComments(String comments) {
		this.m_comments = comments;
	}

	public _NewEggPost(JSONObject json, String prodId) {	
		super(json);
		try {//special treatment for the overall ratings
			if (json.has("Rating")){				
				double label = json.getInt("Rating");
				if(label <= 0)
					setLabel(1);
				else if (label>5)
					setLabel(5);
				else 
					setLabel((int)label);
			}
		} catch (Exception e) {
			setLabel(1);
		}
		
		setDate(Utils.getJSONValue(json, "PublishDate"));
		setProContent(Utils.getJSONValue(json, "Pros"));
		setConContent(Utils.getJSONValue(json, "Cons"));
		setComments(Utils.getJSONValue(json, "Comments"));
		setAuthor(Utils.getJSONValue(json, "LoginNickName"));
		setProdId(prodId);
		
		m_ID = prodId + "-" + m_author;
	}	
}
