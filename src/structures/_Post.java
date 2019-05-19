package structures;

import java.text.DateFormat;
import java.text.ParseException;

import json.JSONException;
import json.JSONObject;
import utils.Utils;

/**
 * @author hongning
 * @version 0.1
 * @category data structure
 * data structure for a forum discussion post, and it can also be used in different scenarios for document processing
 */
public class _Post {
	//unique ID from the corresponding website
	String m_ID;
	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;
	public String getAuthor() {
		return m_author;
	}
	public void setAuthor(String author) {
		this.m_author = author;
	}

	//unique author ID from the corresponding website
	String m_authorID;	
	public String getAuthorID() {
		return m_authorID;
	}
	public void setAuthorID(String authorID) {
		this.m_authorID = authorID;
	}

	//post title (might not be available in some medical forums)
	String m_title;//not available in WebMD
	public String getTitle() {
		return m_title;
	}
	public void setTitle(String title) {
		if (!title.isEmpty())
			this.m_title = title;
	}

	//post content
	String m_content;
	public String getContent() {
		return m_content;
	}
	public void setContent(String content) {
		if (!content.isEmpty()) {
			this.m_content = Utils.cleanHTML(content);
		}
	}

	//timestamp of the post
	String m_date;
	public String getDate() {
		return m_date;
	}
	public void setDate(String date) {
		this.m_date = date;
	}
	
	//post ID that this post is reply to
	String m_replyToID;
	public String getReplyToID() {
		return m_replyToID;
	}
	public void setReplyToID(String replyToID) {
		this.m_replyToID = replyToID;
	}
	
	//only used in eHealth to keep track of reply-to relation
	int m_level; 
	public int getLevel() {
		return m_level;
	}
	public void setLevel(int level) {
		this.m_level = level;
	}
	
	//Used for classification.
	int m_label;
	public void setLabel(int overall){
		this.m_label = overall;
	}
	public int getLabel(){
		return this.m_label;
	}
	
	//Constructor.
	public _Post(JSONObject json) {
		try {//special treatment for the overall ratings
			if (json.has("Overall")){
				if(json.getString("Overall").equals("None")) {
					System.out.print('R');
					setLabel(-1);
				} else{
					double label = json.getDouble("Overall");
					if(label <= 0)
						setLabel(1);
					else if (label>5)
						setLabel(5);
					else 
						setLabel((int)label);
				}
			}
		} catch (Exception e) {
		}
		
		setDate(Utils.getJSONValue(json, "Date"));
		setContent(Utils.getJSONValue(json, "Content"));
		setTitle(Utils.getJSONValue(json, "Title"));
		m_ID = Utils.getJSONValue(json, "ReviewID");
		setAuthor(Utils.getJSONValue(json, "Author"));
	}
	
	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("postID", m_ID);//must contain
		json.put("author", m_author);//must contain
		json.put("authorID", m_authorID);//must contain
		json.put("replyTo", m_replyToID);//might be missing
		json.put("date", m_date);//must contain
		json.put("title", m_title);//might be missing
		json.put("content", m_content);//must contain
		return json;
	}
	
	//check format for each post
	public boolean isValid(DateFormat dateFormatter) {
		if (getLabel() <= 0 || getLabel() > 5){
			//System.err.format("[Error]Missing Lable or wrong label!!");
			System.err.print('L');
			return false;
		}
		else if (getContent() == null){
			//System.err.format("[Error]Missing content!!\n");
			System.err.print('C');
			return false;
		}	
		else if (getDate() == null){
			//System.err.format("[Error]Missing date!!\n");
			System.out.print('d');
			return false;
		}
		else {
			// to check if the date format is correct
			try {
				dateFormatter.parse(getDate());
				return true;
			} catch (ParseException e) {
				System.err.print('D');
			}
			return true;
		} 
	}
}
