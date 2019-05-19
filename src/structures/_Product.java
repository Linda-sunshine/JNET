package structures;

import json.JSONObject;
import utils.Utils;

public class _Product {
	String m_ID;
	public String getID() {
		return m_ID;
	}

	public void setID(String ID) {
		this.m_ID = ID;
	}

	String m_price;
	public String getPrice() {
		return m_price;
	}

	public void setPrice(String price) {
		this.m_price = price;
	}

	String m_features;
	public String getFeatures() {
		return m_features;
	}

	public void setFeatures(String features) {
		this.m_features = features;
	}

	String m_name;	
	public String getName() {
		return m_name;
	}

	public void setName(String name) {
		this.m_name = name;
	}
	
	//Constructor.
	public _Product(String ID) {
		m_ID = ID;
	}
	
	//Constructor.
	public _Product(JSONObject json) throws NumberFormatException {
		setPrice(Utils.getJSONValue(json, "Price"));
		setFeatures(Utils.getJSONValue(json, "Features"));
		setName(Utils.getJSONValue(json, "Name"));
		setID(Utils.getJSONValue(json, "ProductID"));
	}
}
