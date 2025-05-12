from collections import defaultdict
import logging
import numpy as np
logger = logging.getLogger("comments")

customer_comments = defaultdict(list)
employee_comments = defaultdict(list)
employee_customer_comments = defaultdict(list)
ai_comments = defaultdict(list)

        
def add_employee_comment(employee_id, comment):
    employee_comments[employee_id].append(comment)
    logger.info(f'Employee: {employee_id}: {comment}')
    
def add_employee_customer_comment(employee_id, customer_id, comment):
    employee_customer_comments[f"{employee_id}{customer_id}"].append(comment)
    logger.info(f'Employee: {employee_id} and Customer: {customer_id} comment: {comment}')
        
def add_customer_comment(customer_id, comment):
    customer_comments[customer_id].append(comment)
    logger.info(f'Customer: {customer_id}: {comment}')
    
def add_ai_comment(recommendation_id, comment):
    ai_comments[recommendation_id].append(comment)
    logger.info(f'AI: {recommendation_id}: {comment}')
    
def add_abnormality_comment(recommendation_id, shap_values, datapoint, feature_names):
    
    # get index of the two highest shap values in shap_values
    top_indices = np.argsort(shap_values)[-2:][::-1]
    
    print(top_indices)
    print(feature_names)
    
    # get the feature names of the top two shap values
    top_features = [feature_names[i] for i in top_indices]
    
    # get the values of the top two shap values
    top_values = [datapoint[i] for i in top_indices]
    
    comment = ""
    for i in range(len(top_features)):
        value = top_values[i]
        # if the value is a boolean, convert it to Ja (True) or Nein (False)
        if isinstance(value, bool):
            value = "Ja" if value else "Nein"
        comment += f"{top_features[i]}: {value} \n"
    
    comment = f"Diese Empfehlung ist eher unüblich. Folgende Faktoren sind dafür ausschlaggebend: {comment}"
    
    add_ai_comment(recommendation_id, comment)
    
def reset_comments():
    employee_comments.clear()
    customer_comments.clear()
    ai_comments.clear()
    employee_customer_comments.clear()
    
def get_customer_comments(customer_id):
    return customer_comments[customer_id]

def get_employee_comments(employee_id):
    return employee_comments[employee_id]

def get_ai_comments(recommendation_id):
    return ai_comments[recommendation_id]

def get_employee_customer_comment(employee_id, customer_id):
    return employee_customer_comments[f"{employee_id}{customer_id}"][:5]