from collections import defaultdict
import logging

logger = logging.getLogger("comments")

customer_comments = defaultdict(list)
employee_comments = defaultdict(list)
ai_comments = defaultdict(list)

        
def add_employee_comment(employee_id, comment):
    employee_comments[employee_id].append(comment)
    logger.info(f'Employee: {employee_id}: {comment}')
        
def add_customer_comments(customer_id, comment):
    customer_comments[customer_id].append(comment)
    logger.info(f'Customer: {customer_id}: {comment}')
    
def add_ai_comments(recommendation_id, comment):
    ai_comments[recommendation_id].append(comment)
    logger.info(f'AI: {recommendation_id}: {comment}')
    
def reset_comments():
    employee_comments.clear()
    customer_comments.clear()
    ai_comments.clear()
    
def get_customer_comments(customer_id):
    return customer_comments[customer_id]

def get_employee_comments(employee_id):
    return employee_comments[employee_id]

def get_ai_comments(recommendation_id):
    return ai_comments[recommendation_id]

