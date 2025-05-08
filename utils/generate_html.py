from html import escape as html_escape

def generate_html(input_list1, input_list2):
    """
    Converts a Python list of strings into a formatted HTML list.

    Args:
        input_list (list): List of strings to convert
        ordered (bool): True for ordered list (numbers), False for unordered (bullets)
        title (str): Optional title/heading for the list

    Returns:
        str: Formatted HTML string
    """
    # Escape HTML special characters in all list items
    escaped_items1 = [html_escape(item) for item in input_list1]
    escaped_items2 = [html_escape(item) for item in input_list2]

    # Create list items
    list_items1 = "\n".join([f"    <li>{item}</li>" for item in escaped_items1])
    list_items2 = "\n".join([f"    <li>{item}</li>" for item in escaped_items2])

    # HTML template with embedded CSS styling
    html_template = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
    .list-container {{
        max-width: 600px;
        margin: 20px auto;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .custom-list {{
        font-family: Arial, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
        padding-left: 30px;
    }}

    .custom-list li {{
        margin-bottom: 8px;
        padding: 6px 0;
        border-bottom: 1px solid #eee;
    }}

    .custom-list li:last-child {{
        border-bottom: none;
    }}

    .custom-list li:hover {{
        background-color: #f1f1f1;
        transition: background-color 0.2s;
    }}
    </style>
    </head>
    <body>
    <div class="list-container">
    <ul class="custom-list">
    {list_items1}
    </ul>
    <br />
    <br />
    <ul class="custom-list">
    {list_items2}
    </ul>
    </div>
    </body>
    </html>"""

    return html_template
