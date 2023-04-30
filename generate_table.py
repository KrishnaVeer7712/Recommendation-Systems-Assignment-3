import os
from flask import Flask, request,render_template
app = Flask(__name__)

templates_dir = os.path.join(app.root_path, 'templates')
import pandas as pd
data=pd.read_csv('depression_dataset.csv', encoding='latin-1')
cols = ["Target", "IDs", "Date", "Flag", "User", "TweetText"]
data.columns = cols
data=data.iloc[:100, :]
# Set css for html table
css="{{ url_for('static',filename='css/df_style.css') }}"
pd.set_option('colheader_justify', 'center')
# HTML Table for file data
html_string = '''
                  <head>
                      <link rel="stylesheet" type="text/css" href="{css}">
                  </head>
                  <body>
                    {table}
                  </body>
                
                '''
html_table_data = html_string.format(table=data.to_html(classes='mystyle'),css=css)
# Output an HTML file
with open(os.path.join(templates_dir,"data.html"), 'w', encoding="utf-8") as f:
    f.write(html_table_data)
    f.close()
print(data)