import json
import argparse
import mysql.connector

def get_data(jpath="db_connection.json"):
    with open(jpath) as json_file:
        db_info = json.load(json_file)

    db = mysql.connector.connect(host=db_info['host'], user=db_info['user'],\
                    passwd=db_info['passwd'], db=db_info['db'])

    cur = db.cursor()
    cur.execute("select * from mr;")

    data = cur.fetchall()
    db.close()
    return data
    
