import json
import argparse
import mysql.connector
import pandas as pd

def get_data(jpath="db_connection.json", test=False):
    with open(jpath) as json_file:
        db_info = json.load(json_file)

    db = mysql.connector.connect(host=db_info['host'], user=db_info['user'],\
                    passwd=db_info['passwd'], db=db_info['db'])

    cur = db.cursor()
    if test:
        cur.execute("select * from mr_test;")
    else:
        cur.execute("select * from mr;")

    data = cur.fetchall()
    
    ret_list = []
    tmp_dict = {'id':0, 'doc':'', 'label':0}
    for id_doc_label in data:
        txt = id_doc_label[1].replace("[^가-힣 ]", "")
        if txt == '':
            continue
        else:
            tmp_dict['doc'] = txt

        tmp_dict['id'] = id_doc_label[0]
        tmp_dict['label'] = id_doc_label[2]
        ret_list.append(tmp_dict)
        tmp_dict = {'id':0, 'doc':'', 'label':0}

    db.close()
    del tmp_dict, data
    return ret_list
    
if __name__=="__main__":
    get_data()    # for test