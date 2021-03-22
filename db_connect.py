import json
import argparse
import mysql.connector

def main(args):
    with open(args.jpath) as json_file:
        db_info = json.load(json_file)

    db = mysql.connector.connect(host=db_info['host'], user=db_info['user'],\
                    passwd=db_info['passwd'], db=db_info['db'])

    cur = db.cursor()
    cur.execute("select * from mr;")

    data = cur.fetchall()
    db.close()
    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpath',
                        default="db_connection.json",
                        help="json path for retrieve db connection info.")
    args = parser.parse_args()
    main(args)
    
