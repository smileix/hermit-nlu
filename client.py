import requests
import json
import argparse

parser = argparse.ArgumentParser(description='Mummer NLU client')
parser.add_argument('-i', '--ip', type=str, default='127.0.0.1', help='Server ip')
parser.add_argument('-p', '--port', type=str, default='9876', help='Server port')
args = parser.parse_args()

SERVER_IP = args.ip
SERVER_PORT = args.port

while 1:
    sentence = raw_input('Enter a sentence: ')
    to_send = {"sentence": sentence}
    try:
        r = requests.post('http://' + SERVER_IP + ':' + SERVER_PORT + '/predict', json=to_send, timeout=2)
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2, sort_keys=True))
        else:
            print('An error occurred: unable to parse the sentence')
    except requests.exceptions.ConnectionError:
        print('Server ' + SERVER_IP + ':' + SERVER_PORT + ' not reachable!')
