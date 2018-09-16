import pandas as pd
import re
import json
from keras.preprocessing.text import Tokenizer

csv_data = pd.read_csv("../data/train_data.csv", encoding="ISO-8859-1", delimiter=';', header=None)
json_data = pd.read_json("../data/kvret_train_public.json", encoding="ISO-8859-1")

# Knowledgebase creation
colnames = []
for i in range(len(json_data)):
    colnames.append(json_data["scenario"][i]['kb']['column_names'])
# tuples for drive
tuples_d = [('poi', 'address', 'val'), ('poi', 'traffic_info', 'val'), ('poi', 'distance', 'val')]
# tuples for calendar
tuples_c = [('event', 'time', 'val'), ('event', 'room', 'val'), ('event', 'party', 'val'), ('event', 'agenda', 'val'),
            ('event', 'date', 'val')]
# tuples for weather
tuples_w = [('location', 'monday', 'val'), ('location', 'tuesday', 'val'), ('location', 'wednesday', 'val'),
            ('location', 'thursday', 'val'), ('location', 'friday', 'val'), ('location', 'saturday', 'val'),
            ('location', 'sunday', 'val')]
kb = []
for i in range(len(json_data)):
    kb_i = []
    if json_data['scenario'][i]['kb']['items']:
        if json_data['scenario'][i]['kb']['column_names'][0] == "poi":
            for j in range(len(json_data['scenario'][i]['kb']['items'])):
                for k in range(len(tuples_d)):
                    tup = [json_data['scenario'][i]['kb']['items'][j][tuples_d[k][0]], tuples_d[k][1],
                           json_data['scenario'][i]['kb']['items'][j][tuples_d[k][1]]]
                    kb_i.append(tup)
            pass
        elif json_data['scenario'][i]['kb']['column_names'][0] == "event":
            for j in range(len(json_data['scenario'][i]['kb']['items'])):
                for k in range(len(tuples_c)):
                    tup = [json_data['scenario'][i]['kb']['items'][j][tuples_c[k][0]], tuples_c[k][1],
                           json_data['scenario'][i]['kb']['items'][j][tuples_c[k][1]]]
                    kb_i.append(tup)
            pass
        else:
            for j in range(len(json_data['scenario'][i]['kb']['items'])):
                tup = []
                for k in range(len(tuples_w)):
                    tup = [json_data['scenario'][i]['kb']['items'][j][tuples_w[k][0]], tuples_w[k][1],
                           json_data['scenario'][i]['kb']['items'][j][tuples_w[k][1]]]
                    kb_i.append(tup)
            pass
    kb.append(kb_i)

x = {'subject': [], 'relation': [], 'object': []}
for i in kb:
    for j in i:
        x['subject'].append(j[0])
        x['object'].append(j[2])
        x['relation'].append(j[1])
x = pd.DataFrame(x)
x.drop_duplicates(inplace=True)
x.to_csv("kbtuples.csv")
nkb = {'relation': x['relation'], 'subject': x['subject'], 'object': x['subject'] + "_" + x['relation']}
nkb = pd.DataFrame(nkb)
nkb.drop_duplicates(inplace=True)
nkb.to_csv('normalised_kbtuples.csv')
print(len(nkb))
# Canonical representations
objects = []
for kb_i in kb:
    for ki in kb_i:
        objects.append('_'.join(ki[0].split(" ")) + '_' + ki[1])

chats = []
chats_complete = []
last_chat = ""
chat = []
for index, dialog in enumerate(csv_data[1]):
    if not dialog.startswith(last_chat):
        chats.append(chat)
        chat = []
        last_chat = ""
    try:
        dialog = dialog.strip('"').lower()
        new_part = dialog[(len(last_chat)):].strip('"').lower()
        # new_part = re.sub(last_chat, "", dialog).strip('"').lower()
        new_response = csv_data[2][index].strip('"').lower()
        chat.append(new_part)
        chats_complete.append(new_part)
        chat.append(new_response)
        chats_complete.append(new_response)
        last_chat += " " + new_part + "  " + new_response
        last_chat = last_chat.strip()
    except AttributeError:
        print("Empty dialog detected: " + dialog[(len(last_chat)):] + "\n Response: "
              + str(csv_data[2][index]))
chats.append(chat)
chat = []

# Preprocessing replacing values with their canonical representations
count = 0
poi = "odsfh8gr3w8z9febufwebefBUFUOfEHO(hf8ewfebubufesbuofwbuofzuUDVbu"
for i, (chat, kb_i) in enumerate(zip(chats, kb)):
    for j, ch in enumerate(chat):
        for ki in kb_i:
            if ki[0].lower() in ch:
                poi = ki[0].lower()
    for j, ch in enumerate(chat):
        for ki in kb_i:
            if ki[0].lower() == poi:
                if 'day' in ki[1].lower():
                    for kki in ki[1].lower().split(","):
                        if kki in ch and ki[2].lower() != 'home':
                            count = count + 1
                            chats[i][j] = re.sub(kki, '_'.join(ki[0].split(" ")) + '_' + ki[1], ch)
                if ki[2].lower() in ch and ki[2].lower() != 'home':
                    count = count + 1
                    chats[i][j] = re.sub(ki[2].lower(), '_'.join(ki[0].split(" ")) + '_' + ki[1], ch)
print(count)

# Making sure to have even number of dialogues in each chat
for i in range(len(chats)):
    if len(chats[i]) % 2 != 0:
        chats[i] = chats[i][:-1]

t = Tokenizer()
t.fit_on_texts(chats_complete)
vocab = t.word_index
objects = list(set(objects))
count = len(vocab)
for k, v in vocab.items():
    vocab[k] = v - 1
vocab["."] = count
vocab["?"] = count + 1
vocab["!"] = count + 2
vocab[","] = count + 3
vocab["<pad>"] = count + 4
vocab["<unk>"] = count + 5
vocab["<eos>"] = count + 6
count = count + 7
for obj in objects:
    vocab[obj] = count
    count = count + 1
# Dict to json
with open('vocabulary.json', 'w') as fp:
    json.dump(vocab, fp)
