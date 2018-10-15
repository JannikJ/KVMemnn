import pandas as pd
import re
import json
from keras.preprocessing.text import Tokenizer

dataset = "test"
json_dataset = dataset
if json_dataset == "val":
    json_dataset = "dev"
dialog = "original"
dialog_type = " - " + dialog
csv_data = pd.read_csv("../data/" + dataset + "_data" + dialog_type + ".csv", encoding="ISO-8859-1", delimiter=';')
json_data = pd.read_json("../../data/kvret_" + json_dataset + "_public.json", encoding="ISO-8859-1")

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
x.to_csv("kbtuples-" + dataset + ".csv")
nkb = {'relation': x['relation'], 'subject': x['subject'], 'object': x['subject'] + "_" + x['relation']}
nkb = pd.DataFrame(nkb)
nkb.drop_duplicates(inplace=True)
nkb.to_csv("normalised_kbtuples-" + dataset + ".csv")
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
indexes_in_dialogs = []
for index, dialog in enumerate(csv_data['input']):
    if index > 0 and (csv_data['index_in_dialogs'][index] != csv_data['index_in_dialogs'][index - 1]
            or (csv_data['index_in_dialogs'][index - 1] == -1 and not last_chat.startswith(dialog))):
        chats.append(chat.copy())
        indexes_in_dialogs.append(csv_data['index_in_dialogs'][index - 1])
        chat = []
        last_chat = ""
    try:
        dialog = dialog.strip('"').lower()
        new_part = dialog[(len(last_chat)):].strip('"').lower()
        # new_part = re.sub(last_chat, "", dialog).strip('"').lower()
        new_response = csv_data['output'][index].strip('"').lower()
        chat.append(new_part)
        chats_complete.append(new_part)
        chat.append(new_response)
        chats_complete.append(new_response)
        if last_chat != "":
            last_chat += "  " + new_part + "  " + new_response
        else:
            last_chat = new_part + "  " + new_response + " "
        # last_chat = last_chat.strip()
    except AttributeError:
        print("Empty dialog detected: " + dialog[(len(last_chat)):] + "\n Response: "
              + str(csv_data['output'][index]))
chats.append(chat.copy())
indexes_in_dialogs.append(csv_data['index_in_dialogs'][index - 1])
chat = []

#Preporcessing replacing values with their canonical representations
count=0
for i, chat in enumerate(chats):
    pois = []  # "odsfh8gr3w8z9febufwebefBUFUOfEHO(hf8ewfebubufesbuofwbuofzuUDVbu"
    if csv_data['index_in_dialogs'][i] != -1:
        for j,ch in enumerate(chat):
            for ki in kb[indexes_in_dialogs[i]]:
                if ki[0].lower() in ch:
                    pois.append(ki[0].lower())
        for j,_ in enumerate(chat):
            for ki in kb[indexes_in_dialogs[i]]:
                if pois.__contains__(ki[0].lower()):
                    if 'day' in ki[1].lower():
                        for kki in ki[1].lower().split(","):
                            if kki in chats[i][j] and ki[2].lower()!='home':
                                count=count+1
                                chats[i][j]=re.sub(kki,'_'.join(ki[0].split(" "))+'_'+ki[1],chats[i][j])
                    if ki[2].lower() in chats[i][j] and ki[2].lower()!='home':
                        count=count+1
                        chats[i][j]=re.sub(ki[2].lower(),'_'.join(ki[0].split(" "))+'_'+ki[1],chats[i][j])
    #break
print(count)

# Making sure to have even number of dialogues in each chat
for i in range(len(chats)):
    if len(chats[i]) % 2 != 0:
        chats[i] = chats[i][:-1]
#Without having context i.e not concatenating consecutive dialogue turns
inputs=[]
outputs=[]
for i in range(len(chats)):
    #for j in range(len(chats[i])):
        inputs.extend(chats[i][::2])
        outputs.extend(chats[i][1::2])
#FOR ENTIRE CONTEXT
inputs=[]
outputs=[]
for i in range(len(chats)):
    sent=''
    for j in range(0,len(chats[i]),2):
        #print(chats[i][j])
        sent+=chats[i][j]+" "
        inputs.append(sent.strip(" "))
        outputs.append(chats[i][j+1].strip(" "))
        sent+=chats[i][j+1]+" "
print(len(inputs),len(outputs))
#Trainset creation
ndf=pd.DataFrame()
ndf["input"]=inputs
ndf["output"]=outputs
ndf=pd.concat([ndf]*3, ignore_index=True)
ndf=ndf.sample(frac=1)
ndf.to_csv(dataset + "_data_" + dialog_type + " - kb.csv",index=False)

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
with open("vocabulary-" + dataset + dialog_type + ".json", 'w') as fp:
    json.dump(vocab, fp)
