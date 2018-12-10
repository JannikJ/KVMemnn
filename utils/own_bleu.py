import json

import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate import bleu_score
import spacy

from examples import main_examples
from utils import clean_text

use_spacy = False
if use_spacy:
    nlp = spacy.load('en')
smt = bleu_score.SmoothingFunction()
file_name_suffix = " - full"
mode = "STANDARD"  # "GAN"
sentence_level = True
generate_examples = True
weather_index = 2
schedule_index = 0
navigate_index = 3
ubuntu_index = 1


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = clean_text.replace_with_tokens(sentence)
    return sentence


def clean_words_json(data):
    returned_data = pd.DataFrame(columns=['sum', 'clean_sum', 'Full_Context', 'cluster_num', 'distance', 'x', 'y', ],
                                 index=[i for i in range(len(data))])
    returned_data = returned_data.fillna('NULL')
    true_labels = []
    for dialog_index, dialog in enumerate(data):
        if dialog['scenario']['task']['intent'] == "weather":
            true_labels.append(weather_index)
        elif dialog['scenario']['task']['intent'] == "navigate":
            true_labels.append(navigate_index)
        elif dialog['scenario']['task']['intent'] == "schedule":
            true_labels.append(schedule_index)
        for utterance_index, utterance in enumerate(dialog['dialogue']):
            if str(returned_data['clean_sum'][dialog_index]) != "NULL":
                returned_data.loc[dialog_index]['sum'] = str(returned_data['sum'][dialog_index]) + " "\
                                                         + utterance['data']['utterance']
                returned_data.loc[dialog_index]['clean_sum'] = str(returned_data['clean_sum'][dialog_index]) + " "\
                    + clean_sentence(utterance['data']['utterance'])
            else:
                returned_data.loc[dialog_index]['clean_sum'] = clean_sentence(utterance['data']['utterance'])
                returned_data.loc[dialog_index]['sum'] = utterance['data']['utterance']
    return returned_data, true_labels


def load_output(task_index=0):
    if mode == "STANDARD":
        if generate_examples:
            return main_examples(file_name_suffix, "_", " - kb", delimiter=",")
        else:
            return pd.read_csv("../final-" + file_name_suffix[3:] + "/output_kb" + file_name_suffix + ".csv", encoding="ISO-8859-1", sep=';')
    elif mode == "GAN":
        json_file = open("../data/kvret_train_public.json")
        json_data = json.load(json_file)
        original, true_labels = clean_words_json(json_data)
        original_task = []
        for index, line in enumerate(original['clean_sum']):
            if true_labels[index] == task_index:
                original_task.append(line)
        if task_index == weather_index:
            generated = pd.read_csv("log-0612-real-weather-dialogues_subbed.txt", encoding="ISO-8859-1", sep='\n')
        elif task_index == schedule_index:
            generated = pd.read_csv("log-0612-real-schedule-dialogues_subbed.txt", encoding="ISO-8859-1", sep='\n')
        elif task_index == navigate_index:
            generated = pd.read_csv("log-0612-real-navigate-dialogues_subbed.txt", encoding="ISO-8859-1", sep='\n')
        else:
            print("NO CORRECT INDEX in load_output")
            generated = []
        return original, generated


def save_scores(output_file, scores):
    output_file['bleu_score'] = pd.Series(scores)
    output_file.to_csv("../final-" + file_name_suffix[3:] + "/output_kb" + file_name_suffix + "-bleu-eval.csv", index=False, encoding="ISO-8859-1", sep=';', decimal=',')


def main():
    output_file = load_output()
    predicted = output_file['predictions']
    expected = output_file['outputs']
    # predicted = ["What city do you want the weather for?"]
    # expected = ["What city do you want the weather for?"]
    weather_scores = []
    schedule_scores = []
    navigate_scores = []
    ubuntu_scores = []
    scores = []
    # for out,pred in zip(expected["output"], predicted["u1"]):
    if use_spacy:
        predicted = [pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for pred in predicted]
        expected = [out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for out in expected]
        predicted_tokens = [nlp(pred) for pred in predicted]
        expected_tokens = [nlp(out) for out in expected]
        for index, _ in enumerate(predicted_tokens):
            score = expected_tokens[index].similarity(predicted_tokens[index])
            scores.append(score)
            if output_file['actual_cluster'][index] == weather_index:
                weather_scores.append(score)
            elif output_file['actual_cluster'][index] == schedule_index:
                schedule_scores.append(score)
            elif output_file['actual_cluster'][index] == navigate_index:
                navigate_scores.append(score)
            elif output_file['actual_cluster'][index] == ubuntu_index:
                ubuntu_scores.append(score)
            else:
                print("OH FUCK!!! NO CORRECT INDEX FOUND!")
            print(expected_tokens[index], "&", predicted_tokens[index])
            print("SCORE: " + str(score))
        c = 0
        for s in scores:
            c = c + s
        print(c / len(scores), len(scores))
        save_scores(output_file, scores)
    else:
        if sentence_level:
            for index, (out, pred) in enumerate(zip(expected, predicted)):
                out = out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
                pred = pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
                # print(type(eval(pred)))
                print(out, "&", pred)
                try:
                    score = sentence_bleu([out.split(" ")], pred.split(" "), smoothing_function=smt.method7)
                    print("SCORE: " + str(score))
                except:
                    score = 0
                scores.append(score)
                if output_file['actual_cluster'][index] == weather_index:
                    weather_scores.append(score)
                elif output_file['actual_cluster'][index] == schedule_index:
                    schedule_scores.append(score)
                elif output_file['actual_cluster'][index] == navigate_index:
                    navigate_scores.append(score)
                elif output_file['actual_cluster'][index] == ubuntu_index:
                    ubuntu_scores.append(score)
                else:
                    print("OH FUCK!!! NO CORRECT INDEX FOUND!")
            c = 0
            for s in scores:
                c = c + s
            print(c / len(scores), len(scores))
            save_scores(output_file, scores)
        else:
            out = [out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for out in expected]
            pred = [pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for pred in predicted]
            score = corpus_bleu(out, pred, smoothing_function=smt.method7)
            print(str(score))
    complete_scores = [weather_scores, schedule_scores, navigate_scores, ubuntu_scores]
    try:
        print_output = ["WEATHER: ", "SCHEDULE: ", "NAVIGATE: ", "UBUNTU: "]
        for i in range(4):
            c = 0
            for s in complete_scores[i]:
                c = c + s
            print(print_output[i] + str(c / len(complete_scores[i])) + str(len(complete_scores[i])))
    except:
        print("ERROR")


def main_gan():
    weather_scores = []
    schedule_scores = []
    navigate_scores = []
    ubuntu_scores = []
    scores = []
    for curr_task in [weather_index, schedule_index, navigate_index]:
        original, generated = load_output(curr_task)
        if use_spacy:
            original = [orig.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for orig in original]
            generated = [gen.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for gen in generated]
            original_tokens = [nlp(orig) for orig in original]
            generated_tokens = [nlp(gen) for gen in generated]
            for index, _ in enumerate(original_tokens):
                score = generated_tokens[index].similarity(original_tokens[index])
                scores.append(score)
                if curr_task == weather_index:
                    weather_scores.append(score)
                elif curr_task == schedule_index:
                    schedule_scores.append(score)
                elif curr_task == navigate_index:
                    navigate_scores.append(score)
                else:
                    print("OH FUCK!!! NO CORRECT INDEX FOUND!")
                print(generated_tokens[index], "&", original_tokens[index])
                print("SCORE: " + str(score))
            c = 0
            for s in scores:
                c = c + s
            print(c / len(scores), len(scores))
        else:
            for index in range(len(generated)):
                original = [orig.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for orig in original]
                gen = generated[index].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
                print(gen)
                try:
                    score = sentence_bleu([orig.split(" ") for orig in original], gen.split(" "), smoothing_function=smt.method7)
                    print("SCORE: " + str(score))
                except:
                    score = 0
                scores.append(score)
            if curr_task == weather_index:
                weather_scores.append(score)
            elif curr_task == schedule_index:
                schedule_scores.append(score)
            elif curr_task == navigate_index:
                navigate_scores.append(score)
            else:
                print("OH FUCK!!! NO CORRECT INDEX FOUND!")
            c = 0
            for s in scores:
                c = c + s
            print(c / len(scores), len(scores))
    complete_scores = [weather_scores, schedule_scores, navigate_scores, ubuntu_scores]
    try:
        print_output = ["WEATHER: ", "SCHEDULE: ", "NAVIGATE: ", "UBUNTU: "]
        for i in range(4):
            c = 0
            for s in complete_scores[i]:
                c = c + s
            print(print_output[i] + str(c / len(complete_scores[i])) + str(len(complete_scores[i])))
    except:
        print("ERROR")


if __name__ == "__main__":
    if mode == "STANDARD":
        main()
    elif mode == "GAN":
        main_gan()
