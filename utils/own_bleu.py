import json
import re
from json import JSONDecodeError

import pandas as pd
from nltk import word_tokenize
from own_stuff.clean_text import replace_with_tokens

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate import bleu_score
import spacy

from examples import main_examples
from utils import clean_text

use_spacy = False
if use_spacy:
    nlp = spacy.load('en')
smt = bleu_score.SmoothingFunction()
file_name_suffix = " - ubuntu"
final_folder = "decomp"
test_mode = " - rule"
mode = "STANDARD"  # "STANDARD"  # "GAN"
iteration = 150000
sentence_level = False
generate_examples = True
weather_index = 2
schedule_index = 0
navigate_index = 3
ubuntu_index = 1
driver_last_utterances = {"thanks", "thanks car", "thanks so much for your help", "thanks for the help", "thanks bud",
                              "thanks so much", "thanks for the information", "thanks for all the help", "thanks for all your help",
                              "thanks a bunch", "thanks so much for the information", "thanks once again", "thanks that sounds great",
                              "thanks a lot", "thanks again car assistant", "thanks again", "thanks for the reminder",
                              "thanks for directing me via the route with least traffic", "thanks car assistant",
                              "thanks i appreciate it", "thanks so much for the help", "thanks sounds good", "thanks for your help",
                              "thanks for the heads up", "thanks to you car",
                              "thank you", "thank you very much assistant", "thank you car", "thank you car assistant",
                              "thank you kindly", "thank you very much car", "thank you very much", "thank you car you rock",
                              "thank you so much car", "thank you for all your help", "thank you buddy", "thank you car let's go there",
                              "thank you so much", "thank you car let's go there", "thank you for looking", "thank you car assitant",
                              "thank you for the help", "thank you car this sounds awesome",
                              "okay perfect thanks", "okay thanks", "ok thanks", "okay thank you", "ok thank you then",
                              "okay wonderful thank you", "ok thank you", "okay nice thank you",
                              "sounds great thanks", "sounds great thank you", "sounds awesome thanks",
                              "perfect", "perfect thanks so much", "perfect thanks", "perfect thank you", "okay perfect thank you",
                              "fantastic thank you",
                              "that is all thank you", "that is all thanks",
                              "great thank you", "great thanks for the assistance", "great thanks", "great thanks again",
                              "great thanks so much for the help", "great", "that's great thank you car", "great thank you for helping me out",
                              "thanks sound great", "great thank you so much", "great thanks",
                              "that works thanks",
                              "you're awesome thanks", "awesome thanks", "awesome thanks so much",
                              "cool thanks",
                              "alright thanks", "alright thank you", "alrigth thank you",
                              "ok thanks for the information",
                              "excellent that's very helpful",
                              "great", "great goodbye", "that sounds great thanks", "that's great thank you",
                              "that's it thank you car",
                              "wow thank you", "that will do thanks", "that solves it thanks\\", "that solves it thanks",
                              "sounds good thanks for the information"
                              }
car_last_utterances = {"you are welcome", "you are welcome person", "you are welcome driver", "you are very welcome",
                       "you are welcome human", "you are very welcome", "you are most welcome", "you are welcome driver drive safely",
                       "you are welcome friend", "you are welcome sir",
                       "you're welcome", "you're welcome have a great day", "you're welcome happy to help",
                       "you're welcome have a good day", "you're welcome drive carefully", "you're very welcome",
                       "you're welcome drive safely and enjoy", "you're welcome glad i could help", "you're welcome drive safely",
                       "you're welcome glad to help", "you're welcome we should arrive shortly", "you're welcome always happy to help",
                       "you're welcome my driver", "you're welcome anytime", "you're welcome i'm here to assist you",
                       "you're welcome happy to be of assistance", "you're welcome my pleasure", "you're welcome drive carefully and enjoy",
                       "you're welcome take care",
                       "i sent the info on your screen you're welcome",
                       "no problem", "no problem.", "no probem", "not a problem", "no problem driver", "no problem you're welcome",
                       "no problem i'm here to help",
                       "happy to help", "always happy to help you're welcome", "happy to be of service",
                       "glad to help", "glad to help sir", "glad i could help out", "glad i could assist",
                       "anytime", "anytime!",
                       "have a great day", "have a good day", "have a great day and you're welcome", "have a great day and happy to help",
                       "have a great day happy to help",
                       "glad i could be of assistance you're welcome", "glad i could help you're welcome have a great day",
                       "my pleasure",
                       "great you're welcome"}


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = clean_text.replace_with_tokens(sentence)
    sentence = clean_temperature(sentence)
    return sentence


def clean_words_json(data):
    returned_data = pd.DataFrame(columns=['sum', 'clean_sum', 'Full_Context', 'cluster_num', 'distance', 'x', 'y', ],
                                 index=[i for i in range(len(data))])
    returned_data = returned_data.fillna('NULL')
    true_labels = []
    for dialog_index, dialog in enumerate(data):
        try:
            if dialog['scenario']['task']['intent'] == "weather":
                true_labels.append(weather_index)
            elif dialog['scenario']['task']['intent'] == "navigate":
                true_labels.append(navigate_index)
            elif dialog['scenario']['task']['intent'] == "schedule":
                true_labels.append(schedule_index)
        except:
            print("")
        try:
            for utterance_index, utterance in enumerate(dialog['dialogue']):
                if str(returned_data['clean_sum'][dialog_index]) != "NULL":
                    returned_data.loc[dialog_index]['sum'] = str(returned_data['sum'][dialog_index]) + " "\
                                                             + utterance['data']['utterance']
                    returned_data.loc[dialog_index]['clean_sum'] = str(returned_data['clean_sum'][dialog_index]) + " "\
                        + clean_sentence(utterance['data']['utterance'])
                else:
                    returned_data.loc[dialog_index]['clean_sum'] = clean_sentence(utterance['data']['utterance'])
                    returned_data.loc[dialog_index]['sum'] = utterance['data']['utterance']
        except KeyError:
            print("found no dialogue key in this part: " + str(dialog))
    return returned_data, true_labels


def load_output(task_index=0, temp="0.0"):
    if mode == "STANDARD":
        if generate_examples:
            return main_examples(file_name_suffix, "_", " - kb", delimiter=",", iteration=iteration, final_folder=final_folder, test_mode=test_mode)
        else:
            return pd.read_csv("../final-" + file_name_suffix[3:] + "/output_kb" + file_name_suffix + ".csv", encoding="ISO-8859-1", sep=';')
    elif mode == "GAN":
        with open("../data/kvret_train_public.json") as json_file:
            json_data = json.load(json_file)
        original, true_labels = clean_words_json(json_data)
        original_task = []
        for index, line in enumerate(original['clean_sum']):
            if true_labels[index] == task_index:
                original_task.append(line)
        try:
            if task_index == weather_index:
                with open("log-1112-real-weather_subbed-" + temp + ".json") as json_file:
                    json_data = json.load(json_file)
                generated, _ = clean_words_json(json_data)
            elif task_index == schedule_index:
                with open("log-1112-real-schedule_subbed-" + temp + ".json") as json_file:
                    json_data = json.load(json_file)
                generated, _ = clean_words_json(json_data)
            elif task_index == navigate_index:
                with open("log-1112-real-navigate_subbed-" + temp + ".json") as json_file:
                    json_data = json.load(json_file)
                generated, _ = clean_words_json(json_data)
            else:
                print("NO CORRECT INDEX in load_output")
                generated = []
        except JSONDecodeError as e:
            generated = []
            print("JSONDECODEERROR! With file of task index " + str(task_index) + " and temperature " + str(temp))
        return original, generated


def save_scores(output_file, scores):
    output_file['bleu_score'] = pd.Series(scores)
    output_file.to_csv("../final-" + file_name_suffix[3:] + "/output_kb" + file_name_suffix + "-bleu-eval.csv", index=False, encoding="ISO-8859-1", sep=';', decimal=',')


def main():
    output_file = load_output()
    predicted = output_file['predictions']
    expected = output_file['outputs']
    # for pred_index, pred in enumerate(predicted):
    #     if predicted[pred_index].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() in car_last_utterances\
    #             or re.match(".*welcome.*", predicted[pred_index], flags=re.IGNORECASE) is not None:
    #         predicted[pred_index] = "You are welcome"
    #     if predicted[pred_index].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() in driver_last_utterances\
    #             or re.match(".*thank.*", predicted[pred_index], flags=re.IGNORECASE) is not None:
    #         predicted[pred_index] = "Thank you"
    # for expc_index, expc in enumerate(expected):
    #     if expected[expc_index].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() in car_last_utterances\
    #             or re.match(".*welcome.*", expected[expc_index], flags=re.IGNORECASE) is not None:
    #         expected[expc_index] = "You are welcome"
    #     if expected[expc_index].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() in driver_last_utterances\
    #             or re.match(".*thank.*", expected[expc_index], flags=re.IGNORECASE) is not None:
    #         expected[expc_index] = "Thank you"
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
                print("OH!!! NO CORRECT INDEX FOUND!")
            print(expected_tokens[index], "&", predicted_tokens[index])
            print("SCORE: " + str(score))
        c = 0
        for s in scores:
            c = c + s
        try:
            print(c / len(scores), len(scores))
        except ZeroDivisionError:
            print("ZERO DIVISION ERROR")
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
                    print("OH!!! NO CORRECT INDEX FOUND!")
            c = 0
            for s in scores:
                c = c + s
            try:
                print(c / len(scores), len(scores))
            except ZeroDivisionError:
                print("ZERO DIVISION ERROR")
            save_scores(output_file, scores)
        else:
            out = [out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for out in expected]
            pred = [pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for pred in predicted]
            score = corpus_bleu(out, pred, smoothing_function=smt.method7)
            print(str(score))
    complete_scores = [weather_scores, schedule_scores, navigate_scores, ubuntu_scores]

    print_output = ["WEATHER: ", "SCHEDULE: ", "NAVIGATE: ", "UBUNTU: "]
    for i in range(4):
        c = 0
        for s in complete_scores[i]:
            c = c + s
        try:
            print(print_output[i] + str(c / len(complete_scores[i])) + str(len(complete_scores[i])))
        except:
            print("ERROR")


def clean_temperature(text):
    text = re.sub("[0-9]+ ?(of)? ?(th)(January|February|March|April|May|June|July|August|September|October|November|Decembr?e?e?r?)?", "<date>", text, flags=re.IGNORECASE)
    text = re.sub("(January|February|March|April|May|June|July|August|September|October|November|Decembr?e?e?r?) ?[0-9]*(th)?", "<date>", text, flags=re.IGNORECASE)
    text = re.sub("(January|February|March|April|May|June|July|August|September|October|November|Decembr?e?e?r?)? ?[0-9]+(th)", "<date>", text, flags=re.IGNORECASE)
    text = re.sub("[0-9]+:?[0-9]* ?(am|pm|m)", "<time>", text, flags=re.IGNORECASE)
    text = re.sub("[0-9]+ ?- ?<temperature>", "<temperature>-<temperature>", text)
    text = re.sub("[0-9]+ (hours|days?)", "<weekly_time>", text)
    text = re.sub("[0-9]+0s ", "<temperature> ", text, flags=re.IGNORECASE)
    text = re.sub("take my medicine", "<event> ", text, flags=re.IGNORECASE)
    text = re.sub("tennis", "<event> ", text, flags=re.IGNORECASE)
    text = re.sub("yoga [class]?", "<event> ", text, flags=re.IGNORECASE)
    text = re.sub("doctor('s)? appointment", "<event> ", text, flags=re.IGNORECASE)
    text = re.sub("swimming", "<event> ", text, flags=re.IGNORECASE)
    text = re.sub("(my son's)? ?football ?(appointments?|game)?", "<event> ", text, flags=re.IGNORECASE)
    return text


def main_gan():
    weather_scores = []
    schedule_scores = []
    navigate_scores = []
    ubuntu_scores = []
    scores = []
    for curr_task in [weather_index, schedule_index, navigate_index]:
        for temp in range(0, 11, 1):
            temp = (temp / 10).__round__(1)
            temp = str(temp)
            original, generated = load_output(curr_task, temp)
            original = original['clean_sum']
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
                        print("OH!!! NO CORRECT INDEX FOUND!")
                    print(generated_tokens[index], "&", original_tokens[index])
                    print("SCORE: " + str(score))
                c = 0
                for s in scores:
                    c = c + s
                try:
                    print(c / len(scores), len(scores))
                except ZeroDivisionError:
                    print("ZERO DIVISION ERROR")
            else:
                for index in range(len(generated)):
                    original = [orig.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for orig in original]
                    gen = generated['clean_sum'][index].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
                    # print(gen)
                    try:
                        score = sentence_bleu([orig.split(" ") for orig in original], gen.split(" "), smoothing_function=smt.method7)
                        # print("SCORE: " + str(score))
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
                    print("OH!!! NO CORRECT INDEX FOUND!")
                c = 0
                for s in scores:
                    c = c + s
                try:
                    print("SCORE with task " + str(curr_task) + " and temp " + str(temp) + " : " + str(c / len(scores)) + " with length of " +  str(len(scores)))
                except ZeroDivisionError:
                    print("ZERO DIVISION ERROR")
                scores = []
    complete_scores = [weather_scores, schedule_scores, navigate_scores, ubuntu_scores]
    print_output = ["WEATHER: ", "SCHEDULE: ", "NAVIGATE: ", "UBUNTU: "]
    for i in range(4):
        c = 0
        for s in complete_scores[i]:
            c = c + s
        try:
            print(print_output[i] + str(c / len(complete_scores[i])) + str(len(complete_scores[i])))
        except:
            print("ERROR")


if __name__ == "__main__":
    if mode == "STANDARD":
        main()
    elif mode == "GAN":
        main_gan()
