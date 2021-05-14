import sys
import json
import nltk
import imp
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
path = "../data/coqa-dev-v1.0.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
n = len(data['data'])



def remove_punctual(s):
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.replace("?", "")
    s = s.replace("!", "")

    return s

dict1 = ['where', 'when', 'who']
dict2 = ['how', 'why']
for i in range(n):
    for j in range(len(data['data'][i]['questions'])):
        sign = ""
        # print(data['data'][i]['questions'][j], data['data'][i]['answers'][j])
        tokens = nltk.word_tokenize(data['data'][i]['answers'][j]['span_text'])
        tokens = nltk.pos_tag(tokens)
        ques = data['data'][i]['questions'][j]['input_text'].lower()
        ans = data['data'][i]['answers'][j]['span_text'].lower()
        real_ans = data['data'][i]['answers'][j]['input_text'].lower()
        real = remove_punctual(real_ans)
        real = real.split()

        for word in dict1:
            if word in ques or ques[:3] == "was":
                sign = "factual"
                break

        if len(real)<=4:
            sign = "factual"
        if not sign or real_ans=="no" or real_ans=="yes":
            sign = "non-factual"
        data['data'][i]['questions'][j]['type'] = sign

        # print(data['data'][i]['questions'][j]['input_text'])
        # # print(data['data'][i]['answers'][j]['span_text'])
        # print(data['data'][i]['answers'][j]['input_text'])
        # print(sign)
        # print(tokens)
        # print('\n')
        print(data['data'][i]['questions'][j])
    break
    # if i == 10:
    #     break
