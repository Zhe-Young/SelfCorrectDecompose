import re

def check_correctness_boolq(datum,answer):
    true_match = re.search(r'true',answer.lower())
    false_match = re.search(r'false',answer.lower())
    if true_match == None and false_match == None:
        return False
    elif true_match!=None and false_match!=None:
        if true_match.span()[0]<false_match.span()[0]:
            pred = True
        else:
            pred = False
    elif true_match!=None:
        pred = True 
    elif false_match!=None:
        pred = False
    return pred == datum['answer']

# answer = "**True.** False\"The Strangers\" is loosely based on real-life events."
# datum = {'answer':True}
# print(check_correctness_boolq(datum,answer))
# true_match = re.search(r'true',answer.lower())
# print(true_match)
# print(true_match.span()[0])