import json
import argparse
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--task',type=str)
    parser.add_argument('--sampling_times',type=int,default=10)
    args = parser.parse_args()
    model_name = args.model_name.split("/")[-1]


    data_dic = {}
    assert args.task in ['gsm8k','ifeval','humaneval']
    with open(f"log/{model_name.replace('.','_')}_{args.task}_sampling{args.sampling_times}.jsonl") as f:
        for line in f:
            tmp = json.loads(line)
            if tmp['id'] not in data_dic:
                data_dic[tmp['id']] = [[],[]]
            data_dic[tmp['id']][0].append(tmp['correct1'])
            data_dic[tmp['id']][1].append(tmp['correct2'])

    P_a_list, P_b_list, P_c_list = [], [], []

    for l in data_dic.values():

        P_a = sum(l[0])/len(l[0])
        if any(l[0]):
            P_b = sum([(p1 and p2) for p1,p2 in zip(l[0],l[1])])/sum(l[0])
        else:
            P_b = 0.0
        
        if all(l[0]):
            P_c = 0.0
        else:
            P_c = sum([((not p1) and p2) for p1,p2 in zip(l[0],l[1])])/( len(l[0]) - sum(l[0]) )

        P_a_list.append(P_a)
        P_b_list.append(P_b)
        P_c_list.append(P_c)

    acc1 = sum(P_a_list)/len(P_a_list)
    acc2 = sum([p1*p2+(1-p1)*p3 for p1,p2,p3 in zip(P_a_list,P_b_list,P_c_list)])/len(P_a_list)

    metric1 = sum([p1*p2 for p1,p2 in zip(P_a_list,P_b_list)])/sum(P_a_list)
    metric2 = sum([(1-p1)*p2 for p1,p2 in zip(P_a_list,P_c_list)])/(len(P_a_list)-sum(P_a_list))

    print("*"*20)
    print(f"{args.model_name} {args.task}:")
    print(f"Acc1:{100*acc1:.1f} %")
    print(f"Acc2:{100*acc2:.1f} %")
    print(f"CL:{100*metric1:.1f} %")
    print(f"CS:{100*metric2:.1f} %")
    print(f"RSS:{100*(acc2-acc1*acc1)/(2*acc1*(1-acc1)):.1f} %")
    print("*"*20)
    print()
    
if __name__ == "__main__":
    main()

