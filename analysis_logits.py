import json
import argparse
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--task',type=str)
    args = parser.parse_args()
    model_name = args.model_name.split("/")[-1]

    probs1_list,probs2_list_list,answer_list=[] ,[],[]
    P_a_list, P_b_list, P_c_list = [], [], []
    correct1,correct2 = 0,0
    ans=0

    with open(f"log/{model_name.replace('.','_')}_{args.task}_probs.jsonl") as f:
        for line in f:
            tmp = json.loads(line)
            probs1 = tmp['probs1']
            probs2_list = tmp['probs2_list']
            answer = int(tmp['answer'])

            probs1_list.append(tmp['probs1'])
            probs2_list_list.append(tmp['probs2_list'])
            answer_list.append(int(tmp['answer']))
            
            P_a = probs1[answer]

            P_b = probs2_list[answer][answer]

            P_c = 0.0
            for i in range(len(probs2_list)):
                if i==answer:
                    continue
                P_c+=probs1[i]*probs2_list[i][answer]
            try:
                P_c/=(1-P_a)
            except:
                P_c = 0.0

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