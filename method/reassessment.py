import json
from utils import *
from tqdm import tqdm
import argparse
from utils.utils import *
from method.hidden_info_mining import *
from copy import deepcopy


PROMPT_POST_PROCESS = """A claim can be literally correct but still misleading in an implicit way. You are required to review the fact-checking of a claim and refine its veracity.

We have fact-checked a claim and generated a justification. Now, we have discovered hidden information related to the intended conclusion of the claim.

Please review the justification and determine whether the fact-checking has taken the hidden information into account. Then, refine the veracity of the claim accordingly.

You will be provided with the argument supporting the intended conclusion and relevant evidence.

Evidence: [EVIDENCE]
Argument: [ARGUMENT]
Justification: [JUSTIFICATION]

Please refine the veracity of the claim based on the above information.

Original Veracity: [VERACITY]

Choose one of the following options:
A. True
B. Half-true
C. False
D. Unverifiable (e.g., the assumption does not strongly support the intended conclusion. not enough information)

Only output the veracity option. Do not include additional text.
From {A, B, C, D}, your answer is: 
"""

def post_fix(pred, rationale, relevant_evidence, argument):
    all_hidden_evidence_key = []
    for assumption in argument["assumption"]:
        assumption["backing"] = assumption["supported_by"] + assumption["refuted_by"]
        assumption.pop("supported_by")
        assumption.pop("refuted_by")
        all_hidden_evidence_key += assumption["backing"]
    all_hidden_evidence_key = list(set(all_hidden_evidence_key))
    hidden_evidence = {key: relevant_evidence[key] for key in all_hidden_evidence_key}
    prompt = PROMPT_POST_PROCESS.replace("[EVIDENCE]", json.dumps(hidden_evidence, indent=4)).\
        replace("[ARGUMENT]", json.dumps(argument, indent=4)).replace("[JUSTIFICATION]", rationale).replace("[VERACITY]", pred)
    response = call_gpt(prompt)
    if "A" in response[:2]:
        return "true"
    elif "B" in response[:2]:
        return "half-true"
    elif "C" in response[:2]:
        return "false" 
    else:
        return pred



def main(args):
    pred_dict = {}
    logger = DataHandler(task_description="post-process" + "\n" + args.datafile, result_folder="method/result/")
    literal_path = args.literal

    intent_assessor = IntentArgumentation(intent_model_id=args.intent_model)
    hidden_info_collections = []

    with open(literal_path, "r") as f:
        for line in f:
            record = json.loads(line)
            pred_dict[record['example_id']] = record
    with open(args.datafile, 'r') as f:
        data = json.load(f)
    for event in tqdm(data):
        pred = pred_dict[event['example_id']]['pred']
        pred_new = pred
        rationale = pred_dict[event['example_id']]['rationale']
        evidence = event["evidence"]
        evidence_dict = {
            f"Evidence_{i+1}": evidence[i] for i in range(len(evidence))
        }
        if pred == "true":
            event["relevant_evidence"] = evidence_dict
            event = intent_assessor.mining_one_event(event)
            argument = event["argument"]
            hidden_info_collections.append(event)
            if argument["assumption"]:
                ans = post_fix(pred, rationale, evidence_dict, deepcopy(argument))
                pred_new = ans
        logger.log_iteration({
            "example_id": event['example_id'],
            "veracity": event['veracity'],
            "pred": pred_new,
            "old_pred": pred,
            "ruling": event["ruling"],
            "rationale": rationale,
            "relevant_evidence": evidence_dict,
        })

    logger.save_file("hidden_info.jsonl", hidden_info_collections)
    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, help="The data file to be processed.")
    parser.add_argument("--literal", type=str, help="The data file to of literal checking.")
    parser.add_argument("--intent_model", type=str, help="Intent model id of OpenAI")
    args = parser.parse_args()
    main(args)