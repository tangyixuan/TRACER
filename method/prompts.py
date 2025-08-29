import os
import json
import openai
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


def call_gpt(cur_prompt, stop=None, model="gpt-4o-mini"):
    reasoner_messages = [
        {
            "role": "user",
            "content": cur_prompt
        },
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=reasoner_messages,
    )
    returned = completion.choices[0].message.content
    return returned


PROMPT_IMPLICIT_QUESTION = """
A claim could be literally accurate but still misleading in an implicit way.
Your task is to find the important implicit questions that the evidence talk about.

Steps:
1. Read the evidence below to know the whole story and what topics does the evidence talks about.
2. Assume the claim is correct, What important implicit question would you raise to verify the intended conclusion, rather than the claim?
3. You should generate 1-3 implicit questions, with each implicit question not only ask for literal accuracy of the claim.
4. Each question should be output inside an independent <>.

Example 1
Claim: With voting by mail, “you get thousands and thousands of people ... signing ballots all over the place.”
Intended conclusion: Mail-in voting is inherently insecure or problematic because it allows an uncontrolled number of ballots to be signed without proper oversight.
Evidence:
Rare doesn’t mean nonexistent. And experts agree that mail balloting does provide a greater potential for ballot fraud than in-person voting does.
"It's definitely easier to have fraud with absentee voting than in-person voting, and some cases of it have had a big impact," said Rob Richie, president of FairVote, a voting-access advocacy group.
## your output could be:
<Is there a greater risk of voting fraud with mail-in ballots?><Is the mail-in voting process without safeguards?>

Example 2
Claim: You may not have stayed in a hotel in the past year, but illegals arriving since Biden’s inauguration, they get to stay free of charge.
Intended Conclusion: Under Biden's policies, illegal immigrants are receiving undue benefits.
Evidence:
“As part of a story about an influx of migrants at the U.S. border with Mexico, the New York Times reported that some migrants arriving in San Diego were quarantining in a hotel in order to comply with California’s 10-day quarantine requirement for out-of-state travelers.”
"You may not have stayed in a hotel in the past year, but illegals arriving since Biden’s inauguration, they get to stay free of charge," Ingraham said Feb. 8, citing a report in the New York Times.
“To guard against the coronavirus, health authorities in San Diego have arranged housing for hundreds of arriving migrants in a downtown high-rise hotel, where they are being quarantined before being allowed to join family or friends in the interior of the United States,” the newspaper found.
“The hotel stay is not a vacation,” said Nicole Hallett, an associate clinical professor of law and the director of the Immigrants’ Rights Clinic at the University of Chicago. “It is designed as a public health measure to protect the entire community.”
“Thus, the government releases people to a local nonprofit that assists with the quarantine and then ensures that they arrive at their final destination,” Hallett said. “It appears that Jewish Family Service is playing this role in San Diego.”
## your output could be:
<Are the hotel stays for migrants funded as a public health necessity rather than as a luxury or benefit for undocumented individuals?>

All implicit questions must be yes-no questions.

Claim:{claim}
Intended conclusion:{intent}
Evidence:
{evidence}
"""

PROMPT_IMPLICIT_ASSUMPTION = """A claim could be literally accurate but still misleading because of its intended conclusion.
You task is to determine what assumptions does the intended conclusion is based on besides the claim.

Definition:
Claim: A statement assumed to be true.
Intended conclusion: The intended conclusion of the claim, which needs checking.
Questions: Some important questions when checking the claim.
Assumptions: The assumptions that the intended conclusion is based on besides the claim.

Steps:
1. Read the claim, intended conclusion and questions.
2. Assuming the claim is correct, what assumptions does the question imply should serve as the basis for the intended conclusion?
3. Output 1-3 sentences rationale and then 1-{assumption_max_number} assumptions. Assumption should be output inside a <> divided by "||".

Example
Claim: Hillary Clinton stated on April 21, 2016 in a town hall on ABC's 'Good Morning America': "When I withdrew in June of 2008, polls were showing that at least 40 percent of my supporters said, oh, they weren't going to support Sen. Obama."
Intended conclusion: Clinton's situation is better than Obama in 2008 based on the polls.
Questions:
1. Did the 2008 polls accurately reflect the long-term support of Clinton's supporters for Obama?
2. Were the differences in poll question wording between 2008 and 2016 significant enough to affect the comparison of supporter loyalty?
3. Did the 40% of Clinton's 2008 supporters who initially said they wouldn't support Obama actually not vote for him in the general election?
4. Does the fact that Clinton is 10 points better in securing Sanders voters' backing than Obama in securing Clinton backers' backing necessarily mean her situation is better overall?
# Your output could be:
To support intended conclusion, we must admit polls is a good measure of support, and the polls should be comparable the two candidates.
<2008 polls accurately reflect the long-term support of Clinton's supporters for Obama.||The differences in poll question wording between 2008 and 2016 were significant enough to affect the comparison of supporter loyalty.>

Requirements:
1. Ensure that each assumption can independently convey its meaning. For example, never use vague references like "the claim," "the evidence," or "the intent"; instead, refer to specific information.
2. Only include assumptions that you are confident in and that serve as a strong basis for the intended conclusion.

Claim: {claim}
Intended conclusion: {intention}
Questions:
{questions}
"""


PROMPT_INTENT_ANALYSIS = """You are required to determine whether an argument is accurate with the given Toulmin's schema.
For each warrant, we could provide supporting backing, contradiction backing or both.

Is the argument reasonable and plausible? Please give 4-5 sentences rationale and then your answer.

{argument}
"""


TEMPLATE_CONTEXT_CLAIM = "{person} stated on {date} that {claim}."


PROMPT_CAUSAL_EVAL = """You are required to do a counterfactual causal inference on a given causal graph.

Argument = {argument}

{query}

A. The probability of Z does not change. 
B. The probability of Z increases (Z becomes more likely to be true).
C. The probability of Z decreases (Z becomes less likely to be true).

Please answer with one letter only.
"""


PROMPT_IMPLICIT_QUESTION_HYPER = """
A claim could be literally accurate but still misleading in an implicit way.
Your task is to find the important implicit questions that the evidence talk about.

Steps:
1. Read the evidence below to know the whole story and what topics does the evidence talks about.
2. Assume the claim is correct, What important implicit question would you raise to verify the intended conclusion, rather than the claim?
3. You should generate 1-{question_num} implicit questions, with each implicit question not only ask for literal accuracy of the claim.
4. Each question should be output inside an independent <>.

Example 1
Claim: With voting by mail, “you get thousands and thousands of people ... signing ballots all over the place.”
Intended conclusion: Mail-in voting is inherently insecure or problematic because it allows an uncontrolled number of ballots to be signed without proper oversight.
Evidence:
Rare doesn’t mean nonexistent. And experts agree that mail balloting does provide a greater potential for ballot fraud than in-person voting does.
"It's definitely easier to have fraud with absentee voting than in-person voting, and some cases of it have had a big impact," said Rob Richie, president of FairVote, a voting-access advocacy group.
## your output could be:
<Is there a greater risk of voting fraud with mail-in ballots?><Is the mail-in voting process without safeguards?>

Example 2
Claim: You may not have stayed in a hotel in the past year, but illegals arriving since Biden’s inauguration, they get to stay free of charge.
Intended Conclusion: Under Biden's policies, illegal immigrants are receiving undue benefits.
Evidence:
“As part of a story about an influx of migrants at the U.S. border with Mexico, the New York Times reported that some migrants arriving in San Diego were quarantining in a hotel in order to comply with California’s 10-day quarantine requirement for out-of-state travelers.”
"You may not have stayed in a hotel in the past year, but illegals arriving since Biden’s inauguration, they get to stay free of charge," Ingraham said Feb. 8, citing a report in the New York Times.
“To guard against the coronavirus, health authorities in San Diego have arranged housing for hundreds of arriving migrants in a downtown high-rise hotel, where they are being quarantined before being allowed to join family or friends in the interior of the United States,” the newspaper found.
“The hotel stay is not a vacation,” said Nicole Hallett, an associate clinical professor of law and the director of the Immigrants’ Rights Clinic at the University of Chicago. “It is designed as a public health measure to protect the entire community.”
“Thus, the government releases people to a local nonprofit that assists with the quarantine and then ensures that they arrive at their final destination,” Hallett said. “It appears that Jewish Family Service is playing this role in San Diego.”
## your output could be:
<Are the hotel stays for migrants funded as a public health necessity rather than as a luxury or benefit for undocumented individuals?>

All implicit questions must be yes-no questions.

Claim:{claim}
Intended conclusion:{intent}
Evidence:
{evidence}
"""


PROMPT_IMPLICIT_ASSUMPTION = """A claim could be literally accurate but still misleading because of its intended conclusion.
You task is to determine what assumptions does the intended conclusion is based on besides the claim.

Definition:
Claim: A statement assumed to be true.
Intended conclusion: The intended conclusion of the claim, which needs checking.
Questions: Some important questions when checking the claim.
Assumptions: The assumptions that the intended conclusion is based on besides the claim.

Steps:
1. Read the claim, intended conclusion and questions.
2. Assuming the claim is correct, what assumptions does the question imply should serve as the basis for the intended conclusion?
3. Output 1-3 sentences rationale and then 1-{assumption_max_number} assumptions. Assumption should be output inside a <> divided by "||".

Example
Claim: Hillary Clinton stated on April 21, 2016 in a town hall on ABC's 'Good Morning America': "When I withdrew in June of 2008, polls were showing that at least 40 percent of my supporters said, oh, they weren't going to support Sen. Obama."
Intended conclusion: Clinton's situation is better than Obama in 2008 based on the polls.
Questions:
1. Did the 2008 polls accurately reflect the long-term support of Clinton's supporters for Obama?
2. Were the differences in poll question wording between 2008 and 2016 significant enough to affect the comparison of supporter loyalty?
3. Did the 40% of Clinton's 2008 supporters who initially said they wouldn't support Obama actually not vote for him in the general election?
4. Does the fact that Clinton is 10 points better in securing Sanders voters' backing than Obama in securing Clinton backers' backing necessarily mean her situation is better overall?
# Your output could be:
To support intended conclusion, we must admit polls is a good measure of support, and the polls should be comparable the two candidates.
<2008 polls accurately reflect the long-term support of Clinton's supporters for Obama.||The differences in poll question wording between 2008 and 2016 were significant enough to affect the comparison of supporter loyalty.>

Requirements:
1. Ensure that each assumption can independently convey its meaning. For example, never use vague references like "the claim," "the evidence," or "the intent"; instead, refer to specific information.
2. Only include assumptions that you are confident in and that serve as a strong basis for the intended conclusion.

Claim: {claim}
Intended conclusion: {intention}
Questions:
{questions}
"""


class QwenModelWrapper:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _call_qwen(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    def __call__(self, *args, **kwds):
        return self._call_qwen(*args, **kwds)
        


class GPT4oMiniWrapper:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
    
    def call_gpt(self, cur_prompt, stop=None):
        reasoner_messages = [
            {
                "role": "user",
                "content": cur_prompt
            },
        ]
        completion = openai.chat.completions.create(
            model=self.model_name,
            messages=reasoner_messages,
        )
        returned = completion.choices[0].message.content
        return returned

    def __call__(self, *args, **kwds):
        return self.call_gpt(*args, **kwds)