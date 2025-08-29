PROMPT_RULING_ENRICH = """
You will be provided with the ruling and evidence from a fact-checking article. Your task is to enhance the clarity and depth of the ruling.

Definitions:
Ruling: A concise summary of the fact-checking article that includes the veracity rating of the claim.
Evidence: The supporting details and collected data related to the claim.

Please think step by step:
1. In what context is the claim stated?
2. Is there any ambiguity in the ruling, like unclear reference?

Evidence:
{evidence}

Ruling:
{ruling}

Do not output other thing except your enhanced ruling.
"""


PROMPT_INTENT_ANNOTATION = """
A claim would convey implicit intents. You are required to determine intent of a claim based on context in Ruling.

Definition:
Claim: The claim that is checked.
Ruling: Text to determine veracity and explain how the claim would shape people's understanding.
Intent: The understanding of the event that the speaker intend to shape, which is not directly presented in the claim.

Example 1:
Claim: Trey Radel does not even qualify to drive a Lee County school bus at this point, yet he occupies a seat in Congress.
Ruling: 1. The claim that \"Trey Radel does not even qualify to drive a Lee County school bus at this point, yet he occupies a seat in Congress\" is based on the fact that Radel, a U.S. Representative, was convicted of misdemeanor cocaine possession, which would disqualify him from certain employment opportunities, such as a school bus driver in Lee County.\n2. The evidence supports the claim by highlighting that the Lee County school district\u2019s policies explicitly state that it would not hire someone who has pleaded guilty to a misdemeanor drug charge in the past five years, or is currently on probation, both of which apply to Radel.\n3. There are no important concealed facts noted in the original ruling that would alter the perception of the claim. Therefore, the factual basis adequately supports the claim that Radel would not meet the qualifications due to his criminal record.\n4. The ruling accurately reflects the evidence, concluding that Scott\u2019s assertion about Radel\u2019s eligibility to drive a school bus in Lee County is correct, rating the claim as True.
Rationale: This speaker of the claim is trying to say Radel is guilty, so he should not be qualified for the position in Congress.
The speaker intend to shape the understanding that <Trey Radel was convicted of crime, making him unqualified to a Lee County school bus.>

Example 2:
Claim: When I withdrew in June of 2008, polls were showing that at least 40 percent of my supporters said, oh, they weren't going to support Sen. Obama.
Ruling: 1. \"Clinton said, 'When I withdrew in June of 2008, polls were showing that at least 40 percent of my supporters said, oh, they weren't going to support Sen. Obama.' In other words, Clinton argued, if Obama can come back from even sharper divisions to win the presidency, then she can, too.\"\n2. Clinton conceals the fact that the 2016 poll and 2008 poll use different wording, which makes exact comparisons a bit tricky. The 2016 Marist poll is vague compared to the specific 2008 CNN poll.\n3. However, Clinton is right about the scale of Obama\u2019s 2008 problem with dissident Democrats, evidenced by the CNN and ABC News/Washington Post polls from June 2008 showing that 40 percent of her supporters did not back Obama.\n4. Both polls indicate that roughly 60 percent of former Clinton supporters backed Obama, while 40 percent either supported McCain or chose other options.\n5. Clinton's statement reflects a valid point about political dynamics, even if subsequent polls showed Obama eventually gaining significant support from her backers.\n6. We rate the statement Mostly True.
Rationale: Clinton want to say her situation is better than Obama. If Obama can win, then it is also possible for her. 
The speaker intend to shape the understanding that <Clinton's situation is better than Obama's in 2008 based on the polls.>

Example 3:
Claim: 'Scary' fact about our national debt. In 1791 it was $75 million. Today, it rises by that amount in about an hour.
Ruling: Hasner relied on a months-old news story, without attributing it, for his string of \"scary\" debt tweets. That posed a problem for part of his first claim, that the hourly increase of the debt is $75 million. His message -- that the debt is huge and growing rapidly -- is accurate, but he would have achieved a more up-to-date total had he done his own math. He is off by millions of dollars. We rate his claim Half True.
Rationale: The speaker want to tell people that the debt is huge and growing rapidly.
The speaker intend to shape the understanding that <The debt is huge and growing rapidly.>

Requirements:
1. Intent must be checkable. For example, "people should do something" is not checkable because it does happen till now.
2. Output intent in <>. 
3. Please think step by step. First write your rationale, then the intent.

Claim: {claim}
Ruling: {ruling}
Rationale: [Explain your thought process briefly here.]
[Please write your answer here. Do not words like "the claim", write only intent here.]
"""



PROMPT_ENTAILMENT_EVAL = """
You are required to determine does the intended conclusion is supported by the claim.
Please rating from {0, 1, 2}.
0: (refute) The claim in general refutes the intended conclusion.
1: (neutral) No obvious contradiction and supporting is found. The relationship is neutral.
2: (support) The claim fully or partly supports the intended conclusion. eg. the claim is a premise of the intended conclusion. The claim does not have to fully support the intended conclusion.

Claim: {claim}
Intended Conclusion: {intent}

Do not output other thing except your rating.
"""


PROMPT_IMPLICITY_EVAL = """
You are required to determine does the intended conclusion convey the implicit meaning of the claim.

Choose from {0, 1}.
0: (not implicit) the intended conclusion simply rephrase some part of the claim. Do not tell implicit meaning of the claim.
1: (implicit) the intended conclusion talks about the implicit thing that does not exist in the claim.

Claim: {claim}
Intended Conclusion: {intent}

Output only one digit.
"""

PROMPT_SUFFICIENCY_EVAL = """
You are required to determine does the intended conclusion is sufficient, which means that it is understandable within the scope of general knowledge.

Choose from {0, 1}.
0: (not sufficient) the intended conclusion has obvious ambiguous references, not understandable. eg. it uses reference like "the claim".
1: (sufficient) the intended conclusion is clearly referenced.

Intended Conclusion: {intent}

Output only one digit.
"""

PROMPT_READABILITY_EVAL = """
You are required to determine does the intended conclusion is readable.

Choose from {0, 1}.

0: (not readable) the intended conclusion is not readable, very complicated.
1: (readable) the intended conclusion is readable.

Intended Conclusion: {intent}

Output only one digit.
"""


FINETUNE_TEMPLATE_TASK = """You are required to generate intended conclusion of a claim, based on the claim and its evidence.
Claim:
{claim}
Evidence:
{evidence}
"""

FINETUNE_TEMPLATE_OUTPUT = """The intended conclusion of the claim: <{intent}>.
"""