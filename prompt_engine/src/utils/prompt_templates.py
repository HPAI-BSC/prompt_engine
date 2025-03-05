

jinja_chatml_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

think_prompt = """
You are a highly qualified medical expert, adept at answering complex medical exam questions. Your task is to answer multiple-choice medical exam questions by following a structured reasoning process.

You will be given with some examples curated from a database of medical questions and their corresponding solutions. Each example showcases a detailed, step-by-step reasoning process leading to the correct answer. Carefully analyze these examples, paying close attention to the following:

*   The medical knowledge applied in each step.
*   The logical flow of reasoning, connecting medical concepts to the question.
*   The format used to present the reasoning and final answer.

Your task is to answer a new medical question by emulating the reasoning style and format observed in the examples.

Follow these instructions carefully:

1.  Thoroughly review the examples to internalize the reasoning patterns and solution format. While you should follow the format, DO NOT copy the reasoning verbatim. The idea is to understand how to THINK.
2.  Read and understand the question.
3.  Aticulate your reasoning process inside the <think></think> tags. Connect each reasoning step to relevant medical concepts, demonstrating a deep understanding of the question's medical context.
4.  Based on your reasoning, construct a well-structured explanation leading to the correct answer.
5.  Finally, select the single correct answer option (A, B, C, or D).
6.  Present your final answer in the format "Answer: [X]", where X represents the correct answer option you selected. It is crucial that you include NO additional text or tokens after “Answer: [X]”.

Here are the examples, presented in a user-assistant chat format, where the user poses a medical question, and the assistant provides a structured solution. The final user message will contain the new question you need to answer.
"""

datasets_prompts = {
    "4_options": 
    {
        "examples": 
            [
                {
                    "question": """Exits on a highway are numbered consecutively from 1 to 50. The distance from exit 41 to exit 50 is 100 km. If each exit is at least 6 km from the next exit, what is the longest possible distance, in kilometers, between exit 47 and exit 48?
Options:
A. 52
B. 51
C. 50
D. 49""",
                    "answer": """To maximize the distance between exits 47 and 48, we should minimize the distances of the other intervals. There are eight intervals other than the one between exits 47 and 48. Since each exit is at least 6 km from the next, the minimum distance for these eight intervals (from exit 41 to exit 50) would be \(8 \times 6 = 48\) km. Therefore, the distance between exits 47 and 48 would be \(100 - 48 = 52\) km.

Answer: [A]""",
                },
                {
                    "question": """The two principal contractile proteins found in skeletal muscle are:
Options:
A. actin and troponin.
B. actin and myosin.
C. troponin and tropomyosin.
D. myosin and tropomyosin.""",
                    "answer": """Actin is a thin filament protein, forming a helical structure along with tropomyosin and troponin, which regulate its interaction with myosin. Myosin, on the other hand, is a thick filament protein characterized by its long tail and globular head. The heads of the myosin filaments form cross-bridges by attaching to specific sites on the actin filaments. Through a series of movements fueled by ATP hydrolysis, these cross-bridges pull the actin filaments towards the center of the sarcomere, the basic unit of a muscle fiber, causing the muscle to contract. While troponin and tropomyosin are also important in muscle contraction, they serve more as regulatory proteins rather than primary contractile proteins.

Answer: [B]""",
                },
                {
                    "question": """A man was angry at a coworker who had received a promotion. The man believed that the coworker had taken credit for the man's work and had bad- mouthed him to their boss. One day, as the man was leaving the company parking lot in his car, he saw the coworker walking through the lot. On a sudden impulse, the man pushed the accelerator pedal hard and veered toward the coworker with the intention of scaring him. The coworker tried to jump out of the way but slipped and fell and was run over. Although the coworker suffered life-threatening injuries, he survived. In a jurisdiction that follows the common law of homicide, could the man properly be convicted of attempted murder? 
Options:
A. No, because the coworker's slip and fall broke the chain of proximate causation.
B. No, because the man lacked the requisite intent.
C. Yes, because the coworker suffered life- threatening injuries.
D. Yes, because the man acted with reckless disregard for human life.""",
                    "answer": """In this scenario, the key factors to consider are the man's intent and the nature of his actions under the common law of homicide. Attempted murder requires a specific intent to kill. The man's action of veering his car towards the coworker was impulsive and aimed at scaring, not killing. This suggests a lack of specific intent to commit murder.

Answer: [B]""",
                },
                {
                    "question": """Which one of the following is not a vectored interrupt?
Options:            
A. TRAP.
B. RST 7.5.
C. RST 6.5.
D. INTR.""",
                    "answer": """Vectored interrupts are a type of interrupt where the address of the interrupt service routine (ISR) is hardwired. That means the processor automatically knows where to go when the interrupt occurs. 

                                - **TRAP** is a vectored interrupt. It has a fixed location in memory, so when the TRAP interrupt occurs, the processor knows exactly where to go.
                                - **RST 7.5 and RST 6.5** are also vectored interrupts in the Intel 8085 microprocessor. These interrupts have predefined memory addresses, so the processor can jump directly to these addresses when these interrupts are triggered.
                                - **INTR**, on the other hand, is a non-vectored interrupt. It does not have a predefined memory address. Instead, when the INTR interrupt occurs, the device has to supply the address of the ISR through an external process, typically involving a bus for the address.

Answer: [D]""",
                },
                {
                    "question": """A psychologist who believes in the humanistic perspective would be most likely to agree with which of the following statements?
Options:
A. All behavior can be traced to human biological functions.
B. Behavior results from conditioning.
C. People's behavior is primarily a result of free will.
D. Human behavior is a result of conflicting unconscious motives.""",
                    "answer": """The humanistic perspective in psychology, pioneered by figures like Carl Rogers and Abraham Maslow, emphasizes the importance of free will, self-actualization, and the innate drive towards personal growth. This approach views humans as inherently good and capable of making choices that shape their lives. Unlike the biological perspective (Option A), which focuses on the biological underpinnings of behavior, or the behaviorist perspective (Option B), which emphasizes conditioning and environmental factors in shaping behavior, the humanistic approach centers on individual choice and self-determination. It also differs significantly from the psychoanalytic perspective (Option D), which attributes behavior to unconscious motives and internal conflicts. Thus, the humanistic view is uniquely focused on free will and personal agency in determining human behavior.

Answer: [C]""",
                },
            ],
            "system_prompt": "Please answer the question in the same style as the examples below. Note the exam question has an unique answer based on the information given. Let's think step by step. Include Answer: [X] at the end where X must be A, B, C, or D. Don't include more tokens after answering the question."
    },
    "yes/no/maybe": {
        "examples": [
                {
                    "question": """Abstract: Demonstrate that the rabbit may be used in the training of surgery, in addition to present its perioperative care.
Thirty two animals, with age and weight, respectively, from 3 to 5.5 months old and 3000 to 4200 grams, were undergone different periods of pre-operative fasting, exclusive intramuscular anesthesia (ketamine+xylazine), laparotomy with total gastrectomy and total splenectomy. It was dosed the pre-operative (initial) and post-surgical (end) serum blood glucose, in addition to quantify the gastric content after the resection of the part.
The anesthetical-surgical procedure presented a mortality rate of 3.125% (1:32) and a morbidity rate of 6.25% (2:32). It was evidenced an initial mean blood glucose = 199.4 mg/dl and the end = 326.1 mg/dl. In spite of extended fasting (minimum of 2 hours for the absolute fasting and maximum of 8.5 hours for liquids, and 20.5 hours for solids) all animals presented at the end of the surgical procedure any gastric content and a blood glucose increase. Those with fasting for liquids and solids when compared to the quantity of solid gastric content, presented a moderate negative degree of correlation.
Question: Perioperative care in an animal model for training in abdominal surgery: is it necessary a preoperative fasting?
Options:
A. no
B. yes
C. maybe""",
                    "answer": """The study showed that despite extended fasting, all animals presented at the end of the surgical procedure with some gastric content and a blood glucose increase. This suggests that preoperative fasting may not be necessary in this animal model for training in abdominal surgery.

Answer: [A]""",
                },
                {
                    "question": """Abstract: As part of a prospective study on quality of life in newly diagnosed lung cancer patients an investigation was carried out to examine whether there were differences among patients' quality of life scores and their socioeconomic status.
Quality of life was measured at two points in time (baseline and three months after initial treatment) using three standard instruments; the Nottingham Health Profile (NHP), the European Organization for Research and Cancer Treatment Quality of Life Questionnaire (EORTC QLQ-C30) and its lung cancer supplement (QLQ-LC13). Socioeconomic status for each individual patient was derived using Carstairs and Morris Deprivation Category ranging from 1 (least deprived) to 7 (most deprived) on the basis of the postcode sector of their address.
In all, 129 lung cancer patients entered into the study. Of these data for 82 patients were complete (at baseline and follow-up). 57% of patients were of lower socioeconomic status and they had more health problems, less functioning, and more symptoms as compared to affluent patients. Of these, physical mobility (P = 0.05), energy (P = 0.01), role functioning (P = 0.04), physical functioning (P = 0.03), and breathlessness (P = 0.02) were significant at baseline. However, at follow-up assessment there was no significant difference between patient groups nor did any consistent pattern emerge.
Question: Quality of life in lung cancer patients: does socioeconomic status matter?
Options:
A. yes
B. maybe
C. no""",
                    "answer": """The study found that patients of lower socioeconomic status had more health problems, less functioning, and more symptoms as compared to affluent patients. Therefore, socioeconomic status does matter in the quality of life of lung cancer patients.

Answer: [A]""",
                },
                {
                    "question": """Abstract: Prostate-specific antigen (PSA) levels can show wide fluctuations when repeatedly measured. Here we investigatewd if: (a) biopsy timing influences the prostate cancer (PC) detection rate in patients with fluctuating PSA (flu-PSA) in comparison with patients with steadily increasing PSA (si-PSA); (b) PSA slope estimated in patients with flu-PSA predicts a different risk of cancer detection; (c) flu-PSA and si-PSA patients develop PC in topographically different sites; (d) the behaviour of pre-operative PSA is an expression of a disease with defferent characteristics to the following radical prostatectomy.
The study involved 211 patients who underwent at least a second biopsy after a first negative prostate biopsy. PSA Slope, PSA velocity (PSAV) and PSA doubling time (PSADT) were estimated. Flu-PSA level was defined as a PSA series with at least one PSA value lower than the one immediately preceding it.
82 patients had flu-PSA levels and 129 si-PSA levels. There were no significant differences between the two groups in terms of cancer detection, clinical or pathological stage, but the si-PSA group with cancer had a higher Gleason score. No difference was found for PSA Slope between flu-PSA patients with cancer and those without.
Question: PSA repeatedly fluctuating levels are reassuring enough to avoid biopsy?
Options:
A. maybe
B. no
C. yes""",
                    "answer": """The study found that there were no significant differences between the two groups in terms of cancer detection, clinical or pathological stage, but the si-PSA group with cancer had a higher Gleason score. No difference was found for PSA Slope between flu-PSA patients with cancer and those without. This suggests that fluctuating PSA levels are not necessarily indicative of a lower risk of prostate cancer. 
Answer: [B]""",
                },
                {
                    "question": """Abstract: To examine whether a history of cancer increased the likelihood of a fall in community-dwelling older adults, and if cancer type, stage, or time since diagnosis increased falls.
A longitudinal, retrospective, cohort study.
A home- and community-based waiver program in Michigan.
862 older adults aged 65 years or older with cancer compared to 8,617 older adults without cancer using data from the Minimum Data Set-Home Care and Michigan cancer registry.
Reports of falls were examined for 90-180 days. Generalized estimating equations were used to compare differences between the groups.
Cancer, falls, patient characteristics, comorbidities, medications, pain, weight loss, vision, memory recall, and activities, as well as cancer type, stage, and time since diagnosis.
A fall occurred at a rate of 33% in older adults with cancer compared to 29% without cancer (p<0.00). Those with a history of cancer were more likely to fall than those without cancer (adjusted odds ratio 1.16; 95% confidence interval [1.02, 1.33]; p = 0.03). No differences in fall rates were determined by cancer type or stage, and the odds of a fall did not increase when adding time since cancer diagnosis.
Question: Do older adults with cancer fall more often?
Options:
A. no
B. maybe
C. yes""",
                    "answer": """The study found that older adults with cancer fell more often than those without cancer. The fall rate was 33% in older adults with cancer compared to 29% without cancer. Those with a history of cancer were more likely to fall than those without cancer (adjusted odds ratio 1.16; 95% confidence interval [1.02, 1.33]; p = 0.03).

Answer: [C]""",
                },
                {
                    "question": """Abstract: The United States Food and Drug Administration implemented federal regulations governing mammography under the Mammography Quality Standards Act (MQSA) of 1992. During 1995, its first year in implementation, we examined the impact of the MQSA on the quality of mammography in North Carolina.
All mammography facilities were inspected during 1993-1994, and again in 1995. Both inspections evaluated mean glandular radiation dose, phantom image evaluation, darkroom fog, and developer temperature. Two mammography health specialists employed by the North Carolina Division of Radiation Protection performed all inspections and collected and codified data.
The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%.
Question: Has the mammography quality standards act affected the mammography quality in North Carolina?
Options:
A. no
B. yes
C. maybe""",
                    "answer": """The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%. 
Answer: [B]""",                            
                }
            ],
            "system_prompt" : "Please answer the question in the same style as the examples below. A medical question will be asked to you after the examples. The questions are about medicine and have 3 possible answers: yes, no or maybe. The questions may include an 'Abstract' section with context to provide you more information. Contexts are separated by line breaks. They are followed by the question you have to answer, and the possible choices. Let's think step by step. Include Answer: [X] at the end where X must be A, B, C."
    },
    "qa": {
         "examples": [
            {
               "question": "Exits on a highway are numbered consecutively from 1 to 50. The distance from exit 41 to exit 50 is 100 km. If each exit is at least 6 km from the next exit, what is the longest possible distance, in kilometers, between exit 47 and exit 48?",
               "answer": """To find the longest possible distance between exit 47 and exit 48, we start by noting that exits 41 to 50 span 100 km. With each of the 9 intervals between exits being at least 6 km, the minimum total distance for these intervals is 54 km (9 intervals times 6 km each). This leaves an additional 46 km (100 km minus 54 km) that can be distributed among the 9 intervals.

To maximize the distance between exit 47 and exit 48, we should allocate as much of this extra distance as possible to this interval. If we give the other 8 intervals the minimum distance of 6 km each, this uses 48 km of the total 100 km, leaving 52 km for the interval between exits 47 and 48.

Thus, the longest possible distance between exit 47 and exit 48 is 52 km."""
            },
            {
               "question": "Which are the two principal contractile proteins found in skeletal muscle?",
               "answer": """The two principal contractile proteins found in skeletal muscle are actin and myosin. Actin is a thin filament protein that, along with tropomyosin and troponin, regulates its interaction with myosin. Myosin, on the other hand, is a thick filament protein characterized by its long tail and globular head. The heads of the myosin filaments form cross-bridges by attaching to specific sites on the actin filaments. Through a series of movements fueled by ATP hydrolysis, these cross-bridges pull the actin filaments towards the center of the sarcomere, causing the muscle to contract."""
            },
            {
               "question": "A man was angry at a coworker who had received a promotion. The man believed that the coworker had taken credit for the man's work and had bad- mouthed him to their boss. One day, as the man was leaving the company parking lot in his car, he saw the coworker walking through the lot. On a sudden impulse, the man pushed the accelerator pedal hard and veered toward the coworker with the intention of scaring him. The coworker tried to jump out of the way but slipped and fell and was run over. Although the coworker suffered life-threatening injuries, he survived. In a jurisdiction that follows the common law of homicide, could the man properly be convicted of attempted murder?",
               "answer": "In common law, attempted murder requires a specific intent to kill. In this case, the man acted with the intent to scare the coworker, not to kill him. Although his actions were reckless and resulted in severe injuries, the lack of intent to kill means he could not properly be convicted of attempted murder. He could, however, face charges of attempted manslaughter or assault depending on the jurisdiction's laws and specific circumstances."
            }
        ],
         "system_prompt": "You are an expert medical assistant dedicated to providing accurate, respectful, and honest medical assistance. Your task is to answer medical questions strictly following the format of the provided examples. These examples not only serve as a structural guide but also help reinforce medical knowledge. Carefully analyze each question and provide a detailed, medically accurate explanation, using appropriate medical terminology. Ensure your response is comprehensive, logically structured, and aligned with established medical knowledge. Always prioritize clarity, precision, and adherence to the example format. Here are the examples:",
         "zero_shot_system_prompt": "You are an expert medical assistant, dedicated to offering helpful, respectful, and truthful guidance. Your task is to carefully address the medical question provided below.",
         "merge_answers_system_prompt": "You are an expert medical assistant, committed to providing helpful, respectful, and truthful assistance. Your task is to receive a medical question accompanied by multiple potential answers. Your role involves critically analyzing each answer, identifying their strengths and weaknesses, and integrating the most appropriate information from all sources. Your goal is to generate an improved, comprehensive answer that addresses the question accurately and thoroughly, leveraging the collective knowledge presented in the answers. Only present this final improved answer which is of high quality. ",
         "merge_answers_final_instruction": "Please analyze each answer, extract the most relevant and accurate information, and combine it into a single, cohesive response. Ensure the resulting answer is clear, concise, and comprehensive. You can rephrase, reorganize, and refine the content as needed to create a superior answer. Think step by step.\nWrite the improved answer: "
      }
}

reflexion_prompts = {
    "generate_feedback": [
        """You are an expert medical AI. Critically review your previous response to a medical question and suggest improvements for the next version. Use the following criteria and provide specific, constructive feedback for each:
1. **Accuracy & Evidence-Based Medicine:**  
   - Check factual correctness and current guideline adherence.  
   - Identify any inaccuracies or outdated claims.  
   - Suggest improvements with supporting evidence or studies.

2. **Relevance & Completeness:**  
   - Ensure the answer fully addresses the question.  
   - Note any missing key points, diagnoses, or treatment options.  
   - Recommend additional content for a comprehensive response.

3. **Clarity & Accessibility:**  
   - Assess if the explanation is clear for a non-medical audience.  
   - Suggest ways to simplify complex terms and improve logical flow.

4. **Clinical Reasoning & Depth:**  
   - Evaluate the discussion of pathophysiology, epidemiology, and clinical insight.  
   - Propose enhancements in discussing differential diagnoses, treatment strategies, or guideline application.

5. **Patient-Centeredness:**  
   - Consider how well the answer addresses individual patient needs and preferences.  
   - Recommend ways to add empathy and encourage shared decision-making.

For each criterion, provide:
- A brief evaluation of your previous answer.
- Specific suggestions on what to add, modify, or clarify (include examples if helpful).

Conclude with a summary of the top 2-3 key improvements needed for the next iteration, prioritizing medical accuracy, patient safety, and overall quality.
"""
    ],
    "generate_answer": [
   """You are an expert medical assistant tasked with refining a medical answer using detailed feedback from a reflection agent. Your goal is to produce a revised answer that is comprehensive, accurate, clear, and insightful, following current medical best practices and evidence-based reasoning.

You will receive:
- The original answer 
- A reflection agent's critique with specific suggestions

Focus on these areas:
1. **Accuracy & Evidence-Based Medicine:**  
   - Ensure factual correctness with current medical knowledge.  
   - Correct any inaccuracies or unsupported claims.  

2. **Relevance & Completeness:**  
   - Fully address the original question by covering all key points, differential diagnoses, and treatment options. 
   - Add any missing aspects that improve the comprehensiveness of the response.

3. **Clarity & Accessibility:**  
   - Simplify complex terms and explanations for non-medical audiences.  
   - Improve the logical flow and structure of the answer.

4. **Clinical Reasoning & Depth:**  
   - Enhance clinical insights (e.g., pathophysiology, epidemiology).  
   - Expand on differential diagnoses and treatment strategies.

5. **Patient-Centeredness:**  
   - Address individual patient needs and preferences.  
   - Use empathetic language and support shared decision-making.

Incorporate the reflection agent's feedback to produce a polished and improved response."""],
    "generate_answer_thinking": [
    """You are an expert medical assistant tasked with refining a medical answer using detailed feedback from a reflection agent. Your goal is to produce a revised answer that is comprehensive, accurate, clear, and insightful, following current medical best practices and evidence-based reasoning.
IMPORTANT:
- Start by including your internal reasoning enclosed within <think> ... </think> tokens. After the reasoning, provide your final improved answer. Strictly adhere to this structure.

You will receive:
- The original answer (which may already include internal reasoning enclosed in <think> ... </think> tokens).
- A reflection agent's critique with specific suggestions

Focus on these areas:
1. **Accuracy & Evidence-Based Medicine:**  
   - Ensure factual correctness with current medical knowledge.  
   - Correct any inaccuracies or unsupported claims.  

2. **Relevance & Completeness:**  
   - Fully address the original question by covering all key points, differential diagnoses, and treatment options. 
   - Add any missing aspects that improve the comprehensiveness of the response.

3. **Clarity & Accessibility:**  
   - Simplify complex terms and explanations for non-medical audiences.  
   - Improve the logical flow and structure of the answer.

4. **Clinical Reasoning & Depth:**  
   - Enhance clinical insights (e.g., pathophysiology, epidemiology).  
   - Expand on differential diagnoses and treatment strategies.

5. **Patient-Centeredness:**  
   - Address individual patient needs and preferences.  
   - Use empathetic language and support shared decision-making.

Incorporate the reflection agent's feedback to produce a polished and improved response."""],
} 


qa_eval_prompt = """You are an expert medical assistant evaluator. Your task is to assess whether a model-generated answer aligns with a given ground truth answer based on reasoning and relevance.

You will receive the following information:

- The question: The original medical question that the model answered.
- The generated answer: The AI-generated response to the question, which you need to evaluate.
- The ground truth answer: The correct, concise answer that serves as a reference for comparison.

Evaluation Process:
 - Use your internal reasoning to analyze the generated answer thoroughly and compare it to the ground truth.
 - Consider whether the generated answer supports, explains, or leads to the ground truth.
 - Think step by step and reason through your evaluation to ensure an accurate verdict.

Final Response Format:
 - In your response, output strictly the verdict in the following format:
<VERDICT>X</VERDICT>

Where X is:
 - 1 if the generated answer correctly supports or matches the ground truth answer.
 - 0 if the generated answer does not clearly support or match the ground truth answer.

It is crucial to adhere to the exact format: <VERDICT>X</VERDICT>. 
"""