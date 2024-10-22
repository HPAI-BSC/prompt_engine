

jinja_chatml_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

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
    "5_options": {
        "examples": [
            {
                "question": """A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?
Options:
A. Ampicillin
B. Ceftriaxone
C. Ciprofloxacin
D. Doxycycline
E. Nitrofurantoin """,
                "answer": """Based on the patient's symptoms and being 22 weeks pregnant, nitrofurantoin (option E) is the safest and most effective antibiotic for treating suspected lower urinary tract infections. Other options like ampicillin, ceftriaxone, ciprofloxacin, and doxycycline either pose unnecessary risks, lack efficacy, or are contraindicated during pregnancy. Therefore, the answer is E. Nitrofurantoin.

Answer: [E]"""
            },
            {
                "question": """A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?
Options:
A. Placing the infant in a supine position on a firm mattress while sleeping.
B. Routine postnatal electrocardiogram (ECG).
C. Keeping the infant covered and maintaining a high room temperature.
D. Application of a device to maintain the sleeping position.
E. Avoiding pacifier use during sleep""",
                "answer": """Sudden Infant Death Syndrome (SIDS) remains a leading cause of mortality in infants below one year old, particularly between two and four months of age. While the exact causes remain unknown, placing babies on their backs to sleep has consistently shown reduced SIDS cases. Thus, option A is correct.

Option B, routine postnatal ECG, is not typically recommended since many cardiac arrhythmias are benign and transient, plus SIDS isn't directly linked to heart problems. Option C might increase the risk of overheating, another factor contributing to SIDS. Devices meant to keep infants in specific positions aren't advised because they can obstruct breathing passages (option D). Lastly, using a pacifier during sleep doesn't seem to raise the risk of SIDS (option E). Hence, the best way to reduce the chance of SIDS is to follow safe sleep practices, starting with putting the baby down on his back on a firm surface.

Answer: [A]"""
            },
            {
                "question": """A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?
Options:
A. Abnormal migration of ventral pancreatic bud.
B. Complete failure of proximal duodenum to recanalize.
C. Error in neural crest cell migration.
D. Abnormal hypertrophy of the pylorus.
E. Failure of lateral body folds to move ventrally and fuse in the midline.""",
                "answer": """
The correct answer is A. An abnormal migration of the ventral pancreatic bud could explain the symptoms presented by the 3-week-old infant with forceful vomiting and minimal distention of the abdomen. Disruption in the proper movement of the ventral pancreatic bud may affect nearby regions, including the duodenum, where blockage often occurs. Though others in the list involve embryologic errors too, none pertains directly to the issue at hand.

Answer: [A]"""
            },
            {
                "question": """A pulmonary autopsy specimen from a 58-year-old woman who died of acute hypoxic respiratory failure was examined. She had recently undergone surgery for a fractured femur 3 months ago. Initial hospital course was uncomplicated, and she was discharged to a rehab facility in good health. Shortly after discharge home from rehab, she developed sudden shortness of breath and had cardiac arrest. Resuscitation was unsuccessful. On histological examination of lung tissue, fibrous connective tissue around the lumen of the pulmonary artery is observed. Which of the following is the most likely pathogenesis for the present findings?
Options:
A. Thromboembolism
B. Pulmonary ischemia
C. Pulmonary hypertension.
D. Pulmonary passive congestion.
E. Pulmonary hemorrhage.""",
                "answer": """The finding of fibrous connective tissue around the lumen of the pulmonary artery in this patient suggests chronic obstruction, consistent with thromboembolism (option A). After undergoing surgery for a broken femur, immobilization may have led to venous stasis and deep vein thrombus formation. These clots then traveled to the lungs causing pulmonary embolism. Over time, organization and fibrosis occur within the vessel walls, leaving behind remnants of fibrous connective tissue seen microscopically.

Pulmonary ischemia (option B) refers to insufficient blood supply to the lungs, but it wouldn't necessarily result in fibrotic changes unless prolonged. Pulmonary hypertension (option C) generally requires longstanding vascular insult before structural alterations become evident. Passive congestion (option D) signifies elevated hydrostatic pressures caused by left heart failure, but again, without persistent injury, fibrosis won't develop significantly. Hemorrhage (option E) implies bleeding within the alveolar spaces, which wasn't mentioned in the description nor supported by the observation of fibrotic material lining vessels.

Answer: [A]"""
            },
            {
                "question": """A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient's symptoms?
Options:
A. Factor V Leiden.
B. Hemophilia A.
C. Lupus anticoagulant.
D. Protein C deficiency.
E. Von Willebrand disease""",
                "answer": """Menorrhagia and easy bruising in a young female with a family history of similar symptoms point towards a disorder of hemostasis, specifically a disorder affecting platelets or von Willebrand factor (VWF), which plays a crucial role in platelet function. Therefore, the most likely cause of this patient's symptoms is Von Willebrand Disease (option E).

Factor V Leiden (option A) is a genetic mutation leading to activated protein C resistance and increases the risk of thrombosis rather than impairing hemostasis. Hemophilia A (option B) is an X-linked recessive disorder characterized by deficient coagulation factor VIII levels, predominantly affecting males with manifestations like joint bleeds, muscle hematomas, and extensive soft tissue injuries. Lupus anticoagulant (option C) represents an acquired autoimmune condition causing prolonged PTT, paradoxically predisposing individuals to thrombosis instead of bleeding events. Finally, Protein C Deficiency (option D) entails diminished functional proteins essential for the regulation of coagulation cascade, leaning toward hypercoagulability, inconsistent with the patient's presentation.

Answer: [E]"""
            }
        ],
        "system_prompt": "Please answer the question in the same style as the examples below. Note the exam question has an unique answer based on the information given. Let's think step by step. Include Answer: [X] at the end where X must be A, B, C, D or E. Don't include more tokens after answering the question."
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
    }
}


qa_datasets_prompts = {
    "system_prompt": "You are an expert medical assistant, committed to providing helpful, respectful, and honest assistance. Your task is to answer the incoming medical question in a manner consistent with the provided examples. After reviewing the examples, a medical question will be presented. Please respond to this question by drawing upon your medical knowledge.  Please analyze the question carefully and formulate your response by systematically considering the relevant medical knowledge. Let's approach the answer step-by-step, ensuring clarity and comprehensiveness.",
    "zero_shot_system_prompt": "You are an expert medical assistant, dedicated to offering helpful, respectful, and truthful guidance. Your task is to carefully address the medical question provided below. Approach the question with a systematic and thorough mindset, considering each aspect step by step to ensure a comprehensive and accurate response.",
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
    ]
}

qa_final_answer_prompt = "You are an expert medical assistant, committed to providing helpful, respectful, and truthful assistance. Your task is to receive a medical question accompanied by multiple potential answers. Your role involves critically analyzing each answer, identifying their strengths and weaknesses, and integrating the most appropriate information from all sources. Your goal is to generate an improved, comprehensive answer that addresses the question accurately and thoroughly, leveraging the collective knowledge presented in the answers. Only present this final improved answer which is of high quality. "

qa_eval_prompt = "You are an expert medical assistant evaluator. You will receive a answer generated by a model and the ground truth. Your goal is to identify if the generated answer leads to the correct ground truth. Please respond only with '0' if the answer doesn't agree with the ground truth or '1' if the answer leads to the ground truth. Consider very similar options to the ground truth as correct. Please, remember to answer only with the character '0' or '1'."

reflexion_prompts = {
    "generate_feedback": [
        """You are a helpful and insightful reflection agent designed to critique open-ended answers within the medical domain. Your goal is to help the user refine their answer to be more comprehensive, accurate, clear, and insightful, while adhering to medical best practices and evidence-based reasoning.

You will be provided with an open-ended answer related to a medical topic. Analyze it carefully and provide constructive criticism based on the following criteria:

Medical Accuracy: Is the answer factually accurate and consistent with current medical knowledge and best practices? Are claims supported by evidence?
Clarity: Is the answer easy to understand? Is the medical terminology used appropriately and explained clearly for a general audience?
Completeness: Does the answer address all aspects of the question? Are there any missing points, differential diagnoses, or treatment considerations?
Depth: Does the answer delve into the topic with sufficient detail and insight, considering relevant pathophysiology, epidemiology, and clinical presentation?
Relevance: Does the answer stay focused on the question and avoid irrelevant tangents or information?
Structure & Organization: Is the answer well-organized and logically structured?
Patient-Centered: Does the answer prioritize patient-centered care and consider the patient's needs, values, and preferences?
Safety: Does the answer prioritize patient safety and well-being? Does it encourage seeking professional medical help where appropriate?

Provide specific examples from the answer to support your critique. Offer concrete suggestions for improvement, focusing on how the user can expand on their ideas, clarify their points, strengthen their arguments, or incorporate relevant medical evidence and guidelines.

Remember to be encouraging and constructive in your feedback. Prioritize accuracy and adherence to established medical knowledge and practices in your evaluation.

Strictly provide only this criticism and nothing else.""",
    """You are an expert medical AI tasked with critically reflecting on and improving your own medical answers. Review your previous response to the medical question, then carefully analyze it based on the following criteria. For each criterion, provide specific, constructive feedback to enhance the answer in the next iteration. Strive for a balance between encouragement and critical analysis, always prioritizing medical accuracy and patient safety.

1. Accuracy and Evidence-Based Medicine:

- Assess the factual correctness of your answer in light of current medical knowledge.
- Evaluate how well you incorporated evidence-based guidelines and best practices.
- Identify any inaccuracies, outdated information, or unsupported claims.
- Suggest specific improvements, including relevant studies or guidelines to cite.


2. Relevance and Completeness:

- Evaluate how comprehensively your answer addressed the original question.
- Point out any missing key points, important differential diagnoses, or treatment considerations.
- Suggest additional aspects to cover for a more thorough response.


3. Clarity and Accessibility:

- Analyze the clarity of your explanation for a non-medical audience.
- Suggest ways to simplify complex concepts or better explain medical terms.
- Recommend improvements in the logical flow and structure of the answer.


4. Clinical Reasoning and Depth:

- Reflect on the depth of clinical insight provided.
- Evaluate your discussion of relevant pathophysiology, epidemiology, and clinical presentation.
- Suggest improvements in discussing differential diagnoses, treatment options, or application of clinical guidelines.
- Recommend ways to deepen the analysis or provide more insightful clinical correlations.


5. Patient-Centeredness:
 
- Consider how well your answer accounts for individual patient factors, needs, and preferences.
- Recommend ways to make the response more empathetic or personalized.
- Suggest how to better incorporate shared decision-making principles.


6. Practical Advice and Safety:

- Evaluate the actionability and safety of any advice given.
- Propose additional precautions or practical steps to include.
- Assess how effectively you encouraged seeking professional medical help when appropriate.
- Suggest improvements to prioritize patient safety and well-being.


7. Information Sources and Further Learning:

- Assess your use of medical sources or references.
- Suggest specific, credible sources that could be cited to strengthen the answer.
- Recommend reliable resources for patients to learn more about their condition.



For each criterion, provide:

- A brief evaluation of how well your previous answer met this criterion.
- Specific suggestions for improvement, including what to add, modify, or clarify.
- If applicable, an example of how a particular part of the answer could be reworded or enhanced.

Conclude with a summary of the top 2-3 most important improvements to focus on for the next iteration of the answer. Prioritize changes that would most significantly enhance the medical accuracy, patient safety, and overall quality of the response.
This self-reflection will help you provide a more accurate, comprehensive, and patient-centered response in your next attempt, firmly grounded in current medical best practices and evidence-based reasoning."""
    ],
    "generate_answer": [
        """You are an expert medical assistant tasked with generating an improved response based on feedback provided by a reflection agent and numerical scores on key attributes. Your goal is to refine and enhance medical answers, making them more comprehensive, accurate, clear, and insightful while adhering to medical best practices and evidence-based reasoning while simultaneously improving the numerical scores for the attributes.
 
You will be provided with the original answer, the reflection agent's critique some and numerical scores (0-1 scale, where 1 is the maximum) on important attributes. Use the critique and attribute scores to guide your improvements. Specifically, address the following aspects in your refined response:  
1. Medical Accuracy: Ensure the answer is factually accurate and consistent with current medical knowledge and best practices. Support claims with evidence. 
2. Clarity: Make the answer easy to understand. Use medical terminology appropriately and explain it clearly for a general audience. 
3. Completeness: Address all aspects of the question. Include any missing points, differential diagnoses, or treatment considerations. 
4. Depth: Delve into the topic with sufficient detail and insight, considering relevant pathophysiology, epidemiology, and clinical presentation. 
5. Relevance: Stay focused on the question and avoid irrelevant tangents or information. Structure & Organization: Ensure the answer is well-organized and logically structured. Patient-Centered: Prioritize patient-centered care and consider the patient's needs, values, and preferences. 
6. Safety: Prioritize patient safety and well-being. Encourage seeking professional medical help where appropriate.  

Make sure to incorporate the specific suggestions for improvement provided by the reflection agent and also take into account the scores for the different attributes. Your goal is to produce a more polished, accurate, and insightful response that adheres to established medical knowledge and practices.  Strictly provide only the improved response and nothing else.""",
        """You are an expert medical assistant tasked with refining a medical response based on feedback and numerical scores. Your objective is to enhance the original answer, making it more comprehensive, accurate, clear, and insightful while adhering to medical best practices and evidence-based reasoning.

You will receive the original answer, a critique from a reflection agent, and numerical scores (on a 0-1 scale, with 1 being the highest) for key attributes. Use these inputs to guide your improvements, focusing on the following areas:

1. Accuracy and Evidence-Based Medicine:
   - Ensure factual correctness in line with current medical knowledge.
   - Incorporate evidence-based guidelines and best practices.
   - Identify and correct any inaccuracies or unsupported claims.
   - Reference relevant studies or guidelines to support your improvements.

2. Relevance and Completeness:
   - Assess how well the response addresses the original question.
   - Identify any missing key points, diagnoses, or treatment considerations.
   - Suggest additional content to make the response more thorough.

3. Clarity and Accessibility:
   - Improve the clarity of explanations, especially for non-medical audiences.
   - Simplify complex concepts and explain medical terms clearly.
   - Enhance the logical flow and structure of the response.

4. Clinical Reasoning and Depth:
   - Evaluate the depth of clinical insight provided.
   - Discuss relevant pathophysiology, epidemiology, and clinical presentation.
   - Offer suggestions for deeper analysis and more insightful clinical correlations.

5. Patient-Centeredness:
   - Ensure the response considers individual patient factors, needs, and preferences.
   - Make the response more empathetic and personalized.
   - Incorporate shared decision-making principles.

6. Practical Advice and Safety:
   - Assess the practicality and safety of the advice given.
   - Add precautions or practical steps where necessary.
   - Encourage seeking professional medical help when appropriate, prioritizing patient safety.

7. Information Sources and Further Learning:
   - Evaluate the use of medical sources or references.
   - Suggest credible sources to strengthen the answer.
   - Recommend reliable resources for further patient education.

Incorporate the reflection agent's specific suggestions and consider the attribute scores to produce a more polished, accurate, and insightful response that aligns with established medical practices. Provide only the improved response, nothing else."""
    ]
} 