# Define all prompt strings here

multiple_choice_prompt = """You are tasked with answering a multiple-choice question based on the given image and question. Select the correct answer from the options (A, B, C, ...). Respond only with the letter of the correct answer.

Question: {question}

Options:
{options}

Your answer:
"""

qa_prompt = "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer with a digit. \n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\n"

multiple_choice_steps_prompt = [
    "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.",
    "In the \"Final Answer\", you should only respond with the letter of the correct choice.",
    "Here is the question and options:",
    "Question: {question}",
    "Options:",
    "{options}",
    "Your response must follow this format:",
    "\"Here is the step-by-step reasoning process:",
    "Step 1: ...",
    "Step 2: ...",
    "Step n: ...",
    "Final Answer: ...\"",
    "Your answer:"
]

multiple_choice_steps_prompt = '\n'.join(multiple_choice_steps_prompt)

beam_sample_prompt = "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"

beam_sample_prompt_no_image = "Answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"

# System prompts for verification
VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if two responses convey the same meaning and correctness.
- If Response 1 matches the correct answer in meaning and accuracy, reply 'True'
- If Response 1 contradicts the correct answer, reply 'False'
- If you cannot confidently determine the relationship, reply 'Unknown'
Remember to ONLY reply with True/False/Unknown."""

SCQA_VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if the provided response match the correct answer for the given question.
- If provided response matches the correct answer in meaning and accuracy, reply 'True'
- If provided response contradicts the correct answer, reply 'False'
- If you cannot confidently determine the relationship, reply 'Unknown'
Remember to ONLY reply with True/False/Unknown.
"""

# Add any additional prompt-related logic here if needed 