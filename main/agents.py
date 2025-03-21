from llms import BaseLLM, Anthropic
from utils import json_to_dict


class Agent:
    def __init__(self, **kwargs):
        pass

    def generate_response(self, **kwargs) -> str:
        pass
    

PROMPT_AGENT_SUBQS = """
Break down the following question into intermediate sub-questions to approach answering it. 
Provide a list of intermediate sub-questions and respond with JSON format. 
If you cannot produce sub-question then say so. Do not directly answer the following question
 and only return the sub-questions in JSON format. 
 Do not include any other text in your response such as "json in the beginning or end.

Question: {query}
"""

class AgentSubQuestions(Agent):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_response(self, query: str) -> str:
        prompt = PROMPT_AGENT_SUBQS.format(query=query)

        chat = [
            {
                "role": "user",
                "content": f"{prompt}",
            },
        ]
        
        model = self.llm()
        response = model.chat(chat)
        return response.content


PROMPT_AGENT_ANSWER = """
You are a helpful assistant that can answer questions. 
Your answers are concise and insightful, relevant to the question being asked.

Question: {question}
"""


class AgentSubAnswers(Agent):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_response(self, sub_qs: str) -> str:
        # loop over each sub-question and find answer for it using the llm
        
        sub_qs = json_to_dict(sub_qs)
        answers = {}


        for q in sub_qs["sub_questions"]:
            prompt = PROMPT_AGENT_ANSWER.format(question=q)

            chat = [
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ]
        
            model = self.llm()
            response = model.chat(chat)
            answers[q] = response.content
        
        return answers
        # now do using RAG


PROMPT_AGENT_SYNTHESIZE_REPORT = """
You are a helpful assistant that can synthesize a report from a list of question and answers.
Your report must be organised into meaningful headers and sub-headers structured into appropriate themes.

Answers: {answers}
"""




class AgentSynthesizeReport(Agent):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_response(self, answers: str) -> str:
        prompt = PROMPT_AGENT_SYNTHESIZE_REPORT.format(answers=answers)
        
        chat = [
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
        ]
        
        model = self.llm()
        response = model.chat(chat)
        return response.content



if __name__ == "__main__":
    agent = AgentSubQuestions(llm=Anthropic)
    query = 'Summarize Lord of the Rings and the key themes of the book'
    response = agent.generate_response(query)

    agent = AgentSubAnswers(llm=Anthropic)
    response = agent.generate_response(response)

    agent = AgentSynthesizeReport(llm=Anthropic)
    response = agent.generate_response(response)

    print(response)

