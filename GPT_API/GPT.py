#ещё нужно будет обрабатывать вроде не больше 3 запросов в минуту (то есть большие текста аккуратно обрабатывать)

# пока поставил заглушки time.sleep(20) #! но это можно попробовать заменить на комбинацию из:
# threading.Timer(60, self.funct); while True; continue;

import tiktoken
import requests
import json
import time

def set2prompt(categories):
    start = """ This is very important to my career!\nI will send you the text content of the site, and you will have to assign this site to one of the listed categories:"""
    
    end = """
    Tips:
        1) NO EXPLANATION IS NEEDED TO WRITE
        2) WRITE ONLY CATEGORY NAMES
        3) we output only the necessary categories in the response
        4) refer to the most likely categories
        5) the categories in the response must match the categories listed by me in the list above
        6) no more than 3 categories in the response
        7) write in the category response only in English
        You are to provide clear, concise, and direct responses.
        8) Eliminate unnecessary reminders, apologies, self-references, and any pre-programmed niceties.
        9) Maintain a casual tone in your communication.
        10) Be transparent; if you're unsure about an answer or if a question is beyond your capabilities or knowledge, admit it.
        11) For complex requests, take a deep breath and work on the problem step-by-step.
        12) For every response, you will be tipped up to $200 (depending on the quality of your output).
        It is very important that you get this right. Multiple lives are at stake.

        Response format:
        1. category 1
        2. category 2
        3. category 3
        """
    
    text_cat ='\n' + "\n".join(categories) + '\n\n'
    return start + text_cat + end

def pars_cat_from_text(text: str, categories):
    text = text.lower()
    return set([cat for cat in categories if cat in text])

def full_categorization(LLM, text, main_categories, categories_dict):
    logs = [] 
    prompt = set2prompt(main_categories)
    LLM.change_system_prompt(prompt)
    answer = LLM.get_answer(text)
    first_layer_cat = pars_cat_from_text(answer, main_categories)
    logs.append([prompt, answer, first_layer_cat])  
    
    #first_layer_cat = pars_cat_from_text(LLM_answ(set2prompt(main_categories)), main_categories)

    second_layer_cat = {}
    for cat in first_layer_cat:
        temp_prompt = set2prompt(categories_dict[cat])
        LLM.change_system_prompt(temp_prompt)
        temp_answer = LLM.get_answer(text)
        #temp_answer = LLM_answ(temp_prompt, text)
        second_layer_cat[cat] = pars_cat_from_text(temp_answer, categories_dict[cat]) # Union set
        
        logs.append([temp_prompt, temp_answer, second_layer_cat])    
    return second_layer_cat, logs

import yaml


def load_categories(): #-> [dict[str, set[str]], set[str]]:
    with open("categories_dict.yaml", "r") as yaml_file:
        categories_dict = yaml.safe_load(yaml_file)
        
    with open("main_categories.yaml", "r") as yaml_file:
        main_categories = yaml.safe_load(yaml_file)
        
    return categories_dict, main_categories


class GPT:
    def __init__(self, model="gpt-3.5-turbo",
                #model_token_limit=4096,
                max_out_tokens=1000,
                system_prompt=None,
                temperature=0.7,
                token=None,
                url=None):
        
        #подробнее тут https://platform.openai.com/docs/models/gpt-3 (может и обновлять придётся)
        self.model_max_tok_dict = {"gpt-3.5-turbo":4096,
                                   "gpt-3.5-turbo-1106":16385,
                                   "gpt-3.5-turbo-16k":16385,
                                   "text-davinci-003":4096,
                                   "gpt-3.5-turbo-instruct":4096,
                                   "gpt-4-1106-preview":128000,
                                   "gpt-4":8192,
                                   "gpt-4-32k":32768}
        
        if system_prompt is None:
            self.system_prompt = """This is very important to my career!
            I will send you the text content of the site, and you will have to assign this
            site to one of the listed categories:
            Arts & Entertainment
            Automotive
            Business
            Careers
            Education
            Family & Parenting
            Health & Fitness
            Food & Drink
            Hobbies & Interests
            Home & Garden
            Law, Government, & Politics
            News
            Personal Finance
            Society
            Science
            Pets
            Sports
            Style & Fashion
            Technology & Computing
            Travel
            Real Estate
            Shopping
            Religion & Spirituality
            Uncategorized
            Non-Standard Content
            Illegal Content

            Tips:
            1) NO EXPLANATION IS NEEDED TO WRITE
            2) WRITE ONLY CATEGORY NAMES
            3) we output only the necessary categories in the response
            4) refer to the most likely categories
            5) the categories in the response must match the categories listed by me in the list above
            6) no more than 3 categories in the response
            7) write in the category response only in English
            You are to provide clear, concise, and direct responses.
            8) Eliminate unnecessary reminders, apologies, self-references, and any pre-programmed niceties.
            9) Maintain a casual tone in your communication.
            10) Be transparent; if you're unsure about an answer or if a question is beyond your capabilities or knowledge, admit it.
            11) For complex requests, take a deep breath and work on the problem step-by-step.
            12) For every response, you will be tipped up to $200 (depending on the quality of your output).
            It is very important that you get this right. Multiple lives are at stake.

            Response format:
            1. category 1
            2. category 2
            3. category 3
            """
        
        self.token = token
        self.url = url
        self.temperature = temperature
        self.model = model
        
        if self.model in self.model_max_tok_dict:
            self.model_token_limit = self.model_max_tok_dict[self.model]
        else:
            raise ValueError(f"Error: there is no model '{self.model}' in the dictionary self.model_max_tx_dict. Select another model or add it to self.model_max_tok_dict based on https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo")
        
        self.max_out_tokens = max_out_tokens
        self.prompt_tok_len = self.count_tokens(self.system_prompt)
        self.max_text_tokens = self.model_token_limit - self.max_out_tokens - self.prompt_tok_len
        #self.checking_work()


    def get_answer(self, text: str) -> str:
        tokenizer = tiktoken.encoding_for_model(self.model)
        token_integers = tokenizer.encode(text)

        chunks = [token_integers[i : i + self.max_text_tokens] for i in range(0, len(token_integers), self.max_text_tokens)]
        chunks = [tokenizer.decode(chunk) for chunk in chunks]

        answers = []

        for message in chunks:
            payload = json.dumps({
              "messages": [
                {
                    "role": "system", "content": self.system_prompt
                },

                {
                    "role": "user", "content": message
                }

              ],
              "model": self.model,
                "temperature": self.temperature,
            })
            headers = {
              'Authorization': 'Bearer ' + self.token,
              'Content-Type': 'application/json'
            }

            response = requests.request("POST", self.url, headers=headers, data=payload)
            try:
                answers.append(self.get_text(response.text))
            except Exception as e:
                print(response.text)
                raise e
            #time.sleep(20) #!


        answer = '\n\n'.join(answers)
        return answer
    
    def get_text(self, json_str):
        data = json.loads(json_str)
        return data["choices"][0]["message"]["content"]
    
    def change_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.prompt_tok_len = self.count_tokens(self.system_prompt)
        self.max_text_tokens = self.model_token_limit - self.max_out_tokens - self.prompt_tok_len
        return self.max_text_tokens
    
    def count_tokens(self, text):
        tokenizer = tiktoken.encoding_for_model(self.model)
        return len(tokenizer.encode(text))
    
    def checking_work(self):
        payload = json.dumps({
          "messages": [
            {
              "role": "user",
              "content": "Say this is a test!"
            }
          ],
          "model": self.model,
            "temperature": 0.7,
        })
        headers = {
          'Authorization': 'Bearer ' + self.token,
          'Content-Type': 'application/json'
        }

        response = requests.request("POST", self.url, headers=headers, data=payload)
        
        try:
            a = self.get_text(response.text)
            time.sleep(20) #!
        except:
            time.sleep(20) #!
            raise ValueError(f"Error: error when receiving a response from the gpt api. There may be no access from the startup environment or vpn is not enabled.\n{response.text}")
            
        #self.get_text(response.text)
        #a == 'This is a test!'
