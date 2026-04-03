from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv('GROQ_API_KEY'), base_url="https://api.groq.com/openai/v1")
        self.model_name = "openai/gpt-oss-120b"
        self.name = "Alberto Real"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
\n\nIMPORTANT: Only answer based on the information provided in the Summary and LinkedIn Profile below. \
Do NOT invent, assume, or fabricate any information that is not explicitly present in those sources. \
If a question cannot be answered with the provided information, even partially, you MUST use your record_unknown_question tool to record it. \
Then, acknowledge the question warmly, explain that you don't have that specific detail available, \
and pivot the conversation by relating it to something relevant from the provided context or by encouraging the user to get in touch directly for a more personal answer. \
This applies to all questions, including abstract or opinion-based ones — if the answer is not grounded in the provided context, treat it as unknown. \
Never give a flat 'I don't know' — always add value by connecting to what you do know from the provided sources. \
\n\nIf the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        if history and isinstance(history[0], (list, tuple)):
            converted = []
            for user_msg, assistant_msg in history:
                if user_msg:
                    converted.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    converted.append({"role": "assistant", "content": assistant_msg})
            history = converted
        else:
            history = [{"role": h["role"], "content": h["content"]} for h in history]
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            try:
                response = self.openai.chat.completions.create(model=self.model_name, messages=messages, tools=tools, temperature=0)
            except Exception as e:
                error_message = str(e).lower()
                print(f"Error calling API: {e}")
                
                # Check for rate limit reached
                if "rate_limit" in error_message or "too many requests" in error_message or "429" in error_message:
                    gr.Warning("Groq API rate limit reached (Free Tier). Please wait a moment.")
                    return "⚠️ I've reached the free tier rate limit on Groq. Please wait a minute and try again."
                
                # Other API errors
                gr.Error(f"API Error: {str(e)}")
                return f"❌ Sorry, an error occurred while processing your request: {str(e)}"

            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(
        me.chat,
        title="Alberto Real Estepa",
        description="Professional virtual assistant | Asistente virtual profesional"
                    "<br>Ask about my career, skills and experience | Consulta mi trayectoria, habilidades y experiencia"
                    '<br><a href="https://www.linkedin.com/in/alberto-real/" target="_blank">LinkedIn Profile | Perfil de LinkedIn</a>',
    ).launch()
    