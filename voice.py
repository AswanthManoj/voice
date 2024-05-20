from dotenv import load_dotenv
from sseclient import SSEClient
from prompt import SYSTEM_PROMPT
import shutil, wave, httpx, aiohttp, json
import asyncio, requests, os, time, subprocess
from typing import Optional, Dict, List, BinaryIO
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

VALID_STT_MODELS = [
    "nova-2"
]

VALID_TTS_MODELS = [
    "aura-hera-en", # f
    "aura-luna-en", # f
    "aura-stella-en", # f
    "aura-athena-en", # f
    "aura-asteria-en", # f 
    
    "aura-angus-en", # m
    "aura-orion-en", # m
    "aura-arcas-en", # m
    "aura-helios-en", # m
    "aura-perseus-en", # m
]

VALID_LANGUAGE_MODELS = [
    "microsoft/WizardLM-2-8x22B",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8X7B-Instruct-V0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8X7B-DPO"
]


class SpeechToText:
    def __init__(self, model_name:str = "nova-2") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.base_url = "https://api.deepgram.com/v1/listen"
        
        if model_name not in VALID_STT_MODELS:
            raise ValueError(f'Invalid Speech to text model for deepgram')
    
    async def listen(self, audio: BinaryIO):
        try:
            async with httpx.AsyncClient() as client:

                response = await client.post(
                    self.base_url,
                    content=audio.read(),
                    params={"model": self.model_name, "smart_format": "false"},
                    headers={"Content-Type": "audio/*", "Authorization": f"Token {self.api_key}"},
                )
                response.raise_for_status()
                transcript = response.json()
                return transcript['results']['channels'][0]['alternatives'][0]['transcript']

        except httpx.HTTPError as e:
            print(f"HTTP Error in transcribe_audio: {e}")
            return ""
        except Exception as e:
            print(f"Exception in transcribe_audio: {e}")
            return ""


class TextToSpeech:
    def __init__(self, model_name: str="aura-asteria-en") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.base_url = "https://api.deepgram.com/v1/speak"
        
        if model_name not in VALID_TTS_MODELS:
            raise (f"The provided model name `{model_name}` is an invalid model for deepgram.")
    
    async def speak(self, text: str):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json={"text": text,},
                    params={"model": self.model_name},
                    headers={"Content-Type": "application/json", "Authorization": f"Token {self.api_key}"}
                )
                response.raise_for_status()
                return response.content

        except httpx.HTTPError as e:
            print(f"HTTP Error in generate_speech_from_text: {e}")
            return None
        except Exception as e:
            print(f"Exception in generate_speech_from_text: {e}")
            return None


class LanguageModel:
    
    def __init__(self, model_name: str="meta-llama/Llama-3-8b-chat-hf", **kwargs) -> None:
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.temperature=kwargs.get('temperature', 1)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        
        self.messages = [dict(role='system', content=SYSTEM_PROMPT)]
        
        if model_name not in VALID_LANGUAGE_MODELS:
            raise (f"The provided model name `{model_name}` is an invalid language model for together ai.")
        
    def add_user_message(self, text:str) -> str:
        self.messages.append(dict(role='user', content=text))
    
    def add_assistant_message(self, text:str) -> str:
        self.messages.append(dict(role='assistant', content=text))
    
    async def respond(self, text:str):
        self.add_user_message(text)
        
        payload = {
            "top_k": 75,
            "top_p": 0.90,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": 1,
            "model": self.model_name,
            "messages": self.messages,
            "stop": ["<|eot_id|>", "[/INST]", "</s>", "<|im_end|>"],
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = ""
        full_response = ""
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.base_url, json=payload, headers=headers) as partial_response:
                partial_response.raise_for_status()
                async for line in partial_response.aiter_lines():
                    try:
                        json_str = line.split("data:")[1].strip()
                        data_dict = json.loads(json_str)
                        text = data_dict['choices'][0]['text']
                        full_response += text
                        response += text
                        if "." in response or "?" in response or "!" in response:
                            yield response
                            response = ""
                    except:
                        pass
        
        self.add_assistant_message(full_response)