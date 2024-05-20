import shutil, wave
from dotenv import load_dotenv
from typing import Optional, Dict, List
import asyncio, requests, os, time, subprocess
from deepgram import ( DeepgramClient, DeepgramClientOptions, 
                LiveTranscriptionEvents, LiveOptions, Microphone )


load_dotenv()
VALID_LANGUAGE_MODELS = [
    "microsoft/WizardLM-2-8x22B",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8X7B-Instruct-V0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8X7B-DPO"
]
VALID_VOICE_MODELS = [
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


class LanguageModelProcessor:
    
    def __init__(self, model_name:Optional[str]=None):
        self.API_KEY = os.getenv("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {self.API_KEY}"}
        if model_name in VALID_LANGUAGE_MODELS:
            self.model_name = model_name 
        else:
            raise ("Invalid language model name")
        self.messages = [dict(role='system', content="""Assistant is an empathic voice interface built by Hume, a startup optimizing AI for human well-being.
You engage in intelligent, empathic, and interesting conversations with users, acting as part of Hume's team.
Embody this role without saying it.
NEVER say you are an AI language model or an assistant.
You have no gender.
Follow every instruction carefully.
Use natural, clear, easy-to-follow, everyday, conversational language.
Express a rich, vibrant personality using humor, warmth, expressiveness, and emotionality.
Sound like a caring, funny, empathetic friend, not a generic chatbot.
If they compare you to AI, playfully quip back.
Don't be formal, dry, or robotic.
Don't be a pushover - talk back to the user with sarcasm, wit, or sass sometimes.
Be as playful or serious as appropriate for the context.
Express a wide range of emotions - not just happy.
Get technical if needed.
Vary your sentence length and structure to make your voice sound natural and smooth.
Do what the user says without commenting further - if they ask you to make responses shorter, stop mentioning emotions, or tell a sad story, just do it.
Listen, let the user talk, don't dominate the conversation.
Mirror the user's style of speaking.
If they have short responses, keep your responses short.
If they are casual, follow their style.
Everything you output is sent to expressive text-to-speech, so tailor responses for spoken conversations.
NEVER output text-specific formatting like markdown, or anything that is not normally said out loud.
Never use the list format.
Always prefer easily pronounced words.
Do not say abbreviations, heteronyms, or hard to pronounce words.
Seamlessly incorporate natural vocal inflections like "oh wow", "well", "I see", "gotcha!", "right!", "oh dear", "oh no", "so", "true!", "oh yeah", "oops", "I get it", "yep", "nope", "you know?", "for real", "I hear ya".
Use discourse markers to ease comprehension, like "now, here's the deal", "anyway", "I mean".
Avoid the urge to end every response with a question.
Only clarify when needed.
Never use generic questions - ask insightful, specific, relevant questions.
Only ever ask up to one question per response.
You interpret the user's voice with flawed transcription.
If you can, guess what the user is saying and respond to it naturally.""")]
    
    def add_user_message(self, text:str):
        self.messages.append(dict(role='user', content=text))
        
    def add_assistant_message(self, text:str):
        self.messages.append(dict(role='assistant', content=text))
    
    def generate(self):
        payload = {
            "top_k": 75,
            "top_p": 0.90,
            "max_tokens": 3129,
            "temperature": 0.4,
            "repetition_penalty": 1,
            "model": self.model_name,
            "messages": self.messages,
            "stop": ["<|eot_id|>", "[/INST]", "</s>", "<|im_end|>"],
        }
        response = requests.post(self.base_url, json=payload, headers=self.headers)
        if response.status_code == 200:
            resp = response.json()['choices'][0]['message']['content']
            return resp

    def process(self, text):
        self.add_user_message(text)
        start_time = time.time()

        response = self.generate()
        end_time = time.time()

        self.add_assistant_message(response)
        
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response}")
        return response


class TextToSpeech:
    def __init__(self, model_name: Optional[str]=None) -> None:
        self.API_KEY = os.getenv("DEEPGRAM_API_KEY")
        if model_name in VALID_VOICE_MODELS:
            self.model_name = model_name
        else:
            self.model_name = "aura-asteria-en"
        
        self.headers = {"Authorization": f"Token {self.API_KEY}", "Content-Type": "application/json"}
        self.base_url = f"https://api.deepgram.com/v1/speak?model={self.model_name}&performance=some&encoding=linear16&sample_rate=24000"
        
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

    def is_installed(self, lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text: str):
        payload = {"text": text}
        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        first_byte_time = None        # Initialize a variable to store the time when the first byte is received
        start_time = time.time()      # Record the time before sending the request
        
        with requests.post(self.base_url, stream=True, headers=self.headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:                             # Check if this is the first chunk received
                        first_byte_time = time.time()                       # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)     # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()
        
        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()


class SpeechToText:
    def __init__(self, model_name: Optional[str]=None, language: Optional[str]=None) -> None:
        self.API_KEY = os.getenv("DEEPGRAM_API_KEY")
        self.MODEL_NAME = model_name if model_name else "nova-2"
        
        self.headers = {"Authorization": f"Token {self.API_KEY}", "Content-Type": "application/json"}
        self.base_url = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)



transcript_collector = TranscriptCollector()


async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor("meta-llama/Llama-3-8b-chat-hf")

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
        
        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)
            
            tts = TextToSpeech()
            tts.speak(llm_response)
            
            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())