from openai import OpenAI
from PIL import ImageGrab, Image
import io
from faster_whisper import WhisperModel
import speech_recognition as sr
import pyperclip
import cv2
import pyaudio
import os
import time
import re
import base64
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime
from duckduckgo_search import DDGS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Rich Console
console = Console()

# Initialize OpenAI client
wake_word = 'nova'
openai_client = OpenAI(api_key="INSERT API KEY HERE")

# System message for GPT-4
sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

num_cores = os.cpu_count()
whisper_model = WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=num_cores // 2, num_workers=num_cores // 2)

r = sr.Recognizer()

# Store logs in memory
log_messages = []

def log(message, title, style):
    console.print(Panel(Markdown(f"**{message}**"), border_style=style, expand=False, title=title))
    log_messages.append(f"[{title}] {message}")

def save_log():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}.txt"
    with open(filename, "w") as f:
        for message in log_messages:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(f"Log saved to {filename}")

class EnhancedConversationContext:
    def __init__(self, max_turns=5, similarity_threshold=0.3):
        self.history = []
        self.max_turns = max_turns
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer()

    def add_exchange(self, user_input, assistant_response):
        if self.history:
            similarity = self.calculate_similarity(user_input)
            if similarity < self.similarity_threshold:
                self.clear()  # Clear context if topic changes significantly

        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        if len(self.history) > 2:
            return self.summarize_context()
        else:
            return self.format_context()

    def format_context(self):
        context = ""
        for exchange in self.history:
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        return context.strip()

    def summarize_context(self):
        summary = "Previous conversation summary:\n"
        for exchange in self.history[:-1]:  # Summarize all but the last exchange
            summary += f"- User asked about {exchange['user'][:50]}...\n"
        summary += f"\nMost recent exchange:\nUser: {self.history[-1]['user']}\nAssistant: {self.history[-1]['assistant']}"
        return summary

    def calculate_similarity(self, new_input):
        if not self.history:
            return 0
        previous_inputs = [exchange['user'] for exchange in self.history]
        previous_inputs.append(new_input)
        tfidf_matrix = self.vectorizer.fit_transform(previous_inputs)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        return np.mean(cosine_similarities)

    def clear(self):
        self.history = []

    def remember(self, information):
        self.history.append({"user": "Remember this", "assistant": information})

    def forget(self):
        self.clear()
        return "Previous context has been cleared."

# Initialize the enhanced conversation context
conversation_context = EnhancedConversationContext()

def llm_prompt(prompt, img_context):
    context = conversation_context.get_context()
    if context:
        prompt = f"Previous conversation:\n{context}\n\nCurrent user prompt: {prompt}"
    if img_context:
        prompt += f'\n\nIMAGE CONTEXT: {img_context}'
    
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=convo
    )
    response = chat_completion.choices[0].message
    convo.append(response)
    
    conversation_context.add_exchange(prompt, response.content)
    
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the user\'s clipboard content, '
        'taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond '
        'to the user\'s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"].\n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content


def take_screenshot():
    log("Taking screenshot...", title="ACTION", style="bold blue")
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    return path

def web_cam_capture():
    try:
        import pygame
        import pygame.camera

        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        
        if not cameras:
            log('Error: No cameras found', title="ERROR", style="bold red")
            return None

        cam = pygame.camera.Camera(cameras[0], (640, 480))
        cam.start()
        image = cam.get_image()
        cam.stop()
        pygame.camera.quit()

        path = 'webcam.jpg'
        pygame.image.save(image, path)
        
        pil_string_image = pygame.image.tostring(image, "RGB", False)
        pil_image = Image.frombytes("RGB", (640, 480), pil_string_image)
        
        pil_image.save(path, 'JPEG')

        log("Webcam image captured and saved.", title="ACTION", style="bold blue")
        return path
    except Exception as e:
        log(f'Error capturing webcam image: {str(e)}', title="ERROR", style="bold red")
        return None

def get_clipboard_text():
    log("Extracting clipboard text...", title="ACTION", style="bold blue")
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        log("Clipboard text extracted.", title="ACTION", style="bold blue")
        return clipboard_content
    else:
        log('No clipboard text to copy', title="ERROR", style="bold red")
        return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def vision_prompt(prompt, photo_path):
    log("Generating vision prompt...", title="ACTION", style="bold blue")
    encoded_image = encode_image(photo_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze this image in the context of the following prompt: {prompt}. Provide a detailed description focusing on elements relevant to the prompt."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "high"}}
            ]
        }
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    log("Vision prompt generated.", title="ACTION", style="bold blue")
    return response.choices[0].message.content

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1', voice='nova', response_format='pcm', speed='1.75', input=text
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def duckduckgo_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        log(f"Error in DuckDuckGo search: {str(e)}", title="ERROR", style="bold red")
        return []

def process_search_results(results):
    processed = "Search results:\n\n"
    for i, result in enumerate(results, 1):
        processed += f"{i}. {result['title']}\n   {result['body']}\n   URL: {result['href']}\n\n"
    return processed


def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*(.*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)
    if clean_prompt:
        log(f'USER: {clean_prompt}', title="USER INPUT", style="bold green")
        
        # Check for special commands
        if clean_prompt.lower().startswith("remember "):
            conversation_context.remember(clean_prompt[9:])
            response = "I've remembered that information."
        elif clean_prompt.lower() == "forget context":
            response = conversation_context.forget()
        elif clean_prompt.lower().startswith("search "):
            search_query = clean_prompt[7:]
            search_results = duckduckgo_search(search_query)
            processed_results = process_search_results(search_results)
            response = llm_prompt(prompt=f"Based on the following search results, answer the query: {search_query}\n\n{processed_results}", img_context=None)
        else:
            call = function_call(clean_prompt)
            visual_context = None
            if 'take screenshot' in call:
                photo_path = take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path=photo_path)
            elif 'capture webcam' in call:
                photo_path = web_cam_capture()
                if photo_path:
                    visual_context = vision_prompt(prompt=clean_prompt, photo_path=photo_path)
                else:
                    visual_context = "Failed to capture webcam image."
            elif 'extract clipboard' in call:
                paste = get_clipboard_text()
                clean_prompt = f'{clean_prompt}\n\nCLIPBOARD CONTENT: {paste}'
            
            response = llm_prompt(prompt=clean_prompt, img_context=visual_context)
        
        log(f'ASSISTANT: {response}', title="ASSISTANT RESPONSE", style="bold magenta")
        speak(response)


def start_listening():
    log("Adjusting for ambient noise...", title="ACTION", style="bold blue")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        console.print(Panel("Say 'nova' followed with your prompt.", border_style="bold magenta", title="INSTRUCTIONS"))
    stop_listening = r.listen_in_background(sr.Microphone(), callback)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        log("Listening stopped.", title="ACTION", style="bold blue")
        save_log()

if __name__ == "__main__":
    start_listening()
