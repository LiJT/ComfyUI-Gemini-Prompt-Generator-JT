import os
import google.generativeai as genai
import random
import re 

memory = []

def extract_prompt(text):
    pattern = r'<prompt>(.*?)</prompt>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    
class MagifactoryPromptGenerator:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "theme": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "gemini-1.5-flash-8b-exp-0827", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "text/generation"
    
    



    def generate_prompt(self, theme, api_key, model, seed):
        # Use the seed to initialize the random number generator
        random.seed(seed)
        
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            input_prompt = f"Generate me a prompt for image generator. The theme of the prompt is {theme}. You already created those prompts: {memory}. make sure you generate original prompt. Think about it step by step and make some internal critique. Final prompt is encapsulated in <prompt> tags"
            response = gemini_model.generate_content(input_prompt)
            generated_prompt = response.text.strip()
            print("-------------")
            print(generated_prompt)
            memory.append(generated_prompt)
            display_text = f"Theme: {theme}\nSeed: {seed}\n\nGenerated Prompt:\n{generated_prompt}"
            extracted_prompt = extract_prompt(generated_prompt)
            print(extracted_prompt)
            print("-------------")

            return (extracted_prompt,)
        except Exception as e:
            error_message = f"Error: Failed to generate prompt. Please check your API key and model name. Details: {str(e)}"
            print(error_message)
            return (error_message,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MagifactoryPromptGenerator": MagifactoryPromptGenerator
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagifactoryPromptGenerator": "Magifactory Prompt Generator"
}

