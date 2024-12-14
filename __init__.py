import os
import json
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

# 新增函数: 从 config.json 文件中读取 API 密钥
def get_gemini_api_key():
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')  # 确保路径是相对于当前文件的
        with open(config_path, 'r') as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY", "").strip()  # 安全获取并去除多余空格
        if not api_key:
            raise ValueError("API key not found in config.json.")
        return api_key
    except Exception as e:
        print(f"Error: Unable to read API key. {str(e)}")
        return ""

class GeminiPromptGeneratorJT:  # 修改类名
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "theme": ("STRING", {"default": "", "multiline": True}),
                "model": (["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],),
                "prompt_length": ("INT", {"default": 200, "min": 0, "max": 5000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "text/generation"
    
    def generate_prompt(self, theme, model, prompt_length, seed):
        # Use the seed to initialize the random number generator
        random.seed(seed)

        # 通过 config.json 获取 API 密钥
        api_key = get_gemini_api_key()
        if not api_key:
            return ("Error: API key is required. Please check config.json.",)
        
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            input_prompt = f"Generate me a prompt for image generator. The theme of the prompt is {theme}. You already created those prompts: {memory}. make sure you generate original prompt. Think about it step by step and make some internal critique. You must keep the length of prompt around {prompt_length} words, and remove the unnecessary two blank spaces or line breaks between each sentence in the Final prompt. Final prompt is encapsulated in <prompt> tags."
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
    "GeminiPromptGeneratorJT": GeminiPromptGeneratorJT  # 更新节点映射
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiPromptGeneratorJT": "Gemini Prompt Generator-JT"  # 更新显示名称映射
}
