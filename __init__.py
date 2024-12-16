import os
import json
import google.generativeai as genai
import random
import re 
import concurrent.futures
import threading

memory = []

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

class GeminiPromptGeneratorJT:  
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "theme": ("STRING", {"default": "", "multiline": True}),
                "override_system_prompt": ("STRING", {"default": "", "multiline": False}),
                "model": (["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-2.0-flash-exp"],),
                "prompt_length": ("INT", {"default": 200, "min": 0, "max": 5000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "timeout": ("INT", {"default": 30, "min": 0, "max": 6000})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "text/generation"
    
    def generate_prompt(self, theme, override_system_prompt, model, prompt_length, seed, timeout):
        # Use the seed to initialize the random number generator
        random.seed(seed)

        # 通过 config.json 获取 API 密钥
        api_key = get_gemini_api_key()
        if not api_key:
            return ("Error: API key is required. Please check config.json.",)
        
        def generate_content_with_timeout():
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(model)

                # 根据 prompt_length 动态调整 input_prompt
                if not override_system_prompt:
                    input_prompt = f"Generate me a prompt for image generator. The theme of the prompt is {theme}. Make sure you generate original prompt. Think about it step by step and make some internal critique. You only need to output generated prompt and nothing else"
                    
                    # 只有当 prompt_length 不为 0 时，才添加长度和 prompt 标签的限制
                    if prompt_length > 0:
                        input_prompt += f" You must keep the length of your generated prompt around {prompt_length} words."
                else:
                    input_prompt = f"The theme is {theme}. Please generate a response strictly following the instruction: {override_system_prompt}."
                    
                    # 只有当 prompt_length 不为 0 时，才添加长度和 prompt 标签的限制
                    if prompt_length > 0:
                        input_prompt += f" You must keep the length of response around {prompt_length} words."

                response = gemini_model.generate_content(input_prompt)
                generated_prompt = response.text.strip()
                
                print("----INPUT----")
                print(input_prompt)

                print("----OUTPUT----")
                print(generated_prompt)
                print("-------------")
                return generated_prompt
                
            except Exception as e:
                return f"Error: Failed to generate prompt. Details: {str(e)}"

        # 使用 concurrent.futures 实现超时
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(generate_content_with_timeout)
            try:
                # 使用 as_completed 和 timeout 参数
                generated_prompt = future.result(timeout=timeout)
                return (generated_prompt,)
            except concurrent.futures.TimeoutError:
                error_message = f"Error: Prompt generation timed out after {timeout} seconds."
                print(error_message)
                return (error_message,)
            except Exception as e:
                error_message = f"Error: Failed to generate prompt. Details: {str(e)}"
                print(error_message)
                return (error_message,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "GeminiPromptGeneratorJT": GeminiPromptGeneratorJT
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiPromptGeneratorJT": "Gemini Prompt Generator-JT"
}
