import os
import json
import google.generativeai as genai
import random
import re 
import concurrent.futures
import threading
import time  # 添加time模块用于超时处理
from collections import deque

class ComfyUITimeoutError(Exception):
    pass

class ComfyUIAPIError(Exception):
    pass

# 提示词模板常量 - 便于用户自定义修改
PROMPT_TEMPLATES = {
    # 基础提示词模板（无图像）
    "base_with_memory": "Generate me a prompt for image generator. The theme of the prompt is {theme}. You already created those prompts: {history}. Make sure you generate original prompt. Think about it step by step and make some internal critique.",
    "base_without_memory": "Generate me a prompt for image generator. The theme of the prompt is {theme}. Think about it step by step and make some internal critique.",
    
    # 图像提示词模板（有图像）
    "image_with_memory": "Generate me a prompt for image generator based on the provided image(s). The theme of the prompt is {theme}. You already created those prompts: {history}. Make sure you generate original prompt that describes and expands on what you see in the image(s). Think about it step by step and make some internal critique.",
    "image_without_memory": "Generate me a prompt for image generator based on the provided image(s). The theme of the prompt is {theme}. Think about it step by step and make some internal critique.",
    
    # 自定义提示词模板前缀
    "custom_with_images": "The theme is {theme}. I'm providing you with image(s) for reference. Please generate a response strictly following the instruction: {custom_prompt}.",
    "custom_without_images": "The theme is {theme}. Please generate a response strictly following the instruction: {custom_prompt}.",
    
    # 限制性后缀
    "length_restriction": " You must keep the length of your generated prompt around {length} words.",
    "no_length_restriction": "",
    "general_restriction": " **Restrictions: Do not explain your prompt; just output the prompt directly. Do not output any other non-prompt text; only output the prompt itself, and do not include anything like 'Here is the prompt' or similar words.**",
    "english_only_restriction": " Your answer must be in English only."
}

# 使用deque替代list，并设置最大长度为15
prompt_history = deque(maxlen=15)  # 将memory重命名为prompt_history
last_memory_mode = None  # 添加一个变量来跟踪上一次的内存模式

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

# 将图像转换为Pillow图像的辅助函数
def images_to_pillow(image):
    from PIL import Image
    import numpy as np
    
    if image is None:
        return []
        
    # 转换张量为PIL图像
    result = []
    if len(image.shape) == 4:  # 批量图像
        for i in range(image.shape[0]):
            img = image[i].numpy()
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            result.append(img)
    else:  # 单个图像
        img = image.numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        result.append(img)
    
    return result

class GeminiPromptGeneratorJT:  
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "theme": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "输入想要生成的提示词主题"
                }),
                "override_system_prompt": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "自定义提示词。您可以使用以下变量：\n{theme} - 当前主题\n{prompt_history}或{memory} - 历史提示记录(当Memory设置为Disable时将替换为空列表[])\n{prompt_length} - 提示长度设置\n{seed} - 当前种子值\n例如：'为{theme}生成一个图像提示词，历史记录：{prompt_history}'"
                }),
                "model": (["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-3-flash-preview"], {
                    "tooltip": "模型名称 + 免费层速率限制（官方配额会动态调整，以 AI Studio 为准）：\n• gemini-2.5-flash-lite：15 RPM / 1M TPM / 1000 RPD\n• gemini-2.5-flash：10 RPM / 250K TPM / 250 RPD\n• gemini-2.5-pro：2 RPM / 125K TPM / 50 RPD\n• gemini-2.0-flash-lite：30 RPM / 1M TPM / 200 RPD\n• gemini-2.0-flash：15 RPM / 1M TPM / 200 RPD\n• gemini-3-flash-preview：预览模型，免费限额请以 AI Studio 实时配额为准\n参数：RPM=每分钟请求数，TPM=每分钟Token数，RPD=每日请求数"
                }),
                "enable_memory": ("BOOLEAN", {
                    "default": False,  # 修改默认值为False
                    "tooltip": "启用或禁用历史记忆功能。启用时会记住之前生成的提示，避免重复；禁用时每次生成独立提示"
                }),
                "english_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用后将强制生成的提示词只使用英文。会在发送给Gemini的指令中添加英文限制要求"
                }),
                "prompt_length": ("INT", {
                    "default": 120,  # 修改默认值为120
                    "min": 0, 
                    "max": 5000,
                    "tooltip": "控制生成提示词的长度(单词数)。设为0表示不限制长度"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子值，相同的种子会产生相似的结果"
                }),
                "timeout": ("INT", {
                    "default": 15,  # 修改默认值为15
                    "min": 0, 
                    "max": 6000,
                    "tooltip": "API请求超时时间(秒)。如果在指定时间内未收到响应，将中断请求"
                })
            },
            "optional": {
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "image_3": ("IMAGE", )
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "text/generation"
    
    def generate_prompt(self, theme, override_system_prompt, model, enable_memory, english_only, prompt_length, seed, timeout, image_1=None, image_2=None, image_3=None):
        # 将布尔值转换为原来的字符串格式以保持兼容性
        memory = "Enable" if enable_memory else "Disable"
        
        # Use the seed to initialize the random number generator
        random.seed(seed)
        
        # 跟踪内存模式的变化并清除历史记录
        global last_memory_mode
        if last_memory_mode is None:
            # 第一次运行
            if memory == "Disable":
                # 如果第一次运行时就是Disable模式，确保历史记录为空
                prompt_history.clear()
                print("First run with Memory Disable mode. History cleared.")
        elif last_memory_mode != memory:
            # 当模式发生变化时
            if memory == "Enable":
                # 当从Disable切换到Enable模式时，清空历史记录
                prompt_history.clear()
                print("Memory mode changed from Disable to Enable. History cleared.")
            else:
                # 当从Enable切换到Disable模式时，打印通知但不清空历史记录(因为disable模式不使用历史记录)
                print("Memory mode changed from Enable to Disable. New prompts will not be recorded.")
        
        # 更新当前的内存模式
        last_memory_mode = memory

        # 通过 config.json 获取 API 密钥
        api_key = get_gemini_api_key()
        if not api_key:
            raise ComfyUIAPIError("API key is required. Please check config.json.")
        
        # 使用线程级别的取消标志
        cancel_event = threading.Event()
        result = [None]  # 使用列表来存储结果
        error = [None]   # 使用列表来存储错误

        def generate_content_worker():
            try:
                # 配置API并创建模型
                genai.configure(api_key=api_key)
                
                # 设置安全配置以避免被审核过滤
                safety_settings = {
                    "harassment": "block_none",
                    "hate_speech": "block_none", 
                    "sexually_explicit": "block_none",
                    "dangerous_content": "block_none"
                }
                
                # 创建生成模型实例并设置生成参数
                generation_config = {
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "response_mime_type": "text/plain"
                }
                
                try:
                    # 尝试使用新版API格式
                    gemini_model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                except Exception as e:
                    # print(f"Warning: Failed to initialize with new API format: {str(e)}")
                    # print("Falling back to legacy API format")
                    
                    # 降级使用传统API格式 - 一些较旧版本的库使用不同的安全设置格式
                    safety_settings_legacy = [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
                    
                    gemini_model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config,
                        safety_settings=safety_settings_legacy
                    )
                
                # 处理图像输入
                images_to_send = []
                for img in [image_1, image_2, image_3]:
                    if img is not None:
                        images_to_send.extend(images_to_pillow(img))
                
                # 是否包含图像
                has_images = len(images_to_send) > 0
                
                # 根据 prompt_length 动态调整 input_prompt - 使用提示词模板
                if not override_system_prompt:
                    # 选择基础模板
                    if memory == "Enable":
                        # 处理历史记录：如果为空则使用空字符串，否则使用历史记录列表的字符串表示
                        history_str = "" if len(prompt_history) == 0 else str(list(prompt_history))
                        if has_images:
                            input_prompt = PROMPT_TEMPLATES["image_with_memory"].format(
                                theme=theme, 
                                history=history_str
                            )
                        else:
                            input_prompt = PROMPT_TEMPLATES["base_with_memory"].format(
                                theme=theme, 
                                history=history_str
                            )
                    else:  # memory == "Disable"
                        if has_images:
                            input_prompt = PROMPT_TEMPLATES["image_without_memory"].format(theme=theme)
                        else:
                            input_prompt = PROMPT_TEMPLATES["base_without_memory"].format(theme=theme)
                    
                    # 添加长度限制
                    if prompt_length > 0:
                        input_prompt += PROMPT_TEMPLATES["length_restriction"].format(length=prompt_length)
                    else:
                        input_prompt += PROMPT_TEMPLATES["no_length_restriction"]
                    
                    # 添加通用限制
                    input_prompt += PROMPT_TEMPLATES["general_restriction"]
                    
                    # 添加英文限制（如果启用）
                    if english_only:
                        input_prompt += PROMPT_TEMPLATES["english_only_restriction"]
                else:
                    # 处理自定义提示词中的变量替换
                    custom_prompt = override_system_prompt
                    
                    # 支持多种变量格式的替换
                    replacements = {
                        "{theme}": theme,
                        "{prompt_length}": str(prompt_length),
                        "{seed}": str(seed)
                    }
                    
                    # 根据memory状态决定历史记录的值
                    if memory == "Enable":
                        # 启用历史记录时，如果历史记录为空则替换为空字符串，否则正常替换
                        if len(prompt_history) == 0:
                            # 历史记录为空时，使用空字符串而不是"[]"
                            replacements.update({
                                "{prompt_history}": "",
                                "{list(prompt_history)}": "",
                                "{memory}": ""
                            })
                        else:
                            # 历史记录不为空时，正常替换历史记录变量
                            replacements.update({
                                "{prompt_history}": str(list(prompt_history)),
                                "{list(prompt_history)}": str(list(prompt_history)),
                                "{memory}": str(list(prompt_history))
                            })
                    else:
                        # 禁用历史记录时，历史记录变量替换为空字符串
                        replacements.update({
                            "{prompt_history}": "",
                            "{list(prompt_history)}": "",
                            "{memory}": ""
                        })
                    
                    # 替换所有支持的变量
                    for placeholder, value in replacements.items():
                        custom_prompt = custom_prompt.replace(placeholder, value)
                    
                    # 打印调试信息
                    print(f"Memory mode: {memory}")
                    if memory == "Disable":
                        print("历史记录变量被替换为空字符串")
                    elif memory == "Enable" and len(prompt_history) == 0:
                        print("Memory已启用但历史记录为空，历史记录变量被替换为空字符串")
                    
                    # 使用自定义提示词模板
                    if has_images:
                        input_prompt = PROMPT_TEMPLATES["custom_with_images"].format(
                            theme=theme, 
                            custom_prompt=custom_prompt
                        )
                    else:
                        input_prompt = PROMPT_TEMPLATES["custom_without_images"].format(
                            theme=theme, 
                            custom_prompt=custom_prompt
                        )
                    
                    # 添加长度限制
                    if prompt_length > 0:
                        input_prompt += PROMPT_TEMPLATES["length_restriction"].format(length=prompt_length)
                    else:
                        input_prompt += PROMPT_TEMPLATES["no_length_restriction"]
                    
                    # 添加通用限制
                    input_prompt += PROMPT_TEMPLATES["general_restriction"]
                    
                    # 添加英文限制（如果启用）
                    if english_only:
                        input_prompt += PROMPT_TEMPLATES["english_only_restriction"]

                # 检查是否取消
                if cancel_event.is_set():
                    return
                
                # 发送请求到API
                try:
                    if has_images:
                        # 发送文本和图像
                        content_list = [input_prompt] + images_to_send
                        response = gemini_model.generate_content(content_list)
                    else:
                        # 只发送文本
                        response = gemini_model.generate_content(input_prompt)
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "400 Bad Request" in error_msg:
                        raise ComfyUIAPIError(f"API rejected the request due to invalid input. Details: {error_msg}")
                    elif "403 Forbidden" in error_msg:
                        raise ComfyUIAPIError(f"API access denied. Please check your API key and permissions. Details: {error_msg}")
                    elif "429 Too Many Requests" in error_msg:
                        raise ComfyUIAPIError(f"API rate limit exceeded. Please wait before trying again. Details: {error_msg}")
                    elif "500 Internal Server Error" in error_msg:
                        raise ComfyUIAPIError(f"Google API server error. Please try again later. Details: {error_msg}")
                    elif "Connection" in error_msg:
                        raise ComfyUIAPIError(f"Connection error. Please check your internet connection. Details: {error_msg}")
                    else:
                        raise ComfyUIAPIError(f"Failed to generate content. Details: {error_msg}")
                
                # 检查是否取消
                if cancel_event.is_set():
                    return
                
                # 处理响应
                if response and hasattr(response, 'text'):
                    generated_prompt = response.text.strip()
                    
                    print("----INPUT----")
                    print(input_prompt)
                    if has_images:
                        print(f"(包含 {len(images_to_send)} 张图片)")
                    print("----OUTPUT----")
                    print(generated_prompt)
                    # 使用deque的append方法，当达到最大长度时会自动移除最早的元素
                    if memory == "Enable":  # 仅在启用memory时添加到历史记录
                        prompt_history.append(generated_prompt)
                    print("-------------")
                    
                    result[0] = generated_prompt
                else:
                    error[0] = ComfyUIAPIError("Received empty response from Gemini API.")
                
            except Exception as e:
                error[0] = ComfyUIAPIError(f"Failed to generate prompt. Details: {str(e)}")

        # 创建线程执行API调用
        worker_thread = threading.Thread(target=generate_content_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        # 等待线程完成或超时
        start_time = time.time()
        while worker_thread.is_alive():
            if time.time() - start_time > timeout:
                cancel_event.set()  # 通知线程取消操作
                error_message = f"Prompt generation timed out after {timeout} seconds. Please check your network connection or increase the timeout value."
                raise ComfyUITimeoutError(error_message)
            time.sleep(0.1)  # 短暂休眠以避免CPU占用
        
        # 检查是否有错误
        if error[0]:
            raise error[0]
        
        # 检查是否成功获得结果
        if result[0] is None:
            raise ComfyUIAPIError("Failed to generate prompt. No result was returned.")
        
        return (result[0],)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "GeminiPromptGeneratorJT": GeminiPromptGeneratorJT
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiPromptGeneratorJT": "Gemini Prompt Generator-JT"
}
