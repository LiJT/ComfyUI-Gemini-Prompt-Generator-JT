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
                "model": (["gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"], {
                    "tooltip": "选择要使用的Gemini模型。**强烈推荐使用最新模型**：\n\n🔥 **推荐模型（免费可用）**：\n• **gemini-2.5-flash** - 最新混合推理模型，性价比最高，支持思维链推理 [免费：10 RPM, 250K TPM, 250 RPD]\n• **gemini-2.5-flash-lite** - 最经济实惠的模型，适合大规模使用 [免费：15 RPM, 250K TPM, 1000 RPD]\n• **gemini-2.0-flash** - 均衡的多模态模型，适用于各种任务 [免费：15 RPM, 1M TPM, 200 RPD]\n• **gemini-2.0-flash-lite** - 轻量级高效模型，成本最低 [免费：30 RPM, 1M TPM, 200 RPD]\n\n⚠️ **已弃用模型（不推荐）**：\n• **gemini-1.5-flash** - 已弃用，将于2025年9月24日退役，建议迁移到2.0-flash-lite\n• **gemini-1.5-flash-8b** - 已弃用，将于2025年9月24日退役\n• **gemini-1.5-pro** - 已弃用，将于2025年9月24日退役，建议迁移到2.5-flash\n\n💡 **免费层说明**：RPM=每分钟请求数，TPM=每分钟Token数，RPD=每日请求数\n📖 **迁移优势**：新模型提供更高免费配额、更好性能、更低成本和新功能支持"
                }),
                "enable_memory": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "启用或禁用历史记忆功能。启用时会记住之前生成的提示，避免重复；禁用时每次生成独立提示"
                }),
                "prompt_length": ("INT", {
                    "default": 200, 
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
                    "default": 30, 
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
    
    def generate_prompt(self, theme, override_system_prompt, model, enable_memory, prompt_length, seed, timeout, image_1=None, image_2=None, image_3=None):
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
                
                # 根据 prompt_length 动态调整 input_prompt
                if not override_system_prompt:
                    if memory == "Enable":
                        if has_images:
                            input_prompt = f"Generate me a prompt for image generator based on the provided image(s). The theme of the prompt is {theme}. You already created those prompts: {list(prompt_history)}. Make sure you generate original prompt that describes and expands on what you see in the image(s). Think about it step by step and make some internal critique."
                        else:
                            input_prompt = f"Generate me a prompt for image generator. The theme of the prompt is {theme}. You already created those prompts: {list(prompt_history)}. Make sure you generate original prompt. Think about it step by step and make some internal critique."
                    else:  # memory == "Disable"
                        if has_images:
                            input_prompt = f"Generate me a prompt for image generator based on the provided image(s). The theme of the prompt is {theme}. Think about it step by step and make some internal critique."
                        else:
                            input_prompt = f"Generate me a prompt for image generator. The theme of the prompt is {theme}. Think about it step by step and make some internal critique."
                    
                    # 只有当 prompt_length 不为 0 时，才添加长度和 prompt 标签的限制
                    if prompt_length > 0:
                        input_prompt += f" You must keep the length of your generated prompt around {prompt_length} words. **You only need to output generated prompt and nothing else!**"
                    else:
                        input_prompt += f" **You only need to output generated prompt and nothing else!**"
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
                        # 启用历史记录时，正常替换历史记录变量
                        replacements.update({
                            "{prompt_history}": str(list(prompt_history)),
                            "{list(prompt_history)}": str(list(prompt_history)),
                            "{memory}": str(list(prompt_history))
                        })
                    else:
                        # 禁用历史记录时，历史记录变量替换为空列表
                        replacements.update({
                            "{prompt_history}": "[]",
                            "{list(prompt_history)}": "[]",
                            "{memory}": "[]"
                        })
                    
                    # 替换所有支持的变量
                    for placeholder, value in replacements.items():
                        custom_prompt = custom_prompt.replace(placeholder, value)
                    
                    # 打印调试信息
                    print(f"Memory mode: {memory}")
                    if memory == "Disable":
                        print("历史记录变量被替换为空列表[]")
                    
                    if has_images:
                        input_prompt = f"The theme is {theme}. I'm providing you with image(s) for reference. Please generate a response strictly following the instruction: {custom_prompt}."
                    else:
                        input_prompt = f"The theme is {theme}. Please generate a response strictly following the instruction: {custom_prompt}."
                    
                    # 只有当 prompt_length 不为 0 时，才添加长度和 prompt 标签的限制
                    if prompt_length > 0:
                        input_prompt += f" You must keep the length of response around {prompt_length} words. **You only need to give me the answer and nothing else!**"
                    else:
                        input_prompt += f" **You only need to give me the answer and nothing else!**"

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
