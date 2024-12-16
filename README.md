# Gemini prompt generator JT version
Custom node to use Gemini 1.5 and above for Comfyui to generates theme related prompts for image generators
Fork from Magifactory, added many feature on top of it.

<img src="https://github.com/user-attachments/assets/fe987a9f-06c8-4a35-8de2-4b301007b266" width="400">

## New update
Now support Gemini 2.0 Flash!

<img src="https://github.com/user-attachments/assets/bfe6831b-3189-43e8-bc5e-1fde60f24d4f" width="400">

Added System prompt override support, it can turn into a LLM
Please, Leave the Override system prompt area empty, if you wish to use this node as a normal Prompt Generator.
Unless you have other needs.

<img src="https://github.com/user-attachments/assets/ad215761-d8ca-4d1a-bfb5-c774a0b70b66" width="400">

# Usage
cd ur custom_nodes folder location
ie. E:\ComfyUI_windows_portable\ComfyUI\custom_nodes
then type following:

```bash
git clone https://github.com/LiJT/ComfyUI-Gemini-Prompt-Generator-JT
```

# Note
The Seed actually is NOT for Gemini, just tell ComfyUI when to generate the new prompt, if the seed is fixed, then Node wont generate a new one
Input your API Key here: config.json 

<img src="https://github.com/user-attachments/assets/96a03508-8965-4960-8a9e-10e96e94b277" width="400">
