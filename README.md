# AI大模型之美

在「AI大模型之美」时，发现很多代码不能运行。因为 OpenAI 库升级很快，版本兼容性也很差，加上有的模型也废弃了。导致运行代码困难重重。
我相信很多初学者也跟我一样遇到这些问题。所以，我对部分代码进行修改，让它们能在最新版本中运行起来。部分使用 CPU 运行的代码也改成使用 GPU 运行。

# 环境
Win11 WSL2、Docker jupyterLib（基于nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04镜像构建的）、NVIDIA GeForce RTX 3080 Laptop GPU

# 配置
建议将下面配置设置在环境变量中，或放在 {jupyter_path}/.jupyter/jupyter_notebook_config.py 中
```python
import os

os.environ['OPENAI_API_KEY'] = 'sk-xxx' # OpenAI 或第三方代理 KEY
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1' # OpenAI 或 第三方代理
os.environ['AZURE_SPEECH_KEY'] = '' 
os.environ['AZURE_SPEECH_REGION'] = ''
os.environ['DID_API_KEY'] = ''
os.environ['SERPER_API_KEY'] = ''
os.environ['HUGGINGFACE_API_KEY'] = ''
```

# 主要修改
- `COMPLETION_MODEL` 换成 `gpt-3.5-turbo`， `text-davinci-003` 已废弃
- 修改openai的调用及返回数据处理
- 将模型从`text-davinci-003`换成`gpt-3.5-turbo-instruct`
- 修改`get_fasttext_vector` (vec.reshape(1, -1))
- `T5Tokenizer`修改原CPU相关代码，并增加GPU示例代码
- `gradio`调用修改，并指定`server_name`和`server_port`
- `faiss`换成GPU版
- 修改`llama_index`引用及存储调用等
- `spacy`换成GPU版
- 修改`langchain`、`langchain_openai`等引用
- 修改`azure.cognitiveservices.speech`调用，并修改支持生成语音播放
- 将`paddlepaddle`改为GPU版，并处理报错（增加 use_onnx=True）
- 修改`gradio`支持语音播放（支持服务器没有声卡）

# 目录
#### 01｜重新出发，让我们学会和AI说话
#### 02｜无需任何机器学习，如何利用大语言模型做情感分析？
#### 03｜巧用提示语，说说话就能做个聊天机器人
#### 04｜新时代模型性能大比拼，GPT-3到底胜在哪里？
#### 05｜善用Embedding，我们来给文本分分类
#### 06｜ChatGPT来了，让我们快速做个AI应用
#### 07｜文本聚类与摘要，让AI帮你做个总结
#### 08｜文本改写和内容审核，别让你的机器人说错话
#### 09｜语义检索，利用Embedding优化你的搜索功能
#### 10｜AI连接外部资料库，让Llama Index带你阅读一本书
#### 11｜省下钱买显卡，如何利用开源模型节约成本？
#### 12｜让AI帮你写个小插件，轻松处理Excel文件
#### 13 ｜让AI帮你写测试，体验多步提示语
#### 14｜链式调用，用LangChain简化多步提示语
#### 15｜深入使用LLMChain，给AI连上Google和计算器
#### 16｜Langchain里的“记忆力”，让AI只记住有用的事儿
#### 17｜让AI做决策，LangChain里的“中介”和“特工”
#### 18｜流式生成与模型微调，打造极致的对话体验
#### 19｜Whisper+ChatGPT：请AI代你听播客
#### 20｜TTS与语音合成：让你的机器人拥有声音
#### 21｜DID和PaddleGAN：表情生动的数字人播报员
#### 22｜再探HuggingFace：一键部署自己的大模型
#### 23｜OpenClip：让我们搞清楚图片说了些什么
#### 24｜Stable Diffusion：最热门的开源AI画图工具
#### 25｜ControlNet：让你的图拥有一个“骨架”
#### 26｜Visual ChatGPT是如何做到边聊边画的？
#### 27｜从Midjourney开始，探索AI产品的用户体验