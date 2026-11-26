"""llm.py —— 调用大模型的工具（从 D4 的 chat() 抽出来，单独一个文件）

这样拆的好处：调模型和 RAG 流程分开，以后换模型只改这个文件。
"""
import os                                    # 用来读环境变量里的 API Key
import requests                              # 用来发 HTTP 请求
from dotenv import load_dotenv              # 从 .env 文件读配置

load_dotenv()                                # 把 .env 里的变量读进环境变量

API_URL = "https://api.deepseek.com/chat/completions"   # DeepSeek 的对话接口地址
MODEL = "deepseek-chat"                                  # 要用哪个模型


def chat(messages):                          # 定义函数：给一组消息，返回模型回复的文字
    """messages 是一个列表，比如 [{"role": "user", "content": "你好"}]。"""
    headers = {                              # 请求头：通行证 + 内容类型
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",   # 带上 API Key
        "Content-Type": "application/json",  # 告诉服务器：我发的是 JSON
    }
    payload = {                              # 请求体：模型名 + 消息列表
        "model": MODEL,                      # 用哪个模型
        "messages": messages,                # 要说的话
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)   # 发请求（最多等 60 秒）
    if resp.status_code != 200:              # 如果服务器返回错误
        print("API 出错：", resp.status_code, resp.text[:200])   # 打印状态码和错误内容
        return None                          # 返回 None，表示没拿到回复
    return resp.json()["choices"][0]["message"]["content"]   # 从返回的 JSON 里剥出回复文字


if __name__ == "__main__":                   # 只有直接运行这个文件时才执行（自测）
    print(chat([{"role": "user", "content": "只回答两个字：收到"}]))   # 简单测一下能不能通
