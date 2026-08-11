# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


# 定义工具
import os
import requests
from langchain_core.tools import tool
@tool
def get_weather(city_location:str):
    """根据用户输入的城市名获取当前天气状况"""
    # 配置API密钥和请求参数
    params = {
        "key": os.getenv("WEATHER_API_KEY"),
        "location": city_location,
        "language": "zh-Hans",# 语言，zh-Hans为简体中文
        "unit": "c" # 单位，c为摄氏温度
    }
    url = "https://api.seniverse.com/v3/weather/now.json" # 实况天气接口
    try:
        response = requests.get(url, params=params)
        # 解析并打印返回的JSON数据
        data = response.json()
        results = data['results'][0]
        location = results['location']['name']
        weather = results['now']['text']
        temp = results['now']['temperature']
        return f"{location}当前的天气是{weather}，温度为{temp}摄氏度"

    except Exception as e:
        return f"发生错误，错误信息为：{e}"




# 定义智能体
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather],
    system_prompt = "你是一个查询天气助手，通过调用工具帮助用户获取当前天气"
)

# 调用
location = input("请输入您要查询天气的城市：")

from fastapi import FastAPI

app=FastAPI(title="天气查询")

@app.get("/weather/{location}")
def weather(location:str):
    """获取当前天气"""
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=f"帮我查一下当前{location}的天气")]}
        )
        return response['messages'][-1].content
    except Exception as e:
        return f"出现异常，异常信息为：{e}"
