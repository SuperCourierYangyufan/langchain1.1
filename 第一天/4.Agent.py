from email import message
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

chat = ChatOpenAI(
    model_name=os.getenv("OPEN_AI_CHAT_MODEL"),
    openai_api_key=os.getenv("OPEN_AI_API_KEY"),
    openai_api_base=os.getenv("OPEN_AI_BASE_URL"),
    verbose=True
)

def get_info(city: str) -> str:
    """输入城市,输出天气"""
    return f"今天{city}零下20度"

graph = create_agent(
    model= chat,
    tools=[get_info]
)

# result = graph.invoke({
#     "messages": [
#         SystemMessage("你是一个人工助手"),
#         HumanMessage("你好,今天武汉天气如何")
#     ]
# })

# print(result)
# for message in result["messages"]:
#     print(message.pretty_print())
    
    
# 每走一个节点 返回一个event
for event in graph.stream(
    {
        "messages": [
            SystemMessage("你是一个人工助手"),
            HumanMessage("你好,今天武汉天气如何")
        ]
    },
#  values	在每一步之后流式传输图的完整状态
# updates	在每一步之后仅流式传输状态更新
# messages	一个个token输出
# custom	从图节点内部流式传输自定义数据
# debug	在图执行期间流式传输尽可能多的信息
    # stream_mode="values"
    stream_mode="messages"
):
    # print(event["messages"][-1].pretty_print())
    print(event[0].content)