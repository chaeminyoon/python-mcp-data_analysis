from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama

from dotenv import load_dotenv
import os

#load_dotenv()
#os.getenv("OPENAI_API_KEY")
OLLAMA_SERVER = "http://192.168.2.209:11434"
MODEL_NAME = "qwen2.5:72b"


#model = ChatOpenAI(model=MODEL_NAME)
model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_SERVER,temperature=0,num_ctx=8192)
SERVER_VERSION = "v.3.0" 

server_params = StdioServerParameters(
    command="python",
    args=[f"./data_server_{SERVER_VERSION}.py"],
)



async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            ##### AGENT #####
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            # ✅ 대화 히스토리 관리
            conversation_history = []
            system_prompt = await load_mcp_prompt(
                session, "default_prompt", arguments={"message": ""}
            )
            # 시스템 메시지만 추출 (첫 번째 메시지)
            if system_prompt:
                conversation_history.append(system_prompt[0])

            print("\n" + "="*60)
            print(f" MCP 데이터 분석 시스템 (v.3.0) - Model: {MODEL_NAME}")
            print("="*60)
            print("Tip: 이전 대화를 기억합니다. 자연스럽게 대화하세요!")
            print(" 예: '이제 이상치를 제거해줘', '그 결과를 시각화해줘'")
            print(" Commands: 'clear' - 대화 초기화, 'exit/종료' - 종료")
            print("="*60 + "\n")

            while True:
                ##### REQUEST & RESPOND #####
                try:
                    user_input = input("You: ")
                    
                    if user_input.lower() in ["exit", "quit", "q", "종료"]:
                        print("\n👋 종료합니다. 감사합니다!")
                        break
                    
                    # 대화 초기화 명령
                    if user_input.lower() == "clear":
                        conversation_history = [conversation_history[0]]  # 시스템 메시지만 유지
                        print("\n🔄 대화 기록이 초기화되었습니다.\n")
                        continue
                    
                    if not user_input.strip():
                        continue

                    # 사용자 메시지를 히스토리에 추가
                    from langchain_core.messages import HumanMessage
                    conversation_history.append(HumanMessage(content=user_input))

                    print("\n🤔 분석 중...\n")
                    
                    # 전체 대화 히스토리를 agent에 전달
                    response = await agent.ainvoke({"messages": conversation_history})
                    
                    # AI 응답을 히스토리에 추가
                    conversation_history = response["messages"]
                    
                    # 최신 응답 출력
                    ai_response = response["messages"][-1].content
                    print("="*60)
                    print("AI:", ai_response)
                    print("="*60 + "\n")

                except EOFError:
                    print("\n👋 종료합니다.")
                    break
                except KeyboardInterrupt:
                    print("\n\n👋 종료합니다.")
                    break
                except Exception as e:
                    print(f"\n❌ 오류 발생: {str(e)}\n")
                    # 오류 발생 시 마지막 사용자 메시지 제거
                    if len(conversation_history) > 1:
                        conversation_history.pop()


import asyncio

asyncio.run(run())
