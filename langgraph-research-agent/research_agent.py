from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. STATE
# ============================================================

class ResearchState(MessagesState):
    pass


# ============================================================
# 3. LOCAL LLM - OLLAMA
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ============================================================
# 4. INTERNET SEARCH TOOL
# ============================================================

web_search = TavilySearch(
    max_results=3,
    topic="general"
)


tools = [
    web_search
]


# ============================================================
# 5. BIND TOOLS TO LLM
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 6. RESEARCHER NODE
# ============================================================

def researcher(state: ResearchState):

    system_message = {
        "role": "system",
        "content": """
You are a helpful research assistant.

Your job is to answer the user's questions accurately.

Use your own knowledge when it is sufficient.

Use the web search tool when:

1. The user asks for latest or current information.
2. The information may have changed recently.
3. You are uncertain about the answer.
4. The user explicitly asks you to search the internet.
5. The question requires external research.

After receiving search results, analyze them and
provide a clear final answer.

Do not claim that you searched the internet unless
you actually used the search tool.
"""
    }

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# 7. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 8. CONDITIONAL ROUTING
# ============================================================

def should_continue(state: ResearchState):

    last_message = state["messages"][-1]

    # LLM requested a tool
    if last_message.tool_calls:
        return "tools"

    # LLM has finished
    return END


# ============================================================
# 9. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(ResearchState)


# Add researcher node
builder.add_node(
    "researcher",
    researcher
)


# Add tool node
builder.add_node(
    "tools",
    tool_node
)


# START → researcher
builder.set_entry_point(
    "researcher"
)


# researcher → tools OR END
builder.add_conditional_edges(
    "researcher",
    should_continue
)


# tools → researcher
builder.add_edge(
    "tools",
    "researcher"
)


# ============================================================
# 10. COMPILE GRAPH
# ============================================================

# IMPORTANT:
# Do NOT use MemorySaver here.
# LangGraph Studio / LangGraph API handles persistence.
graph = builder.compile()


# ============================================================
# 11. OPTIONAL COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STATEFUL LANGGRAPH RESEARCH AGENT")
    print("=" * 70)

    print("LLM       : Ollama")
    print("Model     : Qwen 2.5 3B")
    print("Framework : LangChain + LangGraph")
    print("Tool      : Tavily Internet Search")
    print("Studio    : LangGraph Studio")

    print("=" * 70)
    print("Type 'exit' to quit.")
    print("=" * 70)

    # Thread ID is supplied by the LangGraph API/Studio.
    config = {
        "configurable": {
            "thread_id": "class-demo-1"
        }
    }

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        try:

            result = graph.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                },
                config
            )

            print("\nAgent:")
            print(
                result["messages"][-1].content
            )

        except Exception as e:

            print("\nError:")
            print(e)

