from langgraph.graph import START, StateGraph

from .nodes import (
    evaluate_answer,
    execute_search,
    generate_answer,
    generate_answer_from_search,
    generate_search_queries,
    should_web_search,
)
from .state import GraphState

# グラフの初期化
graph = StateGraph(GraphState)

# ノードの追加
graph.add_node(should_web_search)
graph.add_node(generate_search_queries)
graph.add_node(execute_search)
graph.add_node(generate_answer_from_search)
graph.add_node(generate_answer)
graph.add_node(evaluate_answer)

graph.add_edge(START, "should_web_search")