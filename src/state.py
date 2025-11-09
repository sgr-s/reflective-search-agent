import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

class SearchResult(TypedDict):
    query: str
    title: str
    url: str
    snippet: str
    content: Optional[str]

def merge_search_results(left: list[SearchResult] | None, right: list[SearchResult] | None) -> list[SearchResult]:
    """検索結果を蓄積するカスタムリデューサー"""
    if right is None:
        return left or []
    if not right:  # 空リストでクリア
        return []
    if not left:
        return right
    return left + right  # 両方ある場合は連結

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    response: Optional[str]
    search_queries: Annotated[list[str], operator.add]
    search_results: Annotated[list[SearchResult], merge_search_results]
    attempt: Optional[int]
    search_improvement_advice: Optional[str]
    answer_improvement_advice: Optional[str]