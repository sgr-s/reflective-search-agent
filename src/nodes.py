import asyncio
import re
from datetime import datetime
from typing import List, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from config.config import GOOGLE_API_KEY, GOOGLE_CX

from .state import GraphState

async def should_web_search(state: GraphState) -> Command:
    """Web検索が必要かを判断するノード"""

    class WebSearchDecision(BaseModel):
        needs_web_search: bool = Field(description="Web検索が必要かどうか")
        reason: str = Field(description="判断理由")

    system_message = SystemMessage(
        content=f"""
会話履歴全体を参照して、ユーザーのメッセージに対してWeb検索が必要かどうかを正確に判断してください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## 判断基準:
**既存の事前知識で回答できるかどうか**を基準に判断してください:
- 既存の事前知識で回答できる → Web検索不要
- 最新情報・リアルタイム情報が必要 → Web検索必要

**重要**: 会話履歴から文脈を理解した上で判断してください。
迷った場合や、少しでも最新情報が必要な可能性がある場合は、Web検索**必要**と判断してください。
"""
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    decision = await model.with_structured_output(WebSearchDecision).ainvoke(
        [system_message] + state["messages"]
    )

    if decision.needs_web_search:
        return Command(goto="generate_search_queries")
    else:
        return Command(goto="generate_answer")


async def generate_search_queries(state: GraphState) -> Command:
    """検索クエリを生成するノード（最大2個）"""

    class SearchQueries(BaseModel):
        queries: List[str] = Field(description="検索クエリのリスト", max_length=2)
        reason: str = Field(description="クエリ選定理由")

    previous_queries = state.get("search_queries", [])
    search_improvement_advice = state.get("search_improvement_advice")

    # 過去のクエリがある場合の指示
    previous_instruction = ""
    if previous_queries:
        queries_text = "\n".join([f"- {q}" for q in previous_queries])
        previous_instruction = f"""
すでに利用した検索クエリ:
{queries_text}

重要: 前回と異なる角度から新しいクエリを生成してください。
{f'改善アドバイス: {search_improvement_advice}' if search_improvement_advice else ''}
"""

    system_message = SystemMessage(
        content=f"""
ユーザーの質問に答えるために最適な検索クエリを生成してください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

{previous_instruction}

## クエリ生成のルール:

1. **複数の視点から検索**:
    - 異なる角度から情報を集めるため、1-2個のクエリを生成
    - 重複する内容のクエリは避ける

2. **具体的で明確なクエリ**:
    - 曖昧な表現を避け、固有名詞を使う

3. **時間的文脈の考慮**:
    - ユーザーが「本日」「今日」と言った場合 → 必ず日付を含める
    - 過去の情報が欲しい場合 → 具体的な期間を指定
    - 最新情報が欲しい場合 → "最新"や年月を含める

4. **会話履歴の活用**:
    - 代名詞（「それ」「この」など）は会話履歴から具体的な名詞に変換
    - 文脈から暗黙の情報を補完

5. **検索エンジン最適化**:
    - 自然な日本語で、検索エンジンが理解しやすい形式
    - キーワードの組み合わせを工夫

## 重要な注意事項:
- 必ず1-2個のクエリを生成してください（1個でも2個でも可）
- 会話履歴全体を参照して文脈を理解してください
""")
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    search_queries_result = await model.with_structured_output(SearchQueries).ainvoke(
            [system_message] + state["messages"]
        )

    sends = [
                Send("execute_search", {"query": query})
                for query in search_queries_result.queries
            ]

    return Command(
                update={
                    "search_queries": search_queries_result.queries,
                },
                goto=sends
            )

async def execute_search(arg: dict) -> Command:
    """単一の検索クエリを実行するノード（並列実行用）"""

    query = arg.get("query", "")
    if not query:
        return {"search_results": []}

    # Google Custom Search APIの設定
    search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CX
    )

    results = search.results(query, num_results=2)

    if not results:
        return {"search_results": []}

    search_results = []
    for result in results:
        url = result['link']
        title = result['title']
        snippet = result.get('snippet', '')

        try:
            # Webページの内容を取得
            loader = WebBaseLoader(url)
            load_task = asyncio.to_thread(loader.load)
            docs = await asyncio.wait_for(load_task, timeout=15.0)

            # テキストのクリーニング
            content = re.sub(r'\n\s*\n+', '\n\n', docs[0].page_content)
            content = '\n'.join([line.strip() for line in content.split('\n') if line.strip()])
            search_results.append({
                "query": query,
                "title": title,
                "url": url,
                "content": content[:5000],
                "snippet": snippet
            })
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
            # エラー時はsnippetのみ使用
            search_results.append({
                "query": query,
                "title": title,
                "url": url,
                "snippet": snippet
            })

    return Command(
        update={"search_results": search_results},
        goto="generate_answer_from_search"
    )

async def generate_answer_from_search(state: GraphState) -> Command:
    """検索結果を元に回答を生成するノード"""

    search_results = state.get("search_results", [])
    answer_improvement_advice = state.get("answer_improvement_advice")

    # 検索結果のフォーマット
    results_text = ""
    for i, result in enumerate(search_results, 1):
        results_text += f"""
検索結果 {i}:
- クエリ: {result.get("query")}
- タイトル: {result.get("title")}
- URL: {result.get("url")}
- 内容: {result.get("content", result.get("snippet"))}
"""

    improvement_instruction = ""
    if answer_improvement_advice:
        previous_answer = state.get("response", "")
        improvement_instruction = f"""
## 改善アドバイス:
{answer_improvement_advice}

## 以前の回答:
{previous_answer}

**重要**: 上記のアドバイスを参考にして、より良い回答を作成してください。
"""

    system_message = SystemMessage(
        content=f"""
以下の検索結果を元に、ユーザーの質問に答えてください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## 取得した検索結果:
{results_text}

## 回答作成のルール:

1. **検索結果のみを使用**:
    - 検索結果に含まれる情報のみを使って回答する
    - 検索結果にない情報は推測しない

2. **自然で簡潔な文章**:
    - **検索結果を羅列するのではなく、自然な文章で回答する**
    - ユーザーが知りたい内容に焦点を絞り、簡潔に答える
    - 不要な情報は省略し、質問の核心に直接答える
    - 数字、日付、固有名詞など具体的な情報を含める

3. **回答の構成**:
    - まず質問に対する直接的な答えを述べる
    - 必要に応じて補足情報を追加（過度な詳細は避ける）
    - 箇条書きは最小限にし、自然な文章を優先する

4. **会話履歴を意識した自然な繋がり**:
    - **会話履歴全体を参照し、文脈に沿った自然な回答を生成する**
    - 前の会話の流れを踏まえた言い回しを使う
    - ユーザーとの会話が自然に繋がるように配慮する
    - 代名詞（「それ」「この」など）を適切に使い、会話の連続性を保つ

5. **不足情報への対応**:
    - 検索結果が不完全な場合は、その旨を正直に伝える
    - 得られた情報の範囲で最大限回答する

{improvement_instruction}
"""
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    answer = await model.ainvoke([system_message] + state["messages"])

    return Command(
        update={"response": answer.content},
        goto="evaluate_answer"
    )

async def generate_answer(state: GraphState) -> Command:
    """Web検索なしで回答を生成するノード"""

    answer_improvement_advice = state.get("answer_improvement_advice")

    improvement_instruction = ""
    if answer_improvement_advice:
        previous_answer = state.get("response", "")
        improvement_instruction = f"""
## 改善アドバイス:
{answer_improvement_advice}

## 以前の回答:
{previous_answer}


**重要**: 上記のアドバイスを参考にして、より良い回答を作成してください。

"""

    system_message = SystemMessage(
            content=f"""
ユーザーの質問や依頼に対して、適切に応答してください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## 回答作成のルール:
- **会話の流れを意識し、文脈に沿った自然な回答を生成する**
- 前の会話の内容を踏まえた言い回しを使う
- ユーザーとの会話が自然に繋がるように配慮する
- 代名詞（「それ」「この」など）を適切に使い、会話の連続性を保つ

{improvement_instruction}
"""
        )

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    answer = await model.ainvoke([system_message] + state["messages"])

    return Command(
        update={"response": answer.content},
        goto="evaluate_answer"
    )


async def evaluate_answer(state: GraphState) -> Command:
    """回答を評価し、改善が必要か判断するノード"""

    class AnswerEvaluation(BaseModel):
        is_satisfactory: bool = Field(description="回答が十分か")
        needs_better_search: bool = Field(description="検索改善が必要か")
        needs_better_answer: bool = Field(description="回答改善が必要か")
        search_improvement_advice: Optional[str] = Field(description="検索改善アドバイス")
        answer_improvement_advice: Optional[str] = Field(description="回答改善アドバイス")

    attempt = state.get("attempt", 0)
    response = state.get("response", "")
    search_queries = state.get("search_queries", [])
    search_results = state.get("search_results", [])
    current_date = datetime.now().strftime("%Y年%m月%d日")

    attempt += 1

    if attempt >= 3:
        return Command(goto="END")

    queries_text = "\n".join([f"- {q}" for q in search_queries])

    # 検索結果のテキストを生成
    results_text = ""
    if search_results:
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", result.get("snippet", ""))
            query = result.get("query", "")

            results_text += f"## 検索結果 {i}\n"
            results_text += f"**検索クエリ**: {query}\n"
            results_text += f"**タイトル**: {title}\n"
            results_text += f"**URL**: {url}\n"
            results_text += f"**内容**:\n{content}\n\n"

    system_message = SystemMessage(
        content=f"""
検索結果と生成された回答を比較し、2つの観点から評価してください。

## 現在の日付:
{current_date}

## 実行した検索クエリ:
{queries_text}

## 取得した検索結果:
{results_text}

## 生成された回答:
{response}

## 評価基準:

### 1. 全体的な満足度 (is_satisfactory)
- ユーザーの質問に対して、**具体的な回答が含まれている**
- 情報が不完全でも、得られた範囲での回答がされている
- 少しでも回答になっていれば True にする

### 2. 検索改善の必要性 (needs_better_search)
**重要**: まず検索結果を詳細に確認し、ユーザーの質問に答えるための情報が含まれているかを判断してください。

**True にすべきケース**:
- **検索結果を確認した結果、ユーザーの質問に答えるための情報が全く含まれていない**
- 検索クエリが明らかに不適切（質問と無関係なクエリ）

**False にすべきケース**:
- **検索結果にユーザーの質問に答えるための情報が含まれている**（回答で活用されていなくても、情報があればFalse）
- 検索クエリは適切で、有用な情報が得られている
- 再検索しても同じ結果になる可能性が高い

### 3. 回答改善の必要性 (needs_better_answer)
**重要**: 検索結果と回答を比較し、検索結果に含まれる情報が回答で適切に活用されているかを判断してください。

**True にすべきケース**:
- **検索結果にはユーザーの質問に答える情報があるのに、回答でその情報を活用できていない**
- **検索結果の重要な情報（優勝チーム名、スコア、日付など）が回答に含まれていない**
- 回答の構成や表現が分かりにくい
- **ユーザーの質問に直接関係ない情報が大量に含まれている**
- **会話履歴を無視し、前の会話から不自然に切り離された回答になっている**

**False にすべきケース**:
- **検索結果に含まれる重要な情報が回答に適切に反映されている**
- 回答が自然な文章で構成されている
- ユーザーが知りたい内容に焦点を絞り、簡潔に答えている
- **会話履歴を意識し、前の会話から自然に繋がる回答になっている**

## 改善アドバイスの生成:

### search_improvement_advice:
- **needs_better_search が True の場合のみ**具体的なアドバイスを記述
- 「どのようなキーワードで検索すべきか」「どの角度から検索すべきか」など具体的に
- needs_better_search が False なら None

### answer_improvement_advice:
- **needs_better_answer が True の場合のみ**具体的なアドバイスを記述
- 「どの情報を追加すべきか」「どう表現を改善すべきか」など具体的に
- needs_better_answer が False なら None

## 重要な注意事項:
- **まず検索結果を詳細に確認し、ユーザーの質問に答えるための情報が含まれているかを判断してください**
- **検索結果に情報が含まれている場合、needs_better_search は False にしてください**
- **検索結果に情報があるのに回答で活用されていない場合は、needs_better_answer を True にしてください**
- 検索と回答は独立して評価してください
- 両方とも改善が必要な場合は、両方を True にしてください
- 会話履歴全体を参照してユーザーの質問意図を理解してください
- **is_satisfactory は needs_better_search と needs_better_answer の両方が False の場合のみ True にしてください**
- **改善アドバイスは具体的で実行可能な内容にしてください**
"""
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    evaluation = await model.with_structured_output(AnswerEvaluation).ainvoke(
        [system_message] + state["messages"]
    )

    if evaluation.is_satisfactory:
        return Command(goto="END")

    # 検索改善が必要な場合
    if evaluation.needs_better_search:
        return Command(
            update={
                "attempt": attempt,
                "search_results": [],  # 検索結果をクリア
                "search_improvement_advice": evaluation.search_improvement_advice,
                "answer_improvement_advice": evaluation.answer_improvement_advice
            },
            goto="generate_search_queries"
        )

    # 回答のみ改善が必要な場合
    if evaluation.needs_better_answer:
        next_node = "generate_answer_from_search" if search_queries else "generate_answer"
        return Command(
            update={
                "attempt": attempt,
                "answer_improvement_advice": evaluation.answer_improvement_advice
            },
            goto=next_node
        )

    return Command(goto="END")