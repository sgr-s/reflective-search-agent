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
from .logger import get_logger

logger = get_logger(__name__)

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
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        decision = await model.with_structured_output(WebSearchDecision).ainvoke(
            [system_message] + state["messages"]
        )

        if decision.needs_web_search:
            return Command(goto="generate_search_queries")
        else:
            return Command(goto="generate_answer")
    except Exception as e:
        logger.error(f"should_web_searchでエラーが発生しました: {str(e)}", exc_info=True)
        raise


async def generate_search_queries(state: GraphState) -> Command:
    """検索クエリを生成するノード（最大2個）"""

    class SearchQueries(BaseModel):
        queries: List[str] = Field(description="検索クエリのリスト", max_length=2)
        reason: str = Field(description="クエリ選定理由")

    previous_queries = state.get("search_queries", [])
    feedback = state.get("feedback")

    # 過去のクエリがある場合の指示
    previous_instruction = ""
    if feedback:
        queries_text = "\n".join([f"- {q}" for q in previous_queries])
        previous_instruction = f"""
すでに利用した検索クエリ:
{queries_text}

重要: 前回と異なる角度から新しいクエリを生成してください。
{f'改善フィードバック: {feedback}'}
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
    try:
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
    except Exception as e:
        logger.error(f"generate_search_queriesでエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def execute_search(arg: dict) -> Command:
    """単一の検索クエリを実行するノード（並列実行用）"""
    try:
        query = arg.get("query", "")
        if not query:
            return {"search_results": []}

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
                logger.warning(f"URL {url} の読み込み中にエラーが発生しました: {str(e)}")
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
    except Exception as e:
        logger.error(f"execute_searchでエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def generate_answer_from_search(state: GraphState) -> Command:
    """検索結果を元に回答を生成するノード"""
    try:
        search_results = state.get("search_results", [])
        feedback = state.get("feedback")

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
        if feedback:
            previous_answer = state.get("response", "")
            improvement_instruction = f"""
## 改善フィードバック:
{feedback}

## 以前の回答:
{previous_answer}

**重要**: 上記のフィードバックを参考にして、より良い回答を作成してください。
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
    except Exception as e:
        logger.error(f"generate_answer_from_searchでエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def generate_answer(state: GraphState) -> Command:
    """Web検索なしで回答を生成するノード"""
    try:
        feedback = state.get("feedback")

        improvement_instruction = ""
        if feedback:
            previous_answer = state.get("response", "")
            improvement_instruction = f"""
## 改善フィードバック:
{feedback}

## 以前の回答:
{previous_answer}


**重要**: 上記のフィードバックを参考にして、より良い回答を作成してください。

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
    except Exception as e:
        logger.error(f"generate_answerでエラーが発生しました: {str(e)}", exc_info=True)
        raise


async def evaluate_answer(state: GraphState) -> Command:
    """回答を評価し、改善が必要か判断するノード"""
    try:
        from typing import Literal

        class AnswerEvaluation(BaseModel):
            is_satisfactory: bool = Field(description="回答がユーザーの質問に十分答えているかどうか")
            need: Optional[Literal["search", "generate"]] = Field(
                description="改善が必要な場合、どの部分の改善が必要か。search: 検索クエリや検索結果の改善、generate: 回答の改善。改善不要ならNone。"
            )
            reason: str = Field(description="上記の判断理由。is_satisfactoryの判断理由、または改善が必要な場合はその理由を記述。")
            feedback: Optional[str] = Field(
                description="改善が必要な場合（needがNoneでない場合）の具体的なフィードバック。searchならクエリに関するアドバイス、generateなら回答に関するアドバイス。"
            )

        attempt = state.get("attempt", 0)
        response = state.get("response", "")
        search_queries = state.get("search_queries", [])
        search_results = state.get("search_results", [])

        attempt += 1

        if attempt >= 3:
            return Command(goto="END")

        system_message = SystemMessage(
            content=f"""
あなたは回答品質を評価する専門家です。検索結果と生成された回答を比較し、評価してください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## 評価の流れ:
以下の順序で評価を行ってください:

### 1. 検索結果の確認 (need = "search" かどうか)
まず検索結果を詳細に確認し、ユーザーの質問に答えるための情報が含まれているかを判断してください。

**need = "search" (検索改善が必要):**
- **検索結果にユーザーの質問に答えるための情報が全く含まれていない**
- 検索クエリが明らかに不適切（質問と無関係なクエリ）
- 検索結果が質問と全く関連性がない
- 重要な情報が検索できていない（異なる角度からの検索で改善できそう）

### 2. 回答の確認 (need = "generate" かどうか)
検索結果が十分な場合、次に検索結果と回答を比較し、適切に活用されているかを判断してください。

**need = "generate" (回答改善が必要):**
- **検索結果にはユーザーの質問に答える情報があるのに、回答でその情報を活用できていない**
- **検索結果の重要な情報が回答に含まれていない**
- 回答の構成や表現が分かりにくい
- 検索結果を羅列しているだけで、自然な文章になっていない
- 質問に直接関係ない情報が大量に含まれている
- 会話履歴を無視し、前の会話から不自然に切り離された回答になっている

### 3. 全体的な満足度 (is_satisfactory)
検索と回答の両方が適切な場合、最終的に満足できるかを判断してください。

**need = None (改善不要):**
- **検索結果に含まれる重要な情報が回答に適切に反映されている**
- 回答が自然な文章で構成されている
- ユーザーが知りたい内容に焦点を絞り、簡潔に答えている
- 会話履歴を意識し、前の会話から自然に繋がる回答になっている

**is_satisfactory:**
- need が None の場合のみ True
- need が "search" または "generate" の場合は False

### 4. 判断理由 (reason)
- 上記の判断理由を具体的に記述してください

### 5. フィードバック (feedback)
- **need = "search" の場合**: 検索クエリに関するアドバイス（「どのようなキーワードで検索すべきか」「どの角度から検索すべきか」など）
- **need = "generate" の場合**: 回答に関するアドバイス（「どの情報を追加すべきか」「どう表現を改善すべきか」など）
- **need = None の場合**: None

## 重要な注意事項:
- **優先順位**: 検索結果に問題がある場合は need = "search"、検索結果は十分だが回答に問題がある場合は need = "generate"
- **is_satisfactory は need が None の場合のみ True にしてください**
- **reasonとfeedbackは具体的で実行可能な内容にしてください**
- **会話履歴全体を参照してユーザーの質問意図を理解してください**
"""
        )

        # HumanMessageで動的な値を渡す
        human_content_parts = []

        # 検索クエリを追加
        if search_queries:
            queries_text = "\n".join([f"- {q}" for q in search_queries])
            human_content_parts.append(f"## 実行した検索クエリ:\n{queries_text}")

        # 検索結果を追加
        if search_results:
            human_content_parts.append("\n## 取得した検索結果:")
            for i, result in enumerate(search_results, 1):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", result.get("snippet", ""))
                query = result.get("query", "")

                human_content_parts.append(f"\n### 検索結果 {i}")
                human_content_parts.append(f"**検索クエリ**: {query}")
                human_content_parts.append(f"**タイトル**: {title}")
                human_content_parts.append(f"**URL**: {url}")
                human_content_parts.append(f"**内容**:\n{content}\n")

        # 生成された回答を追加
        human_content_parts.append(f"\n## 生成された回答:\n{response}")

        from langchain_core.messages import HumanMessage
        human_message = HumanMessage(content="\n".join(human_content_parts))

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        evaluation = await model.with_structured_output(AnswerEvaluation).ainvoke(
            [system_message] + state["messages"] + [human_message]
        )

        if evaluation.is_satisfactory:
            return Command(goto="END")

        # 検索改善が必要な場合
        if evaluation.need == "search":
            return Command(
                update={
                    "attempt": attempt,
                    "search_results": [],  # 検索結果をクリア
                    "feedback": evaluation.feedback
                },
                goto="generate_search_queries"
            )

        # 回答のみ改善が必要な場合
        if evaluation.need == "generate":
            next_node = "generate_answer_from_search" if search_queries else "generate_answer"
            return Command(
                update={
                    "attempt": attempt,
                    "feedback": evaluation.feedback
                },
                goto=next_node
            )

        return Command(goto="END")
    except Exception as e:
        logger.error(f"evaluate_answerでエラーが発生しました: {str(e)}", exc_info=True)
        raise