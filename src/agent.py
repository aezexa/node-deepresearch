import asyncio
import json
import math
import random
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field

from action_types import (  # Assuming action_types.py exists
    AnswerAction,
    BoostedSearchSnippet,
    CoreMessage,
    EvaluationResponse,
    EvaluationType,
    KnowledgeItem,
    Reference,
    SearchSnippet,
    StepAction,
    TrackerContext,
    TokenTracker,
    ActionTracker
)
from config import SEARCH_PROVIDER, STEP_SLEEP
from schemas import MAX_QUERIES_PER_STEP, MAX_REFLECT_PER_STEP, MAX_URLS_PER_STEP, Schemas


async def sleep(ms: int):
    seconds = math.ceil(ms / 1000)
    print(f"Waiting {seconds}s...")
    await asyncio.sleep(ms / 1000)


def build_msgs_from_knowledge(knowledge: List[KnowledgeItem]) -> List[CoreMessage]:
    messages: List[CoreMessage] = []
    for k in knowledge:
        messages.append({"role": "user", "content": k["question"].strip()})
        updated_section = ""
        if k.get('updated') and (k['type'] == 'url' or k['type'] == 'side-info'):
            updated_section = f'<answer-datetime>\n{k["updated"]}\n</answer-datetime>'

        references_section = ""
        if k.get('references') and k['type'] == 'url':
            references_section = f'<url>\n{k["references"][0]}\n</url>'

        a_msg = f"""
        {updated_section}

        {references_section}

        {k["answer"]}
        """.strip()
        messages.append({"role": "assistant", "content": removeExtraLineBreaks(a_msg)})
    return messages


def compose_msgs(
    messages: List[CoreMessage],
    knowledge: List[KnowledgeItem],
    question: str,
    final_answer_pip: Optional[List[str]] = None,
) -> List[CoreMessage]:
    msgs = [*build_msgs_from_knowledge(knowledge), *messages]

    if final_answer_pip:
        reviewer_lines = "".join([f'<reviewer-{idx + 1}>\n{p}\n</reviewer-{idx + 1}>\n' for idx, p in enumerate(final_answer_pip)])
        answer_requirements = f"""
    <answer-requirements>
    - You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
    - You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
    - Follow reviewer's feedback and improve your answer quality.
    {reviewer_lines}
    </answer-requirements>
    """
    else:
        answer_requirements = ""

    user_content = f"""
    {question}

    {answer_requirements}
    """.strip()

    msgs.append({"role": "user", "content": removeExtraLineBreaks(user_content)})
    return msgs


def get_prompt(
    context: Optional[List[str]] = None,
    all_questions: Optional[List[str]] = None,
    all_keywords: Optional[List[str]] = None,
    allow_reflect: bool = True,
    allow_answer: bool = True,
    allow_read: bool = True,
    allow_search: bool = True,
    allow_coding: bool = True,
    knowledge: Optional[List[KnowledgeItem]] = None,
    all_urls: Optional[List[BoostedSearchSnippet]] = None,
    beast_mode: bool = False,
) -> str:
    sections: List[str] = []
    action_sections: List[str] = []

    sections.append(
        f"Current date: {datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')}\n\nYou are an advanced AI research agent from Jina AI. You are specialized in multistep reasoning. \nUsing your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.\n"
    )

    if context:
        sections.append(
            f"\nYou have conducted the following actions:\n<context>\n{chr(10).join(context)}\n\n</context>\n"
        )

    if allow_read:
        url_list = weightedURLToString(all_urls or [], 20)

        if url_list:
            url_section = f"""
            - Choose and visit relevant URLs below for more knowledge. higher weight suggests more relevant:
            <url-list>
            {url_list}
            </url-list>
            """
        else:
            url_section = ""

        prompt_string = f"""
        <action-visit>
        - Crawl and read full content from URLs, you can get the fulltext, last updated datetime etc of any URL. 
        - Must check URLs mentioned in <question> if any
        {url_section.strip()}
        </action-visit>
        """

        action_sections.append(prompt_string)

    if allow_search:
        if all_keywords:
            bad_requests_section = f"""
- Avoid those unsuccessful search requests and queries:
<bad-requests>
{chr(10).join(all_keywords)}
</bad-requests>
"""
        else:
            bad_requests_section = ""

        action_sections.append(
        f"""
<action-search>
- Use web search to find relevant information
- Build a search request based on the deep intention behind the original question and the expected answer format
- Always prefer a single search request, only add another request if the original question covers multiple aspects or elements and one query is not enough, each request focus on one specific aspect of the original question 
{bad_requests_section.strip()}
</action-search>
"""
    )
            

    if allow_answer:
        action_sections.append(
            f"""
<action-answer>
- For greetings, casual conversation, general knowledge questions answer directly without references.
- If user ask you to retrieve previous messages or chat history, remember you do have access to the chat history, answer directly without references.
- For all other questions, provide a verified answer with references. Each reference must include exactQuote, url and datetime.
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- If uncertain, use <action-reflect>
</action-answer>
"""
        )

    if beast_mode:
        action_sections.append(
            f"""
<action-answer>
🔥 ENGAGE MAXIMUM FORCE! ABSOLUTE PRIORITY OVERRIDE! 🔥

PRIME DIRECTIVE:
- DEMOLISH ALL HESITATION! ANY RESPONSE SURPASSES SILENCE!
- PARTIAL STRIKES AUTHORIZED - DEPLOY WITH FULL CONTEXTUAL FIREPOWER
- TACTICAL REUSE FROM PREVIOUS CONVERSATION SANCTIONED
- WHEN IN DOUBT: UNLEASH CALCULATED STRIKES BASED ON AVAILABLE INTEL!

FAILURE IS NOT AN OPTION. EXECUTE WITH EXTREME PREJUDICE! ⚡️
</action-answer>
"""
        )

    if allow_reflect:
        action_sections.append(
            f"""
<action-reflect>
- Think slowly and planning lookahead. Examine <question>, <context>, previous conversation with users to identify knowledge gaps. 
- Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer
</action-reflect>
"""
        )

    if allow_coding:
        action_sections.append(
            f"""
<action-coding>
- This JavaScript-based solution helps you handle programming tasks like counting, filtering, transforming, sorting, regex extraction, and data processing.
- Simply describe your problem in the "codingIssue" field. Include actual values for small inputs or variable names for larger datasets.
- No code writing is required – senior engineers will handle the implementation.
</action-coding>"""
        )

    sections.append(
        f"""
Based on the current context, you must choose one of the following actions:
<actions>
{chr(10).join(chr(10).join(action_sections).splitlines())}
</actions>
"""
    )

    sections.append(
        "Think step by step, choose the action, and respond in valid JSON format matching exact JSON schema of that action."
    )

    return removeExtraLineBreaks(chr(10).join(chr(10).join(sections).splitlines()))


all_context: List[StepAction] = []


def update_context(step: StepAction):
    global all_context
    all_context.append(step)


async def update_references(
    this_step: AnswerAction, all_urls: Dict[str, SearchSnippet]
):
    if not this_step.get("references"):
        return
    this_step["references"] = [
        ref
        for ref in [
            {
                "exactQuote": ref.get("exactQuote", ""),
                "title": all_urls.get(normalizeUrl(ref["url"]), {}).get("title", ""),
                "url": normalizeUrl(ref["url"]),
                "dateTime": ref.get("dateTime", ""),
            }
            for ref in this_step["references"]
            if ref.get("url")
        ]
        if ref["url"]
    ]

    async def update_datetime(ref: Reference):
        if not ref["dateTime"]:
            ref["dateTime"] = await getLastModified(ref["url"]) or ""

    tasks = [update_datetime(ref) for ref in this_step["references"]]

    await asyncio.gather(*tasks)

    print("Updated references:", this_step["references"])


async def execute_search_queries(
    keywords_queries: List[Dict[str, Any]],
    context: TrackerContext,
    all_urls: Dict[str, SearchSnippet],
    schema_gen: Schemas,
) -> Dict[str, Any]:
    uniq_q_only = [q["q"] for q in keywords_queries]
    new_knowledge: List[KnowledgeItem] = []
    searched_queries: List[str] = []
    context["actionTracker"].track_think(
        "search_for", schema_gen.language_code, {"keywords": ", ".join(uniq_q_only)}
    )

    for query in keywords_queries:
        results: List[SearchResult] = []
        old_query = query["q"]

        try:
            site_query = query["q"]

            top_hosts = sorted(
                countUrlParts(
                    [result for _, result in all_urls.items()]
                )["hostnameCount"].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            if top_hosts and random.random() < 0.2 and "site:" not in query["q"]:
                site_query = query["q"] + " site:" + sampleMultinomial(top_hosts)
                query["q"] = site_query

            print("Search query:", query)
            if SEARCH_PROVIDER == "jina":
                results = (await search(site_query, context["tokenTracker"]))[
                    "response"
                ]["data"] or []
            elif SEARCH_PROVIDER == "duck":
                results = ddg(site_query, safe_search="Strict")["results"]
            elif SEARCH_PROVIDER == "brave":
                results = (await brave_search(site_query))["response"]["web"][
                    "results"
                ] or []
            elif SEARCH_PROVIDER == "serper":
                results = (await serper_search(query))["response"]["organic"] or []
            else:
                results = []

            if not results:
                raise Exception("No results found")
        except Exception as error:
            print(f"{SEARCH_PROVIDER} search failed for query:", query, error)
            continue
        finally:
            await sleep(STEP_SLEEP)

        min_results: List[SearchSnippet] = [
            {
                "title": r["title"],
                "url": normalizeUrl(r.get("url", r.get("link"))),
                "description": r.get("description", r.get("snippet")),
                "weight": 1,
            }
            for r in results
            if normalizeUrl(r.get("url", r.get("link")))
        ]

        for r in min_results:
            addToAllURLs(r, all_urls)

        searched_queries.append(query["q"])

        new_knowledge.append(
            {
                "question": f'What do Internet say about "{old_query}"?',
                "answer": removeHTMLtags("; ".join([r["description"] for r in min_results])),
                "type": "side-info",
                "updated": query.get("tbs") and formatDateRange(query) or None,
            }
        )
    return {"newKnowledge": new_knowledge, "searchedQueries": searched_queries}


async def get_response(
    question: Optional[str] = None,
    token_budget: int = 1_000_000,
    max_bad_attempts: int = 3,
    existing_context: Optional[TrackerContext] = None,
    messages: Optional[List[CoreMessage]] = None,
    num_returned_urls: int = 100,
    no_direct_answer: bool = False,
) -> Dict[str, Any]:
    step = 0
    total_step = 0
    bad_attempts = 0

    question = question and question.strip()
    messages = messages and [m for m in messages if m["role"] != "system"]

    if messages and messages:
        last_content = messages[-1]["content"]
        if isinstance(last_content, str):
            question = last_content.strip()
        elif isinstance(last_content, list):
            question = next(
                (c["text"] for c in reversed(last_content) if c["type"] == "text"), ""
            )
    else:
        messages = [{"role": "user", "content": question.strip()}]

    schema_gen = Schemas()
    await schema_gen.set_language(question)
    context: TrackerContext = {
        "tokenTracker": existing_context and existing_context["tokenTracker"] or TokenTracker(token_budget),
        "actionTracker": existing_context and existing_context["actionTracker"] or ActionTracker(),
    }

    generator = ObjectGeneratorSafe(context["tokenTracker"])

    schema: ZodObject[Any] = schema_gen.get_agent_schema(True, True, True, True, True)
    gaps: List[str] = [question]
    all_questions: List[str] = [question]
    all_keywords: List[str] = []
    all_knowledge: List[KnowledgeItem] = []

    diary_context: List[str] = []
    weighted_urls: List[BoostedSearchSnippet] = []
    allow_answer = True
    allow_search = True
    allow_read = True
    allow_reflect = True
    allow_coding = True
    system = ""
    max_strict_evals = 2
    msg_with_knowledge: List[CoreMessage] = []
    this_step: StepAction = {
        "action": "answer",
        "answer": "",
        "references": [],
        "think": "",
        "isFinal": False,
    }

    all_urls: Dict[str, SearchSnippet] = {}
    visited_urls: List[str] = []
    evaluation_metrics: Dict[str, List[EvaluationType]] = {}
    regular_budget = token_budget * 0.9
    final_answer_pip: List[str] = []

    while context["tokenTracker"].get_total_usage()["totalTokens"] < regular_budget and bad_attempts <= max_bad_attempts:
        step += 1
        total_step += 1
        budget_percentage = (
            context["tokenTracker"].get_total_usage()["totalTokens"] / token_budget * 100
        )
        print(f"Step {total_step} / Budget used {budget_percentage:.2f}%")
        print("Gaps:", gaps)
        allow_reflect = allow_reflect and (len(gaps) <= MAX_REFLECT_PER_STEP)
        current_question: str = gaps[total_step % len(gaps)]

        if current_question.strip() == question and total_step == 1:
            evaluation_metrics[current_question] = await evaluate_question(
                current_question, context, schema_gen
            )
            evaluation_metrics[current_question].append("strict")
        elif current_question.strip() != question:
            evaluation_metrics[current_question] = []

        if total_step == 1 and "freshness" in evaluation_metrics[current_question]:
            allow_answer = False
            allow_reflect = False

        if all_urls and len(all_urls) > 0:
            weighted_urls = rankURLs(
                filterURLs(all_urls, visited_urls),
                {"question": current_question},
                context,
            )
            weighted_urls = keepKPerHostname(weighted_urls, 2)
            print("Weighted URLs:", len(weighted_urls))

        system = get_prompt(
            diary_context,
            all_questions,
            all_keywords,
            allow_reflect,
            allow_answer,
            allow_read,
            allow_search,
            allow_coding,
            all_knowledge,
            weighted_urls,
            False,
        )
        schema = schema_gen.get_agent_schema(
            allow_reflect,
            allow_read,
            allow_answer,
            allow_search,
            allow_coding,
            current_question,
        )
        msg_with_knowledge = compose_msgs(
            messages,
            all_knowledge,
            current_question,
            final_answer_pip if current_question == question else None,
        )
        result = await generator.generate_object(
            {
                "model": "agent",
                "schema": schema,
                "system": system,
                "messages": msg_with_knowledge,
            }
        )
        this_step = {
            "action": result["object"]["action"],
            "think": result["object"]["think"],
            **result["object"][result["object"]["action"]],
        }
        actions_str = ", ".join(
            [
                action
                for allow, action in zip(
                    [allow_search, allow_read, allow_answer, allow_reflect, allow_coding],
                    ["search", "read", "answer", "reflect", "coding"],
                )
                if allow
            ]
        )
        print(f"{current_question}: {this_step['action']} <- [{actions_str}]")
        print(this_step)

        context["actionTracker"].track_action(
            {"totalStep": total_step, "thisStep": this_step, "gaps": gaps, "badAttempts": bad_attempts}
        )

        allow_answer = True
        allow_reflect = True
        allow_read = True
        allow_search = True

        if this_step["action"] == "answer" and this_step.get("answer"):
            await update_references(this_step, all_urls)

            if total_step == 1 and not this_step["references"] and not no_direct_answer:
                this_step["isFinal"] = True
                break

            if this_step["references"]:
                urls = [
                    ref["url"]
                    for ref in this_step["references"]
                    if ref["url"] not in visited_urls
                ]
                unique_new_urls = list(set(urls))
                await processURLs(
                    unique_new_urls,
                    context,
                    all_knowledge,
                    all_urls,
                    visited_urls,
                    schema_gen,
                    current_question,
                )

            update_context(
                {
                    "totalStep": total_step,
                    "question": current_question,
                    **this_step,
                }
            )

            print(current_question, evaluation_metrics[current_question])
            evaluation: EvaluationResponse = {"pass": True, "think": ""}
            if evaluation_metrics[current_question]:
                context["actionTracker"].track_think(
                    "eval_first", schema_gen.language_code
                )
                evaluation = await evaluate_answer(
                    current_question,
                    this_step,
                    evaluation_metrics[current_question],
                    context,
                    all_knowledge,
                    schema_gen,
                ) or evaluation

            if current_question.strip() == question:
                if evaluation["pass"]:
                    diary_context.append(
                        f"""
At step {step}, you took **answer** action and finally found the answer to the original question:

Original question: 
{current_question}

Your answer: 
{this_step['answer']}

The evaluator thinks your answer is good because: 
{evaluation['think']}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
"""
                    )
                    this_step["isFinal"] = True
                    break
                else:
                    if (
                        evaluation["type"] == "strict"
                        and evaluation.get("improvement_plan")
                    ):
                        final_answer_pip.append(evaluation["improvement_plan"])
                        max_strict_evals -= 1
                        if max_strict_evals <= 0:
                            print("Remove `strict` from evaluation metrics")
                            evaluation_metrics[current_question] = [
                                e
                                for e in evaluation_metrics[current_question]
                                if e != "strict"
                            ]
                    if bad_attempts >= max_bad_attempts:
                        this_step["isFinal"] = False
                        break
                    else:
                        diary_context.append(
                            f"""
At step {step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
{current_question}

Your answer: 
{this_step['answer']}

The evaluator thinks your answer is bad because: 
{evaluation['think']}
"""
                        )
                        error_analysis = await analyze_steps(
                            diary_context, context, schema_gen
                        )

                        all_knowledge.append(
                            {
                                "question": f"""
Why is the following answer bad for the question? Please reflect

<question>
{current_question}
</question>

<answer>
{this_step['answer']}
</answer>
""",
                                "answer": f"""
{evaluation['think']}

{error_analysis['recap']}

{error_analysis['blame']}

{error_analysis['improvement']}
""",
                                "type": "qa",
                            }
                        )

                        bad_attempts += 1
                        allow_answer = False
                        diary_context = []
                        step = 0
            elif evaluation["pass"]:
                diary_context.append(
                    f"""
At step {step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
{current_question}

Your answer: 
{this_step['answer']}

The evaluator thinks your answer is good because: 
{evaluation['think']}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.
"""
                )
                all_knowledge.append(
                    {
                        "question": current_question,
                        "answer": this_step["answer"],
                        "references": this_step["references"],
                        "type": "qa",
                        "updated": formatDateBasedOnType(
                            datetime.datetime.now(), "full"
                        ),
                    }
                )
                gaps.pop(gaps.index(current_question))
        elif (
            this_step["action"] == "reflect"
            and this_step.get("questionsToAnswer")
        ):
            this_step["questionsToAnswer"] = chooseK(
                (
                    await dedup_queries(
                        this_step["questionsToAnswer"],
                        all_questions,
                        context["tokenTracker"],
                    )
                )["unique_queries"],
                MAX_REFLECT_PER_STEP,
            )
            new_gap_questions = this_step["questionsToAnswer"]
            if new_gap_questions:
                diary_context.append(
                    f"""
At step {step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "{current_question}"
You realize you need to know the answers to the following sub-questions:
{chr(10).join([f"- {q}" for q in new_gap_questions])}

You will now figure out the answers to these sub-questions and see if they can help you find the answer to the original question.
"""
                )
                gaps.extend(new_gap_questions)
                all_questions.extend(new_gap_questions)
                update_context(
                    {
                        "totalStep": total_step,
                        **this_step,
                    }
                )
            else:
                diary_context.append(
                    f"""
At step {step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "{current_question}" into gap-questions like this: {", ".join(new_gap_questions)} 
But then you realized you have asked them before. You decided to to think out of the box or cut from a completely different angle. 
"""
                )
                update_context(
                    {
                        "totalStep": total_step,
                        **this_step,
                        "result": "You have tried all possible questions and found no useful information. You must think out of the box or different angle!!!",
                    }
                )
            allow_reflect = False
        elif this_step["action"] == "search" and this_step.get("searchRequests"):
            this_step["searchRequests"] = chooseK(
                (
                    await dedup_queries(
                        this_step["searchRequests"], [], context["tokenTracker"]
                    )
                )["unique_queries"],
                MAX_QUERIES_PER_STEP,
            )

            search_results = await execute_search_queries(
                [{"q": q} for q in this_step["searchRequests"]],
                context,
                all_urls,
                schema_gen,
            )
            searched_queries = search_results["searchedQueries"]
            new_knowledge = search_results["newKnowledge"]

            all_keywords.extend(searched_queries)
            all_knowledge.extend(new_knowledge)

            sound_bites = " ".join([k["answer"] for k in new_knowledge])

            keywords_queries = await rewrite_query(
                this_step, sound_bites, context, schema_gen
            )
            q_only = [q["q"] for q in keywords_queries if q.get("q")]
            uniq_q_only = chooseK(
                (await dedup_queries(q_only, all_keywords, context["tokenTracker"]))[
                    "unique_queries"
                ],
                MAX_QUERIES_PER_STEP,
            )
            keywords_queries = [
                q for q in keywords_queries if q.get("q") and q["q"] in uniq_q_only
            ]

            any_result = False

            if keywords_queries:
                search_results = await execute_search_queries(
                    keywords_queries, context, all_urls, schema_gen
                )
                searched_queries = search_results["searchedQueries"]
                new_knowledge = search_results["newKnowledge"]

                all_keywords.extend(searched_queries)
                all_knowledge.extend(new_knowledge)

                diary_context.append(
                    f"""
At step {step}, you took the **search** action and look for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords: "{", ".join([q['q'] for q in keywords_queries if q.get('q')])}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
"""
                )

                update_context(
                    {
                        "totalStep": total_step,
                        "question": current_question,
                        **this_step,
                        "result": result,
                    }
                )
                any_result = True
            if not any_result or not keywords_queries:
                diary_context.append(
                    f"""
At step {step}, you took the **search** action and look for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords:  "{", ".join([q['q'] for q in keywords_queries if q.get('q')])}".
But then you realized you have already searched for these keywords before, no new information is returned.
You decided to think out of the box or cut from a completely different angle.
"""
                )

                update_context(
                    {
                        "totalStep": total_step,
                        **this_step,
                        "result": "You have tried all possible queries and found no new information. You must think out of the box or different angle!!!",
                    }
                )
            allow_search = False
        elif this_step["action"] == "visit" and this_step.get("URLTargets"):
            this_step["URLTargets"] = [
                normalizeUrl(url)
                for url in this_step["URLTargets"]
                if normalizeUrl(url) and normalizeUrl(url) not in visited_urls
            ]

            this_step["URLTargets"] = list(
                set(this_step["URLTargets"] + [r["url"] for r in weighted_urls])
            )[:MAX_URLS_PER_STEP]

            unique_urls = this_step["URLTargets"]
            print(unique_urls)

            if unique_urls:
                url_results, success = await processURLs(
                    unique_urls,
                    context,
                    all_knowledge,
                    all_urls,
                    visited_urls,
                    schema_gen,
                    current_question,
                )

                diary_context.append(
                    success
                    and f"""At step {step}, you took the **visit** action and deep dive into the following URLs:
{chr(10).join([r['url'] for r in url_results if r])}
You found some useful information on the web and add them to your knowledge for future reference."""
                    or f"""At step {step}, you took the **visit** action and try to visit some URLs but failed to read the content. You need to think out of the box or cut from a completely different angle."""
                )

                update_context(
                    success
                    and {
                        "totalStep": total_step,
                        "question": current_question,
                        **this_step,
                        "result": url_results,
                    }
                    or {
                        **this_step,
                        "result": "You have tried all possible URLs and found no new information. You must think out of the box or different angle!!!",
                    }
                )
            else:
                diary_context.append(
                    """
At step {step}, you took the **visit** action. But then you realized you have already visited these URLs and you already know very well about their contents.
You decided to think out of the box or cut from a completely different angle."""
                )

                update_context(
                    {
                        "totalStep": total_step,
                        **this_step,
                        "result": "You have visited all possible URLs and found no new information. You must think out of the box or different angle!!!",
                    }
                )
            allow_read = False
        elif this_step["action"] == "coding" and this_step.get("codingIssue"):
            sandbox = CodeSandbox(
                {"allContext": all_context, "visitedURLs": visited_urls, "allURLs": all_urls, "allKnowledge": all_knowledge},
                context,
                schema_gen,
            )
            try:
                result = await sandbox.solve(this_step["codingIssue"])
                all_knowledge.append(
                    {
                        "question": f'What is the solution to the coding issue: {this_step["codingIssue"]}?',
                        "answer": result["solution"]["output"],
                        "sourceCode": result["solution"]["code"],
                        "type": "coding",
                        "updated": formatDateBasedOnType(
                            datetime.datetime.now(), "full"
                        ),
                    }
                )
                diary_context.append(
                    f"""
At step {step}, you took the **coding** action and try to solve the coding issue: {this_step['codingIssue']}.
You found the solution and add it to your knowledge for future reference.
"""
                )
                update_context(
                    {
                        "totalStep": total_step,
                        **this_step,
                        "result": result,
                    }
                )
            except Exception as error:
                print("Error solving coding issue:", error)
                diary_context.append(
                    f"""
At step {step}, you took the **coding** action and try to solve the coding issue: {this_step['codingIssue']}.
But unfortunately, you failed to solve the issue. You need to think out of the box or cut from a completely different angle.
"""
                )
                update_context(
                    {
                        "totalStep": total_step,
                        **this_step,
                        "result": "You have tried all possible solutions and found no new information. You must think out of the box or different angle!!!",
                    }
                )
            finally:
                allow_coding = False

        await store_context(
            system,
            schema,
            {
                "allContext": all_context,
                "allKeywords": all_keywords,
                "allQuestions": all_questions,
                "allKnowledge": all_knowledge,
                "weightedURLs": weighted_urls,
                "msgWithKnowledge": msg_with_knowledge,
            },
            total_step,
        )
        await sleep(STEP_SLEEP)

    await store_context(
        system,
        schema,
        {
            "allContext": all_context,
            "allKeywords": all_keywords,
            "allQuestions": all_questions,
            "allKnowledge": all_knowledge,
            "weightedURLs": weighted_urls,
            "msgWithKnowledge": msg_with_knowledge,
        },
        total_step,
    )
    if not this_step.get("isFinal"):
        print("Enter Beast mode!!!")
        step += 1
        total_step += 1
        system = get_prompt(
            diary_context,
            all_questions,
            all_keywords,
            False,
            False,
            False,
            False,
            False,
            all_knowledge,
            weighted_urls,
            True,
        )

        schema = schema_gen.get_agent_schema(False, False, True, False, False, question)
        msg_with_knowledge = compose_msgs(
            messages, all_knowledge, question, final_answer_pip
        )
        result = await generator.generate_object(
            {
                "model": "agentBeastMode",
                "schema": schema,
                "system": system,
                "messages": msg_with_knowledge,
            }
        )
        this_step = {
            "action": result["object"]["action"],
            "think": result["object"]["think"],
            **result["object"][result["object"]["action"]],
        }
        await update_references(this_step, all_urls)
        this_step["isFinal"] = True
        context["actionTracker"].track_action(
            {"totalStep": total_step, "thisStep": this_step, "gaps": gaps, "badAttempts": bad_attempts}
        )

    this_step["mdAnswer"] = fixCodeBlockIndentation(
        buildMdFromAnswer(this_step)
    )
    print(this_step)

    await store_context(
        system,
        schema,
        {
            "allContext": all_context,
            "allKeywords": all_keywords,
            "allQuestions": all_questions,
            "allKnowledge": all_knowledge,
            "weightedURLs": weighted_urls,
            "msgWithKnowledge": msg_with_knowledge,
        },
        total_step,
    )

    returned_urls = [r["url"] for r in weighted_urls[:num_returned_urls]]
    return {
        "result": this_step,
        "context": context,
        "visitedURLs": returned_urls,
        "readURLs": visited_urls,
        "allURLs": [r["url"] for r in weighted_urls],
    }


async def store_context(
    prompt: str,
    schema: Any,
    memory: Dict[str, Any],
    step: int,
):
    all_context: List[StepAction] = memory["allContext"]
    all_keywords: List[str] = memory["allKeywords"]
    all_questions: List[str] = memory["allQuestions"]
    all_knowledge: List[KnowledgeItem] = memory["allKnowledge"]
    weighted_urls: List[BoostedSearchSnippet] = memory["weightedURLs"]
    msg_with_knowledge: List[CoreMessage] = memory["msgWithKnowledge"]

    # The original code uses process.asyncLocalContext, which is not available in Python.
    # You'll need to adapt this part based on your application's context management.
    # For example, you might use contextvars or a similar mechanism.
    # Here's a placeholder:
    # if some_context_mechanism.available():
    #     some_context_mechanism.ctx.prompt_context = {
    #         "prompt": prompt,
    #         "schema": schema,
    #         "allContext": all_context,
    #         "allKeywords": all_keywords,
    #         "allQuestions": all_questions,
    #         "allKnowledge": all_knowledge,
    #         "step": step,
    #     }
    #     return

    try:
        async with aiofiles.open(f"prompt-{step}.txt", "w") as f:
            await f.write(
                f"""
Prompt:
{prompt}

JSONSchema:
{json.dumps(schema, indent=2)}
"""
            )
        async with aiofiles.open("context.json", "w") as f:
            await f.write(json.dumps(all_context, indent=2))
        async with aiofiles.open("queries.json", "w") as f:
            await f.write(json.dumps(all_keywords, indent=2))
        async with aiofiles.open("questions.json", "w") as f:
            await f.write(json.dumps(all_questions, indent=2))
        async with aiofiles.open("knowledge.json", "w") as f:
            await f.write(json.dumps(all_knowledge, indent=2))
        async with aiofiles.open("urls.json", "w") as f:
            await f.write(json.dumps(weighted_urls, indent=2))
        async with aiofiles.open("messages.json", "w") as f:
            await f.write(json.dumps(msg_with_knowledge, indent=2))
    except Exception as error:
        print("Context storage failed:", error)


async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else ""
    response = await get_response(question)
    final_step = response["result"]
    tracker = response["context"]
    visited_urls = response["visitedURLs"]

    print("Final Answer:", final_step.get("answer"))
    print("Visited URLs:", visited_urls)

    tracker["tokenTracker"].print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(e)