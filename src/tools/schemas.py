from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field, conlist, constr, conint, conbool, condecimal
from enum import Enum

MAX_URLS_PER_STEP = 4
MAX_QUERIES_PER_STEP = 7
MAX_REFLECT_PER_STEP = 2

class PromptPair(BaseModel):
    system: str
    user: str

def getLanguagePrompt(question: str) -> PromptPair:
    return PromptPair(
        system=f"""Identifies both the language used and the overall vibe of the question

<rules>
Combine both language and emotional vibe in a descriptive phrase, considering:
    - Language: The primary language or mix of languages used
    - Emotional tone: panic, excitement, frustration, curiosity, etc.
    - Formality level: academic, casual, professional, etc.
    - Domain context: technical, academic, social, etc.
</rules>

<examples>
Question: "fam PLEASE help me calculate the eigenvalues of this 4x4 matrix ASAP!! [matrix details] got an exam tmrw 😭"
Evaluation: {{
    "langCode": "en",
    "langStyle": "panicked student English with math jargon"
}}

Question: "Can someone explain how tf did Ferrari mess up their pit stop strategy AGAIN?! 🤦‍♂️ #MonacoGP"
Evaluation: {{
    "langCode": "en",
    "languageStyle": "frustrated fan English with F1 terminology"
}}

Question: "肖老师您好，请您介绍一下最近量子计算领域的三个重大突破，特别是它们在密码学领域的应用价值吗？🤔"
Evaluation: {{
    "langCode": "zh",
    "languageStyle": "formal technical Chinese with academic undertones"
}}

Question: "Bruder krass, kannst du mir erklären warum meine neural network training loss komplett durchdreht? Hab schon alles probiert 😤"
Evaluation: {{
    "langCode": "de",
    "languageStyle": "frustrated German-English tech slang"
}}

Question: "Does anyone have insights into the sociopolitical implications of GPT-4's emergence in the Global South, particularly regarding indigenous knowledge systems and linguistic diversity? Looking for a nuanced analysis."
Evaluation: {{
    "langCode": "en",
    "languageStyle": "formal academic English with sociological terminology"
}}

Question: "what's 7 * 9? need to check something real quick"
Evaluation: {{
    "langCode": "en",
    "languageStyle": "casual English"
}}
</examples>""",
        user=question
    )

class LanguageSchema(BaseModel):
    langCode: constr(max_length=10) = Field(description='ISO 639-1 language code')
    langStyle: constr(max_length=100) = Field(description='[vibe & tone] in [what language], such as formal english, informal chinese, technical german, humor english, slang, genZ, emojis etc.')

class QuestionEvaluateSchema(BaseModel):
    think: constr(max_length=500) = Field(description='A very concise explain of why those checks are needed.')
    needsDefinitive: conbool()
    needsFreshness: conbool()
    needsPlurality: conbool()
    needsCompleteness: conbool()

class CodeGeneratorSchema(BaseModel):
    think: constr(max_length=200) = Field(description='Short explain or comments on the thought process behind the code.')
    code: str = Field(description='The JavaScript code that solves the problem and always use \'return\' statement to return the result. Focus on solving the core problem; No need for error handling or try-catch blocks or code comments. No need to declare variables that are already available, especially big long strings or arrays.')

class ErrorAnalysisSchema(BaseModel):
    recap: constr(max_length=500) = Field(description='Recap of the actions taken and the steps conducted in first person narrative.')
    blame: constr(max_length=500) = Field(description='Which action or the step was the root cause of the answer rejection.')
    improvement: constr(max_length=500) = Field(description='Suggested key improvement for the next iteration, do not use bullet points, be concise and hot-take vibe.')

class QuerySchema(BaseModel):
    tbs: Optional[Enum('TimeBasedSearch', {'qdr:h': 'qdr:h', 'qdr:d': 'qdr:d', 'qdr:w': 'qdr:w', 'qdr:m': 'qdr:m', 'qdr:y': 'qdr:y'})] = Field(description='time-based search filter, must use this field if the search request asks for latest info. qdr:h for past hour, qdr:d for past 24 hours, qdr:w for past week, qdr:m for past month, qdr:y for past year. Choose exactly one.')
    gl: Optional[constr(max_length=2)] = Field(description='defines the country to use for the search. a two-letter country code. e.g., us for the United States, uk for United Kingdom, or fr for France.')
    hl: Optional[constr(max_length=2)] = Field(description='the language to use for the search. a two-letter language code. e.g., en for English, es for Spanish, or fr for French.')
    location: Optional[str] = Field(description='defines from where you want the search to originate. It is recommended to specify location at the city level in order to simulate a real user’s search.')
    q: constr(max_length=50) = Field(description='keyword-based search query, 2-3 words preferred, total length < 30 characters')

class QueryRewriterSchema(BaseModel):
    think: constr(max_length=500) = Field(description='Explain why you choose those search queries.')
    queries: conlist(QuerySchema, max_items=MAX_QUERIES_PER_STEP) = Field(description=f'Array of search keywords queries, orthogonal to each other. Maximum {MAX_QUERIES_PER_STEP} queries allowed.')

class EvaluationType(Enum):
    definitive = "definitive"
    freshness = "freshness"
    plurality = "plurality"
    attribution = "attribution"
    completeness = "completeness"
    strict = "strict"

class EvaluatorSchemaDefinitive(BaseModel):
    type: str = Field(default="definitive")
    think: constr(max_length=500) = Field(description='Explanation the thought process why the answer does not pass the evaluation.')
    pass_: conbool() = Field(alias='pass')

class EvaluatorSchemaFreshness(BaseModel):
    type: str = Field(default="freshness")
    think: constr(max_length=500) = Field(description='Explanation the thought process why the answer does not pass the evaluation.')
    freshness_analysis: dict = Field(description='datetime of the **answer** and relative to today.')
    pass_: conbool() = Field(alias='pass', description='If "days_ago" <= "max_age_days" then pass!')

class EvaluatorSchemaPlurality(BaseModel):
    type: str = Field(default="plurality")
    think: constr(max_length=500) = Field(description='Explanation the thought process why the answer does not pass the evaluation.')
    plurality_analysis: dict = Field(description='Minimum and actual counts of items.')
    pass_: conbool() = Field(alias='pass', description='If count_provided >= count_expected then pass!')

class EvaluatorSchemaAttribution(BaseModel):
    type: str = Field(default="attribution")
    think: constr(max_length=500) = Field(description='Explanation the thought process why the answer does not pass the evaluation.')
    exactQuote: Optional[constr(max_length=200)] = Field(description='Exact relevant quote and evidence from the source.')
    pass_: conbool() = Field(alias='pass')

class EvaluatorSchemaCompleteness(BaseModel):
    type: str = Field(default="completeness")
    think: constr(max_length=500) = Field(description='Explanation the thought process why the answer does not pass the evaluation.')
    completeness_analysis: dict = Field(description='Expected and provided aspects.')
    pass_: conbool() = Field(alias='pass')

class EvaluatorSchemaStrict(BaseModel):
    type: str = Field(default="strict")
    think: constr(max_length=500) = Field(description='Explanation the thought process why the answer does not pass the evaluation.')
    improvement_plan: constr(max_length=500) = Field