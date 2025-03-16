
from typing import List, Dict, Optional, TypedDict, Union, Any


#Added TrackerContext class.
class TrackerContext:
    def __init__(self, tokenTracker: TokenTracker, actionTracker: ActionTracker):
        self.tokenTracker = tokenTracker
        self.actionTracker = actionTracker



class ActionTracker(EventEmitter):
    def __init__(self):
        super().__init__()
        self._state: ActionState = {
            'thisStep': {'action': 'answer', 'answer': '', 'references': [], 'think': ''},
            'gaps': [],
            'badAttempts': 0,
            'totalStep': 0
        }

    def track_action(self, new_state: Dict) -> None:
        self._state.update(new_state)
        self.emit('action', self._state['thisStep'])

    def track_think(self, think: str, lang: Optional[str] = None, params: Dict = {}) -> None:
        if lang:
            think = getI18nText(think, lang, params)
        self._state['thisStep']['think'] = think
        self.emit('action', self._state['thisStep'])

    def get_state(self) -> ActionState:
        return self._state.copy()

    def reset(self) -> None:
        self._state = {
            'thisStep': {'action': 'answer', 'answer': '', 'references': [], 'think': ''},
            'gaps': [],
            'badAttempts': 0,
            'totalStep': 0
        }



class TokenTracker(EventEmitter):
    def __init__(self, budget: Optional[int] = None):
        super().__init__()
        self._usages: List[TokenUsage] = []
        self._budget: Optional[int] = budget

        # Assuming process.asyncLocalContext is not directly available in Python.
        # This part of the code needs adaptation based on your environment.
        # In a typical python environment, async local storage is handled by contextvars.
        # This part of the code is heavily dependent on the environment.
        # I will leave the general structure, but you will need to replace the asyncLocalContext logic.
        # Example for contextvars is below, but I am commenting it out.
        """
        import contextvars

        charge_amount = contextvars.ContextVar('charge_amount', default=0)

        def on_usage_event():
            if charge_amount.get() is not None:
                charge_amount.set(self.get_total_usage()['totalTokens'])

        self.on('usage', on_usage_event)
        """

    def track_usage(self, tool: str, usage: LanguageModelUsage) -> None:
        u: TokenUsage = {'tool': tool, 'usage': usage}
        self._usages.append(u)
        self.emit('usage', usage)

    def get_total_usage(self) -> LanguageModelUsage:
        initial_usage: LanguageModelUsage = {'promptTokens': 0, 'completionTokens': 0, 'totalTokens': 0}
        return {
            'promptTokens': sum(u['usage']['promptTokens'] for u in self._usages),
            'completionTokens': sum(u['usage']['completionTokens'] for u in self._usages),
            'totalTokens': sum(u['usage']['totalTokens'] for u in self._usages),
        }

    def get_total_usage_snake_case(self) -> Dict[str, int]:
        return {
            'prompt_tokens': sum(u['usage']['promptTokens'] for u in self._usages),
            'completion_tokens': sum(u['usage']['completionTokens'] for u in self._usages),
            'total_tokens': sum(u['usage']['totalTokens'] for u in self._usages),
        }

    def get_usage_breakdown(self) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for u in self._usages:
            tool = u['tool']
            usage = u['usage']['totalTokens']
            breakdown[tool] = breakdown.get(tool, 0) + usage
        return breakdown

    def print_summary(self) -> None:
        breakdown = self.get_usage_breakdown()
        print('Token Usage Summary:', {
            'budget': self._budget,
            'total': self.get_total_usage(),
            'breakdown': breakdown
        })

    def reset(self) -> None:
        self._usages = []



class StepAction(TypedDict):
    action: str
    answer: str
    references: List[str]
    think: str


class TokenUsage(TypedDict):
    tool: str
    usage: LanguageModelUsage




# Assuming CoreMessage, i18nJSON are defined elsewhere.
# Replace these with your actual implementations.

class CoreMessage(TypedDict):
    role: str
    content: str

class LanguageModelUsage(TypedDict):
    promptTokens: int
    completionTokens: int
    totalTokens: int

class SERPQuery(TypedDict):
    q: str
    hl: Optional[str]
    gl: Optional[str]
    location: Optional[str]
    tbs: Optional[str]

class Reference(TypedDict):
    exactQuote: str
    url: str
    title: str
    dateTime: Optional[str]

class BaseAction(TypedDict):
    action: str
    think: str

class SearchAction(BaseAction):
    action: str
    searchRequests: List[str]

class AnswerAction(BaseAction):
    action: str
    answer: str
    references: List[Reference]
    isFinal: Optional[bool]
    mdAnswer: Optional[str]

class KnowledgeItem(TypedDict):
    question: str
    answer: str
    references: Optional[Union[List[Reference], List[Any]]]
    type: str
    updated: Optional[str]
    sourceCode: Optional[str]

class ReflectAction(BaseAction):
    action: str
    questionsToAnswer: List[str]

class VisitAction(BaseAction):
    action: str
    URLTargets: List[str]

class CodingAction(BaseAction):
    action: str
    codingIssue: str

StepAction = Union[SearchAction, AnswerAction, ReflectAction, VisitAction, CodingAction]

EvaluationType = str


class SearchResponse(TypedDict):
    code: int
    status: int
    data: Optional[List[Dict[str, Any]]]
    name: Optional[str]
    message: Optional[str]
    readableMessage: Optional[str]

class BraveSearchResponse(TypedDict):
    web: Dict[str, List[Dict[str, str]]]

class SerperSearchResponse(TypedDict):
    knowledgeGraph: Optional[Dict[str, Any]]
    organic: List[Dict[str, Any]]
    topStories: Optional[List[Dict[str, Any]]]
    relatedSearches: Optional[List[str]]
    credits: int

class ReadResponse(TypedDict):
    code: int
    status: int
    data: Optional[Dict[str, Any]]
    name: Optional[str]
    message: Optional[str]
    readableMessage: Optional[str]

class EvaluationResponse(TypedDict):
    pass_: bool
    think: str
    type_: Optional[EvaluationType]
    freshness_analysis: Optional[Dict[str, Any]]
    plurality_analysis: Optional[Dict[str, Any]]
    exactQuote: Optional[str]
    completeness_analysis: Optional[Dict[str, Any]]
    improvement_plan: Optional[str]

class CodeGenResponse(TypedDict):
    think: str
    code: str

class ErrorAnalysisResponse(TypedDict):
    recap: str
    blame: str
    improvement: str

SearchResult = Union[Dict[str, Any], Dict[str, Any]]

SearchSnippet = Dict[str, Any]

BoostedSearchSnippet = Dict[str, Any]

class Model(TypedDict):
    id: str
    object: str
    created: int
    owned_by: str

PromptPair = Dict[str, str]

class ResponseFormat(TypedDict):
    type: str
    json_schema: Optional[Any]

class ChatCompletionRequest(TypedDict):
    model: str
    messages: List[CoreMessage]
    stream: Optional[bool]
    reasoning_effort: Optional[str]
    max_completion_tokens: Optional[int]
    budget_tokens: Optional[int]
    max_attempts: Optional[int]
    response_format: Optional[ResponseFormat]
    no_direct_answer: Optional[bool]
    max_returned_urls: Optional[int]

class URLAnnotation(TypedDict):
    type: str
    url_citation: Reference

class ChatCompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    visitedURLs: Optional[List[str]]
    readURLs: Optional[List[str]]
    numURLs: Optional[int]

class ChatCompletionChunk(TypedDict):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Dict[str, Any]]
    usage: Optional[Any]
    visitedURLs: Optional[List[str]]
    readURLs: Optional[List[str]]
    numURLs: Optional[int]

def getI18nText(key: str, lang: str = 'en', params: Dict[str, str] = {}) -> str:
    i18n_data: Dict[str, Any] = i18nJSON  # Assuming i18nJSON is defined
    if lang not in i18n_data:
        print(f"Language '{lang}' not found, falling back to English.")
        lang = 'en'

    text: Optional[str] = i18n_data.get(lang, {}).get(key)

    if text is None:
        print(f"Key '{key}' not found for language '{lang}', falling back to English.")
        text = i18n_data.get('en', {}).get(key)

        if text is None:
            print(f"Key '{key}' not found for English either.")
            return key

    if params:
        for param_key, param_value in params.items():
            text = text.replace(f"${{{param_key}}}", param_value)

    return text

    
class ActionState(TypedDict):
    thisStep: StepAction
    gaps: List[str]
    badAttempts: int
    totalStep: int