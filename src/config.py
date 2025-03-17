import os
import json
import requests  # For proxy handling if needed

# Load config.json
try:
    with open('config.json', 'r') as f:
        config_json = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found.")
    exit(1)

# Types (as close as possible in Python)
LLMProvider = str  # 'openai' | 'gemini' | 'vertex'
ToolName = str  # Assuming tools are string keys in configJson

# Environment setup
env = dict(config_json['env'])  # Create a copy of the env dict

for key in env:
    if os.environ.get(key):
        env[key] = os.environ.get(key) or env[key]

# Setup proxy if present
if env.get('https_proxy'):
    try:
        proxy_url = requests.utils.urlparse(env['https_proxy']).geturl()  # Parse URL
        proxies = {
            'http': proxy_url,
            'https': proxy_url,
        }
        # In Python, you'd typically configure the proxy for the requests library or
        # any other library that makes HTTP requests.
        # Example using requests:
        # requests.get('some_url', proxies=proxies)
        # Note: undici's ProxyAgent and setGlobalDispatcher are Node.js specific.
        print(f"Proxy configured: {proxy_url}") # just print a message, as undici proxy logic cannot be directly translated.
    except Exception as error:
        print(f'Failed to set proxy: {error}')

# Export environment variables
OPENAI_BASE_URL = env.get('OPENAI_BASE_URL')
GEMINI_API_KEY = env.get('GEMINI_API_KEY')
OPENAI_API_KEY = env.get('OPENAI_API_KEY')
JINA_API_KEY = env.get('JINA_API_KEY')
BRAVE_API_KEY = env.get('BRAVE_API_KEY')
SERPER_API_KEY = env.get('SERPER_API_KEY')
SEARCH_PROVIDER = config_json['defaults']['search_provider']
STEP_SLEEP = config_json['defaults']['step_sleep']

# Determine LLM provider
LLM_PROVIDER: LLMProvider = os.environ.get('LLM_PROVIDER') or config_json['defaults']['llm_provider']

def isValidProvider(provider: str) -> bool:
    return provider in ['openai', 'gemini', 'vertex']

if not isValidProvider(LLM_PROVIDER):
    raise ValueError(f'Invalid LLM provider: {LLM_PROVIDER}')

# Get tool configuration
def getToolConfig(toolName: ToolName) -> dict:
    provider_config = config_json['models']['gemini'] if LLM_PROVIDER == 'vertex' else config_json['models'][LLM_PROVIDER]
    default_config = provider_config['default']
    tool_overrides = provider_config['tools'].get(toolName, {})

    return {
        'model': os.environ.get('DEFAULT_MODEL_NAME') or default_config['model'],
        'temperature': tool_overrides.get('temperature', default_config['temperature']),
        'maxTokens': tool_overrides.get('maxTokens', default_config['maxTokens'])
    }

def getMaxTokens(toolName: ToolName) -> int:
    return getToolConfig(toolName)['maxTokens']

# Get model instance (replace with your actual AI SDK logic)
def getModel(toolName: ToolName):
    config = getToolConfig(toolName)
    provider_config = config_json['providers'].get(LLM_PROVIDER)

    if LLM_PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError('OPENAI_API_KEY not found')

        opt = {
            'apiKey': OPENAI_API_KEY,
            'compatibility': provider_config.get('clientConfig', {}).get('compatibility')
        }

        if OPENAI_BASE_URL:
            opt['baseURL'] = OPENAI_BASE_URL

        # Replace with your actual OpenAI SDK call
        print(f"OpenAI model requested: {config['model']}")
        return config['model'] # placeholder, return model name for now

    elif LLM_PROVIDER == 'vertex':
        # Replace with your actual Vertex AI SDK call
        print(f"Vertex AI model requested: {config['model']}")
        return config['model'] # placeholder, return model name for now

    else:  # Gemini
        if not GEMINI_API_KEY:
            raise ValueError('GEMINI_API_KEY not found')

        # Replace with your actual Gemini SDK call
        print(f"Gemini model requested: {config['model']}")
        return config['model'] # placeholder, return model name for now

# Validate required environment variables
if LLM_PROVIDER == 'gemini' and not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY not found')
if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
    raise ValueError('OPENAI_API_KEY not found')
if not JINA_API_KEY:
    raise ValueError('JINA_API_KEY not found')

# Log all configurations
config_summary = {
    'provider': {
        'name': LLM_PROVIDER,
        'model': config_json['models']['openai']['default']['model'] if LLM_PROVIDER == 'openai' else config_json['models']['gemini']['default']['model'],
        'baseUrl': OPENAI_BASE_URL if LLM_PROVIDER == 'openai' else None
    },
    'search': {
        'provider': SEARCH_PROVIDER
    },
    'tools': {
        name: getToolConfig(name)
        for name in config_json['models']['gemini' if LLM_PROVIDER == 'vertex' else LLM_PROVIDER]['tools']
    },
    'defaults': {
        'stepSleep': STEP_SLEEP
    }
}

print('Configuration Summary:', json.dumps(config_summary, indent=2))