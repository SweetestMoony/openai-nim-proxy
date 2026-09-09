// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());

// 🔧 FIX: definir limite explícito (o padrão do Express é 100kb, muito pequeno
// para prompts longos, histórico de chat ou imagens em base64)
app.use(express.json({ limit: '64mb' }));
app.use(express.urlencoded({ limit: '64mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔧 FIX: timeout para chamadas à NIM API (evita requests pendurados indefinidamente
// quando um modelo está sobrecarregado ou lento, ex: Deep4 Pro)
const NIM_REQUEST_TIMEOUT = parseInt(process.env.NIM_REQUEST_TIMEOUT_MS || '120000', 10); // 120s default

// 🆕 Configuração de retry para erros transitórios da NIM
const NIM_MAX_RETRIES = parseInt(process.env.NIM_MAX_RETRIES || '3', 10);
const NIM_RETRY_BASE_DELAY_MS = parseInt(process.env.NIM_RETRY_BASE_DELAY_MS || '1000', 10);
const RETRYABLE_STATUSES = new Set([410, 429, 500, 502, 503, 504]);

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'llama3': 'meta/llama-3.3-70b-instruct',
  'k3': 'moonshotai/kimi-k3',
  'fast': 'deepseek-ai/deepseek-v4-flash-0731',
  'deep4': 'deepseek-ai/deepseek-v4-pro-0813',
  'step': 'stepfun-ai/step-3.7-flash',
  'nemotron': 'nvidia/nemotron-3-super-120b-a12b'
};

// 🆕 Estado simples de saúde por modelo, só pra observabilidade via /health
const modelHealth = {}; // { [nimModelId]: { failures: number, lastStatus: number|null, lastFailureAt: string|null } }

function recordModelFailure(nimModelId, status) {
  if (!modelHealth[nimModelId]) {
    modelHealth[nimModelId] = { failures: 0, lastStatus: null, lastFailureAt: null };
  }
  modelHealth[nimModelId].failures += 1;
  modelHealth[nimModelId].lastStatus = status;
  modelHealth[nimModelId].lastFailureAt = new Date().toISOString();
}

function recordModelSuccess(nimModelId) {
  if (modelHealth[nimModelId]) {
    modelHealth[nimModelId].failures = 0;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 🆕 Chamada à NIM API com retry/backoff para erros transitórios (410/429/5xx)
async function callNimWithRetry(nimRequest, { stream }) {
  const headers = {
    Authorization: `Bearer ${NIM_API_KEY}`,
    'Content-Type': 'application/json'
  };

  let lastError;

  for (let attempt = 0; attempt <= NIM_MAX_RETRIES; attempt++) {
    try {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers,
        responseType: stream ? 'stream' : 'json',
        // 🔧 FIX: por padrão o Axios limita o corpo enviado/recebido.
        maxBodyLength: 64 * 1024 * 1024,
        maxContentLength: 64 * 1024 * 1024,
        timeout: NIM_REQUEST_TIMEOUT
      });
      recordModelSuccess(nimRequest.model);
      return response;
    } catch (error) {
      lastError = error;
      const status = error.response?.status;
      recordModelFailure(nimRequest.model, status || 'network_error');

      const isRetryable = status ? RETRYABLE_STATUSES.has(status) : true; // erro de rede/timeout também tenta de novo
      const isLastAttempt = attempt === NIM_MAX_RETRIES;

      console.error(
        `NIM call failed (model=${nimRequest.model}, attempt=${attempt + 1}/${NIM_MAX_RETRIES + 1}, status=${status || 'n/a'}):`,
        error.response?.data ? JSON.stringify(error.response.data) : error.message
      );

      if (!isRetryable || isLastAttempt) {
        throw error;
      }

      const retryAfterHeader = error.response?.headers?.['retry-after'];
      const backoff = retryAfterHeader
        ? Number(retryAfterHeader) * 1000
        : NIM_RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
      const jitter = Math.random() * 300;

      console.warn(`Retentando em ${Math.round(backoff + jitter)}ms...`);
      await sleep(backoff + jitter);
    }
  }

  throw lastError;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    request_timeout_ms: NIM_REQUEST_TIMEOUT,
    max_retries: NIM_MAX_RETRIES,
    model_health: modelHealth
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        const testRes = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500,
          timeout: 15000 // teste rápido de fallback, não precisa do timeout longo
        });
        if (testRes.status >= 200 && testRes.status < 300) {
          nimModel = model;
        }
      } catch (e) {
        console.error('Fallback model test failed:', e.message);
      }

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    // Faz a chamada com retry/backoff para erros transitórios (410/429/5xx)
    const response = await callNimWithRetry(nimRequest, { stream: !!stream });

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = '';

                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message, error.response?.data ? JSON.stringify(error.response.data) : '');

    // 🔧 FIX: identificar timeout explicitamente (antes caía tudo em 500 genérico)
    const isTimeout = error.code === 'ECONNABORTED' || error.message?.includes('timeout');
    const status = error.response?.status || (isTimeout ? 504 : 500);

    res.status(status).json({
      error: {
        message: isTimeout
          ? `Request to NIM API timed out after ${NIM_REQUEST_TIMEOUT}ms`
          : (error.response?.data?.error?.message || error.message || 'Internal server error'),
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Request timeout: ${NIM_REQUEST_TIMEOUT}ms`);
  console.log(`Max retries per model: ${NIM_MAX_RETRIES}`);
});
