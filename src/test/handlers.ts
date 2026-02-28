import { http, HttpResponse } from 'msw';

// Mock API handlers for all backend endpoints
export const handlers = [
  // Auth endpoints
  http.post('http://127.0.0.1:8000/v1/auth/login', () => {
    return HttpResponse.json({
      access_token: 'mock-jwt-token',
      token_type: 'bearer',
      user: {
        id: 1,
        email: 'test@example.com',
        username: 'testuser',
      },
    });
  }),

  http.post('http://127.0.0.1:8000/v1/auth/logout', () => {
    return HttpResponse.json({ message: 'Logged out successfully' });
  }),

  http.get('http://127.0.0.1:8000/v1/auth/me', () => {
    return HttpResponse.json({
      id: 1,
      email: 'test@example.com',
      username: 'testuser',
    });
  }),

  http.post('http://127.0.0.1:8000/v1/auth/validate', () => {
    return HttpResponse.json({
      valid: true,
      user: {
        id: 1,
        email: 'test@example.com',
        username: 'testuser',
      },
    });
  }),

  // Settings endpoints
  http.get('http://127.0.0.1:8000/v1/settings/', () => {
    return HttpResponse.json({
      providers: [
        {
          name: 'openai',
          api_key: null,
          base_url: null,
          models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
          enabled: true,
        },
        {
          name: 'anthropic',
          api_key: null,
          base_url: null,
          models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
          enabled: true,
        },
        {
          name: 'google',
          api_key: null,
          base_url: null,
          models: ['gemini-pro', 'gemini-pro-vision'],
          enabled: true,
        },
        {
          name: 'deepseek',
          api_key: null,
          base_url: null,
          models: ['deepseek-chat', 'deepseek-coder'],
          enabled: true,
        },
        {
          name: 'aliyun',
          api_key: null,
          base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
          models: ['qwen-plus', 'qwen-turbo', 'qwen2.5-coder-32b-instruct'],
          enabled: true,
        },
        {
          name: 'azure_openai',
          api_key: null,
          base_url: 'https://example-resource.openai.azure.com',
          models: ['gpt-4o', 'gpt-4.1', 'gpt-4-turbo'],
          enabled: true,
        },
        {
          name: 'ollama_gcp',
          api_key: null,
          base_url: 'http://34.55.121.214:11434',
          models: ['qwen2.5:3b', 'gemma:2b', 'mistral:7b'],
          enabled: true,
        },
        {
          name: 'llamacpp_gcp',
          api_key: null,
          base_url: 'http://34.123.36.255:8000',
          models: ['phi-3-mini-4k-instruct-q4', 'mistral-7b-instruct-v0.2-q4_k_m'],
          enabled: true,
        },
        {
          name: 'ollama',
          api_key: null,
          base_url: null,
          models: ['llama2', 'codellama', 'mistral'],
          enabled: true,
        },
        {
          name: 'groq',
          api_key: null,
          base_url: null,
          models: ['mixtral-8x7b', 'llama2-70b'],
          enabled: true,
        },
        {
          name: 'together',
          api_key: null,
          base_url: null,
          models: ['llama-2-70b', 'codellama-34b'],
          enabled: true,
        },
        {
          name: 'replicate',
          api_key: null,
          base_url: null,
          models: ['llama-2-70b', 'codellama-34b'],
          enabled: true,
        },
        {
          name: 'huggingface',
          api_key: null,
          base_url: null,
          models: ['microsoft/DialoGPT-medium'],
          enabled: true,
        },
        {
          name: 'cohere',
          api_key: null,
          base_url: null,
          models: ['command', 'base'],
          enabled: true,
        },
        {
          name: 'ai21',
          api_key: null,
          base_url: null,
          models: ['j2-ultra', 'j2-mid'],
          enabled: true,
        },
      ],
      models: [
        {
          name: 'gpt-4',
          provider: 'openai',
          model_id: 'gpt-4',
          temperature: 0.7,
          max_tokens: 4096,
          enabled: true,
        },
        {
          name: 'claude-3',
          provider: 'anthropic',
          model_id: 'claude-3',
          temperature: 0.7,
          max_tokens: 4096,
          enabled: true,
        },
      ],
      default_provider: 'aliyun',
      default_model: 'qwen-plus',
    });
  }),

  // Routing endpoints
  http.get('http://127.0.0.1:8000/v1/routing/providers', () => {
    return HttpResponse.json({
      providers: [
        {
          id: 'openai',
          name: 'OpenAI',
          models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
          capabilities: ['chat', 'reasoning', 'code', 'embedding'],
          status: 'healthy',
          latency: 150,
          cost_per_token: 0.00002,
          reliability: 0.99,
          bandwidth: 1000,
        },
        {
          id: 'anthropic',
          name: 'Anthropic',
          models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 200,
          cost_per_token: 0.000015,
          reliability: 0.98,
          bandwidth: 800,
        },
        {
          id: 'google',
          name: 'Google',
          models: ['gemini-pro', 'gemini-pro-vision'],
          capabilities: ['chat', 'reasoning', 'code', 'image'],
          status: 'healthy',
          latency: 180,
          cost_per_token: 0.00001,
          reliability: 0.97,
          bandwidth: 900,
        },
        {
          id: 'deepseek',
          name: 'DeepSeek',
          models: ['deepseek-chat', 'deepseek-coder'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 120,
          cost_per_token: 0.000005,
          reliability: 0.95,
          bandwidth: 600,
        },
        {
          id: 'aliyun',
          name: 'Aliyun',
          models: ['qwen-plus', 'qwen-turbo', 'qwen2.5-coder-32b-instruct'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 140,
          cost_per_token: 0.0000014,
          reliability: 0.96,
          bandwidth: 650,
        },
        {
          id: 'azure_openai',
          name: 'Azure OpenAI',
          models: ['gpt-4o', 'gpt-4.1', 'gpt-4-turbo'],
          capabilities: ['chat', 'reasoning', 'code', 'embedding'],
          status: 'healthy',
          latency: 130,
          cost_per_token: 0.000006,
          reliability: 0.98,
          bandwidth: 850,
        },
        {
          id: 'ollama_gcp',
          name: 'Ollama (GCP)',
          models: ['qwen2.5:3b', 'gemma:2b', 'mistral:7b'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 70,
          cost_per_token: 0,
          reliability: 0.97,
          bandwidth: 900,
        },
        {
          id: 'llamacpp_gcp',
          name: 'LlamaCpp (GCP)',
          models: ['phi-3-mini-4k-instruct-q4', 'mistral-7b-instruct-v0.2-q4_k_m'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 75,
          cost_per_token: 0,
          reliability: 0.96,
          bandwidth: 850,
        },
        {
          id: 'ollama',
          name: 'Ollama',
          models: ['llama2', 'codellama', 'mistral'],
          capabilities: ['chat', 'reasoning', 'code', 'local'],
          status: 'healthy',
          latency: 50,
          cost_per_token: 0,
          reliability: 0.99,
          bandwidth: 200,
        },
        {
          id: 'groq',
          name: 'Groq',
          models: ['mixtral-8x7b', 'llama2-70b'],
          capabilities: ['chat', 'reasoning', 'code', 'fast'],
          status: 'healthy',
          latency: 80,
          cost_per_token: 0.00001,
          reliability: 0.96,
          bandwidth: 700,
        },
        {
          id: 'together',
          name: 'Together AI',
          models: ['llama-2-70b', 'codellama-34b'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 100,
          cost_per_token: 0.000008,
          reliability: 0.94,
          bandwidth: 500,
        },
        {
          id: 'replicate',
          name: 'Replicate',
          models: ['llama-2-70b', 'codellama-34b'],
          capabilities: ['chat', 'reasoning', 'code'],
          status: 'healthy',
          latency: 150,
          cost_per_token: 0.000012,
          reliability: 0.93,
          bandwidth: 400,
        },
        {
          id: 'huggingface',
          name: 'Hugging Face',
          models: ['microsoft/DialoGPT-medium'],
          capabilities: ['chat', 'reasoning'],
          status: 'healthy',
          latency: 200,
          cost_per_token: 0.000006,
          reliability: 0.92,
          bandwidth: 300,
        },
        {
          id: 'cohere',
          name: 'Cohere',
          models: ['command', 'base'],
          capabilities: ['chat', 'reasoning', 'embedding'],
          status: 'healthy',
          latency: 160,
          cost_per_token: 0.000009,
          reliability: 0.95,
          bandwidth: 600,
        },
        {
          id: 'ai21',
          name: 'AI21 Labs',
          models: ['j2-ultra', 'j2-mid'],
          capabilities: ['chat', 'reasoning'],
          status: 'healthy',
          latency: 140,
          cost_per_token: 0.000011,
          reliability: 0.94,
          bandwidth: 450,
        },
      ],
    });
  }),

  http.post('http://127.0.0.1:8000/v1/routing/route', async ({ request }) => {
    const body = (await request.json()) as { task: string; requirements?: string[] };
    const { task, requirements = [] } = body;

    // Simple routing logic based on task type
    let selectedProvider = 'aliyun'; // default

    if (task.toLowerCase().includes('code') || task.toLowerCase().includes('programming')) {
      selectedProvider = 'deepseek'; // Best for coding tasks
    } else if (task.toLowerCase().includes('vision') || task.toLowerCase().includes('image')) {
      selectedProvider = 'google'; // Best for vision tasks
    } else if (task.toLowerCase().includes('fast') || task.toLowerCase().includes('quick')) {
      selectedProvider = 'groq'; // Fastest provider
    } else if (task.toLowerCase().includes('local') || task.toLowerCase().includes('offline')) {
      selectedProvider = 'ollama'; // Local provider
    } else if (requirements.includes('low-cost')) {
      selectedProvider = 'deepseek'; // Lowest cost
    } else if (requirements.includes('high-reliability')) {
      selectedProvider = 'azure_openai'; // Most reliable
    }

    return HttpResponse.json({
      provider: selectedProvider,
      model: 'gpt-4', // Default model, would be selected based on provider
      reasoning: `Selected ${selectedProvider} for task: ${task}`,
      score: 0.95,
      alternatives: ['anthropic', 'google', 'deepseek'],
    });
  }),

  // Get models for a specific provider
  http.get('http://127.0.0.1:8000/v1/routing/providers/:provider', ({ params }) => {
    const { provider } = params as { provider: string };

    // Mock models based on provider
    const providerModels: Record<string, string[]> = {
      openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
      anthropic: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
      google: ['gemini-pro', 'gemini-pro-vision'],
      deepseek: ['deepseek-chat', 'deepseek-coder'],
      aliyun: ['qwen-plus', 'qwen-turbo', 'qwen2.5-coder-32b-instruct'],
      azure_openai: ['gpt-4o', 'gpt-4.1', 'gpt-4-turbo'],
      ollama_gcp: ['qwen2.5:3b', 'gemma:2b', 'mistral:7b'],
      llamacpp_gcp: ['phi-3-mini-4k-instruct-q4', 'mistral-7b-instruct-v0.2-q4_k_m'],
      ollama: ['llama2', 'codellama', 'mistral'],
      groq: ['mixtral-8x7b', 'llama2-70b'],
      together: ['llama-2-70b', 'codellama-34b'],
      replicate: ['llama-2-70b', 'codellama-34b'],
      huggingface: ['microsoft/DialoGPT-medium'],
      cohere: ['command', 'base'],
      ai21: ['j2-ultra', 'j2-mid'],
    };

    return HttpResponse.json(providerModels[provider] || []);
  }),

  // Search endpoints
  http.post('http://127.0.0.1:8000/v1/search/query', () => {
    return HttpResponse.json({
      results: [
        {
          id: 'doc1',
          document: 'This is a sample document about AI and machine learning.',
          metadata: { source: 'test' },
          score: 0.95,
        },
      ],
      total_results: 1,
    });
  }),

  http.get('http://127.0.0.1:8000/v1/search/collections', () => {
    return HttpResponse.json({
      collections: ['default', 'documents', 'code'],
    });
  }),

  // Execute endpoints
  http.post('http://127.0.0.1:8000/v1/execute', () => {
    return HttpResponse.json({
      taskId: 'task-123',
      status: 'queued',
    });
  }),

  http.get('http://127.0.0.1:8000/v1/execute/:taskId', () => {
    return HttpResponse.json({
      taskId: 'task-123',
      status: 'completed',
      result: 'Task completed successfully',
      cost: 0.001,
      provider: 'openai',
      model: 'gpt-4',
    });
  }),

  http.get('http://127.0.0.1:8000/v1/execute/:taskId/stream', () => {
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(
            'data: {"chunk": "Starting task...", "token_count": 10, "cost_delta": 0.0005}\n\n'
          );
          controller.enqueue(
            'data: {"chunk": "Processing...", "token_count": 20, "cost_delta": 0.0005}\n\n'
          );
          controller.enqueue(
            'data: {"chunk": "Task completed", "token_count": 5, "cost_delta": 0.0001}\n\n'
          );
          controller.close();
        },
      }),
      {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
      }
    );
  }),

  // API Keys endpoints
  http.post('http://127.0.0.1:8000/v1/api-keys/:provider', () => {
    return HttpResponse.json({ message: 'API key stored successfully' });
  }),

  http.get('http://127.0.0.1:8000/v1/api-keys/:provider', () => {
    return HttpResponse.json({ key: 'sk-...' });
  }),

  http.delete('http://127.0.0.1:8000/v1/api-keys/:provider', () => {
    return HttpResponse.json({ message: 'API key deleted successfully' });
  }),

  // Parse endpoints
  http.post('http://127.0.0.1:8000/v1/parse', () => {
    return HttpResponse.json({
      steps: [
        {
          id: 'step1',
          goblin: 'docs-writer',
          task: 'document this code',
        },
        {
          id: 'step2',
          goblin: 'code-writer',
          task: 'write a unit test',
        },
      ],
      total_batches: 1,
    });
  }),

  http.post('http://127.0.0.1:8000/v1/parse/orchestration', () => {
    return HttpResponse.json({
      steps: [
        {
          id: 'step1',
          goblin: 'docs-writer',
          task: 'document this code',
          dependencies: [],
        },
        {
          id: 'step2',
          goblin: 'code-writer',
          task: 'write a unit test',
          dependencies: ['step1'],
        },
      ],
      estimated_duration: 60,
      complexity: 'medium',
    });
  }),

  // Raptor endpoints
  http.post('http://127.0.0.1:8000/v1/raptor/start', () => {
    return HttpResponse.json({ running: true });
  }),

  http.post('http://127.0.0.1:8000/v1/raptor/stop', () => {
    return HttpResponse.json({ running: false });
  }),

  http.get('http://127.0.0.1:8000/v1/raptor/status', () => {
    return HttpResponse.json({
      running: true,
      config_file: 'config/raptor.ini',
    });
  }),

  http.post('http://127.0.0.1:8000/v1/raptor/logs', () => {
    return HttpResponse.json({
      log_tail: '[INFO] Raptor monitoring started\n[INFO] System check passed',
    });
  }),

  http.get('http://127.0.0.1:8000/v1/raptor/demo/:value', ({ params }) => {
    const { value } = params as { value: string };
    return HttpResponse.json({
      result: `Demo executed with value: ${decodeURIComponent(value)}`,
    });
  }),

  // Health endpoint
  http.get('http://127.0.0.1:8000/v1/health', () => {
    return HttpResponse.json({ status: 'healthy' });
  }),

  http.get('http://127.0.0.1:8000/health', () => {
    return HttpResponse.json({ status: 'healthy' });
  }),
];
