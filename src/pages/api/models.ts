import type { NextApiRequest, NextApiResponse } from 'next';

const cleanUrl = (value?: string) => (value || '').trim().replace(/\/$/, '');

const BACKEND_URL =
  cleanUrl(
      process.env.GOBLIN_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      'https://goblin-backend.onrender.com',
  ) || 'https://goblin-backend.onrender.com';

const INTERNAL_PROXY_API_KEY =
  (process.env.INTERNAL_PROXY_API_KEY ||
    process.env.BACKEND_API_KEY ||
    process.env.INTERNAL_API_SECRET ||
    '').trim();

interface ForwardModelsResult {
  status: number;
  body: unknown;
  correlationId?: string;
  transportError?: boolean;
}

async function safeJson<T = unknown>(res: Response): Promise<T | null> {
  try {
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}

function logProxyEvent(payload: {
  proxy_mode: 'thin';
  backend_status: number | null;
  legacy_fallback_used: boolean;
  correlation_id?: string;
  latency_ms: number;
}) {
  console.info('[api/models] proxy_event', payload);
}

async function forwardToBackendModels(): Promise<ForwardModelsResult> {
  try {
    const headers: Record<string, string> = {};
    if (INTERNAL_PROXY_API_KEY) {
      headers['X-Internal-API-Key'] = INTERNAL_PROXY_API_KEY;
    }

    const response = await fetchWithTimeout(
      `${BACKEND_URL}/v1/providers/models`,
      {
        method: 'GET',
        headers,
      },
      10000,
    );

    const body = (await safeJson(response)) ?? {
      detail: 'Backend returned a non-JSON response',
    };

    return {
      status: response.status,
      body,
      correlationId: response.headers.get('x-correlation-id') || undefined,
    };
  } catch {
    return {
      status: 502,
      body: { error: 'Backend unreachable' },
      transportError: true,
    };
  }
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const startedAt = Date.now();
  const result = await forwardToBackendModels();

  if (result.correlationId) {
    res.setHeader('X-Correlation-ID', result.correlationId);
  }

  res.status(result.status).json(result.body ?? {});

  logProxyEvent({
    proxy_mode: 'thin',
    backend_status: result.transportError ? null : result.status,
    legacy_fallback_used: false,
    correlation_id: result.correlationId,
    latency_ms: Date.now() - startedAt,
  });
}
