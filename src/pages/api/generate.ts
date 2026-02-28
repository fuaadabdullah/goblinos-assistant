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

interface ForwardResponse {
  status: number;
  body: unknown;
  correlationId?: string;
  requestId?: string;
  licenseTier?: string;
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

function getRequestHeader(req: NextApiRequest, name: string): string | undefined {
  const value = req.headers[name.toLowerCase()];
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'string' && value.trim()) return value.trim();
  return undefined;
}

function logProxyEvent(payload: {
  proxy_mode: 'thin';
  backend_status: number | null;
  legacy_fallback_used: boolean;
  correlation_id?: string;
  latency_ms: number;
}) {
  console.info('[api/generate] proxy_event', payload);
}

function writeResponse(
  res: NextApiResponse,
  result: {
    status: number;
    body: unknown;
    correlationId?: string;
    requestId?: string;
    licenseTier?: string;
  },
) {
  if (result.correlationId) res.setHeader('X-Correlation-ID', result.correlationId);
  if (result.requestId) res.setHeader('X-Request-ID', result.requestId);
  if (result.licenseTier) res.setHeader('X-License-Tier', result.licenseTier);
  return res.status(result.status).json(result.body ?? {});
}

async function forwardToBackendGenerate(
  req: NextApiRequest,
): Promise<ForwardResponse> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (INTERNAL_PROXY_API_KEY) {
      headers['X-Internal-API-Key'] = INTERNAL_PROXY_API_KEY;
    }

    const forwardedFor = getRequestHeader(req, 'x-forwarded-for');
    if (forwardedFor) {
      headers['X-Forwarded-For'] = forwardedFor;
    }

    const response = await fetchWithTimeout(
      `${BACKEND_URL}/v1/api/generate`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify(req.body ?? {}),
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
      requestId: response.headers.get('x-request-id') || undefined,
      licenseTier: response.headers.get('x-license-tier') || undefined,
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
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  const startedAt = Date.now();
  const result = await forwardToBackendGenerate(req);

  writeResponse(res, {
    status: result.status,
    body: result.body,
    correlationId: result.correlationId,
    requestId: result.requestId,
    licenseTier: result.licenseTier,
  });

  logProxyEvent({
    proxy_mode: 'thin',
    backend_status: result.transportError ? null : result.status,
    legacy_fallback_used: false,
    correlation_id: result.correlationId,
    latency_ms: Date.now() - startedAt,
  });
}
