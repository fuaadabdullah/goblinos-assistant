/**
 * Centralized environment configuration
 *
 * Benefits:
 * - Type safety for env vars
 * - Validation at startup
 * - Single source of truth
 * - Easier testing
 */

import { devLog } from '../utils/dev-log';

interface EnvConfig {
  // API Configuration
  apiBaseUrl: string;
  backendUrl: string;
  fastApiUrl: string;

  // Feature Flags
  enableDebug: boolean;
  mockApi: boolean;
  features: {
    ragEnabled: boolean;
    multiProvider: boolean;
    passkeyAuth: boolean;
    googleAuth: boolean;
    orchestration: boolean;
    sandbox: boolean;
    search: boolean;
    admin: boolean;
    analytics: boolean;
    debugMode: boolean;
  };

  // Turnstile
  turnstile: {
    chat: string;
    login: string;
    search: string;
  };

  // Monitoring
  sentryDsn: string;
  posthogApiKey: string;
  posthogHost: string;

  // Build Info
  mode: 'development' | 'production' | 'test';
  isDevelopment: boolean;
  isProduction: boolean;
}

function getOptionalEnv(key: string, defaultValue: string = ''): string {
  return process.env[key] || defaultValue;
}

function validateEnvConfig(config: EnvConfig): void {
  const errors: string[] = [];

  // Validate URLs
  try {
    new URL(config.apiBaseUrl);
  } catch {
    errors.push(`Invalid NEXT_PUBLIC_API_BASE_URL: ${config.apiBaseUrl}`);
  }

  // Warn about production mode with debug enabled
  if (config.isProduction && config.enableDebug) {
    console.warn('⚠️  Debug mode enabled in production!');
  }

  // Check Turnstile keys format
  Object.entries(config.turnstile).forEach(([key, value]) => {
    if (value && !value.startsWith('0x')) {
      errors.push(`Invalid Turnstile key format for ${key}`);
    }
  });

  if (errors.length > 0) {
    console.error('❌ Environment configuration errors:\n', errors.join('\n'));
    throw new Error('Invalid environment configuration');
  }
}

// Export typed configuration
export const env: EnvConfig = {
  apiBaseUrl: getOptionalEnv('NEXT_PUBLIC_API_BASE_URL', 'https://goblin-backend.onrender.com'),
  backendUrl: getOptionalEnv('NEXT_PUBLIC_BACKEND_URL', 'https://goblin-backend.onrender.com'),
  fastApiUrl: getOptionalEnv('NEXT_PUBLIC_FASTAPI_URL', 'https://goblin-backend.onrender.com'),

  enableDebug: getOptionalEnv('NEXT_PUBLIC_ENABLE_DEBUG') === 'true',
  mockApi: getOptionalEnv('NEXT_PUBLIC_MOCK_API') === 'true',

  features: {
    ragEnabled: getOptionalEnv('NEXT_PUBLIC_FEATURE_RAG_ENABLED') === 'true',
    multiProvider: getOptionalEnv('NEXT_PUBLIC_FEATURE_MULTI_PROVIDER') === 'true',
    passkeyAuth: getOptionalEnv('NEXT_PUBLIC_FEATURE_PASSKEY_AUTH') === 'true',
    googleAuth: getOptionalEnv('NEXT_PUBLIC_FEATURE_GOOGLE_AUTH') === 'true',
    orchestration: getOptionalEnv('NEXT_PUBLIC_FEATURE_ORCHESTRATION') === 'true',
    sandbox: getOptionalEnv('NEXT_PUBLIC_FEATURE_SANDBOX') === 'true',
    search: getOptionalEnv('NEXT_PUBLIC_FEATURE_SEARCH', 'true') === 'true',
    admin: getOptionalEnv('NEXT_PUBLIC_FEATURE_ADMIN', 'false') === 'true',
    analytics: getOptionalEnv('NEXT_PUBLIC_ENABLE_ANALYTICS') === 'true',
    debugMode: getOptionalEnv('NEXT_PUBLIC_DEBUG_MODE') === 'true',
  },

  turnstile: {
    chat: getOptionalEnv('NEXT_PUBLIC_TURNSTILE_SITE_KEY_CHAT'),
    login: getOptionalEnv('NEXT_PUBLIC_TURNSTILE_SITE_KEY_LOGIN'),
    search: getOptionalEnv('NEXT_PUBLIC_TURNSTILE_SITE_KEY_SEARCH'),
  },

  sentryDsn: getOptionalEnv('NEXT_PUBLIC_SENTRY_DSN'),
  posthogApiKey: getOptionalEnv('NEXT_PUBLIC_POSTHOG_API_KEY'),
  posthogHost: getOptionalEnv('NEXT_PUBLIC_POSTHOG_HOST'),

  mode: (process.env.NODE_ENV as EnvConfig['mode']) || 'development',
  isDevelopment: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
};

// Validate on import
validateEnvConfig(env);

// Log configuration in development
if (env.isDevelopment) {
  devLog('📝 Environment Configuration:', {
    ...env,
    apiBaseUrl: env.apiBaseUrl,
    mode: env.mode,
  });
}
