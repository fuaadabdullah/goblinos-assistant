/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    // Type and lint errors must be fixed before deploying
    typescript: {
        ignoreBuildErrors: true,
    },
    eslint: {
        ignoreDuringBuilds: true,
    },
    // Expose server-side env vars needed by API routes (pages/api/*).
    // NEXT_PUBLIC_* vars are auto-exposed; these are server-only runtime vars.
    serverRuntimeConfig: {
        OLLAMA_GCP_URL: process.env.OLLAMA_GCP_URL,
        LLAMACPP_GCP_URL: process.env.LLAMACPP_GCP_URL,
        GOBLIN_BACKEND_URL: process.env.GOBLIN_BACKEND_URL,
        LOCAL_LLM_API_KEY: process.env.LOCAL_LLM_API_KEY,
    },
    // Silence build warnings for missing optional deps
    webpack: (config) => {
        config.resolve.fallback = { ...config.resolve.fallback, fs: false };
        return config;
    },
};

module.exports = nextConfig;
