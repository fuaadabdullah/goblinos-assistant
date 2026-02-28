const { FlatCompat } = require("@eslint/eslintrc");
const compat = new FlatCompat({ baseDirectory: __dirname });

module.exports = [
    {
        ignores: [
            ".next/**",
            "node_modules/**",
            "dist/**",
            "coverage/**",
            "public/**",
            ".git/**",
            ".vercel/**",
            "**/._*",
            "**/*.stories.*",
        ],
    },
    ...compat.extends("next/core-web-vitals", "next/typescript"),
    {
        rules: {
            "no-console": ["warn", { allow: ["warn", "error", "debug"] }],
            "react/no-unescaped-entities": "off",
            "max-lines-per-function": [
                "warn",
                { max: 140, skipComments: true, skipBlankLines: true },
            ],
            complexity: ["warn", 12],
            "max-params": ["warn", 5],
            "max-statements": ["warn", 50],
        },
    },
];
