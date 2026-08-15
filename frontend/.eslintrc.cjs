/**
 * ESLint 8 config (.eslintrc, not flat) to match the installed major version.
 *
 * Deliberately narrow: type-aware linting would need a second tsconfig pass on
 * every run, and `tsc --noEmit` already covers types in both `npm run build` and
 * `npm run typecheck`. What is left for ESLint is the class of mistake the type
 * checker cannot see - stale hook dependencies above all.
 */
module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parser: '@typescript-eslint/parser',
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ecmaFeatures: { jsx: true } },
  plugins: ['@typescript-eslint', 'react-hooks'],
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended'],
  rules: {
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
    // The API returns JSON typed as `unknown`/`Record<string, unknown>` in places
    // and the narrowing is done with String()/Number() at the point of use.
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
    ],
    // `no-undef` duplicates the type checker and misfires on TS globals.
    'no-undef': 'off',
  },
  ignorePatterns: ['dist', 'node_modules', '*.config.ts', '*.cjs'],
};
