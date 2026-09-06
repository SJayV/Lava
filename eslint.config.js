export default [
  {
    files: ['**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
    },
    rules: {
      'no-unused-vars': ['error', { args: 'none' }],
      eqeqeq: 'error',
      'no-var': 'error',
    },
  },
];
