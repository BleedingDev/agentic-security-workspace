const path = require('path');
module.exports = {
  entry: './src/trailing-iife/index.cjs',
  output: {
    path: path.resolve(__dirname, 'dist/wp5-trailing-iife-min'),
    filename: 'bundle.js',
  },
  mode: 'production',
  devtool: false,
  target: 'web',
};
