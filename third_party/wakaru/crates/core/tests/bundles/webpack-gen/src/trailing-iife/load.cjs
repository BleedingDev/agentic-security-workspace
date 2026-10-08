module.exports = async function load(value) {
  if (globalThis.failLoad) throw new Error('load failed');
  return value + 1;
};
