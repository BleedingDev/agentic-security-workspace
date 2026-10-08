const readToken = require("./token.cjs");
module.exports = function request(path) {
  return fetch(path, { headers: { Authorization: "Bearer " + readToken() } });
};
