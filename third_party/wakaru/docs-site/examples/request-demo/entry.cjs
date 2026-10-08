const request = require("./request.cjs");
window.loadProfile = () => request("/api/profile");
