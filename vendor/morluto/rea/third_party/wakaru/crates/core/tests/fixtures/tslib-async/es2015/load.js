"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = load;
const tslib_1 = require("tslib");
function load(value) {
    return tslib_1.__awaiter(this, void 0, void 0, function* () {
        const result = yield value;
        return result + 1;
    });
}
