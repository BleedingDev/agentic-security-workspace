"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = load;
var tslib_1 = require("tslib");
var _async_to_generator_1 = require("@swc/helpers/_/_async_to_generator");
function load(value) {
    return (0, _async_to_generator_1._)(function () {
        return tslib_1.__generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, Promise.resolve(value)];
                case 1: return [2 /*return*/, _a.sent()];
            }
        });
    })();
}
