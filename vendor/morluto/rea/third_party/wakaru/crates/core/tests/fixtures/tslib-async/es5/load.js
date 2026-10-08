"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = load;
var tslib_1 = require("tslib");
function load(value) {
    return tslib_1.__awaiter(this, void 0, void 0, function () {
        var result;
        return tslib_1.__generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, value];
                case 1:
                    result = _a.sent();
                    return [2 /*return*/, result + 1];
            }
        });
    });
}
