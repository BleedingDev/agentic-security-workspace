"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = load;
var tslib_1 = require("tslib");
function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) {
    try {
        var info = gen[key](arg);
        var value = info.value;
    }
    catch (error) {
        reject(error);
        return;
    }
    if (info.done)
        resolve(value);
    else
        Promise.resolve(value).then(_next, _throw);
}
function _async_to_generator(fn) {
    return function () {
        var self = this, args = arguments;
        return new Promise(function (resolve, reject) {
            var gen = fn.apply(self, args);
            function _next(value) {
                asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value);
            }
            function _throw(err) {
                asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err);
            }
            _next(undefined);
        });
    };
}
function load(value) {
    return _async_to_generator(function () {
        return tslib_1.__generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, Promise.resolve(value)];
                case 1: return [2 /*return*/, _a.sent()];
            }
        });
    })();
}
