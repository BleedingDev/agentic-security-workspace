"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Child = void 0;
var tslib_1 = require("tslib");
var Child = /** @class */ (function (_super) {
    tslib_1.__extends(Child, _super);
    function Child() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Child.prototype.value = function () { return _super.prototype.value.call(this) + 1; };
    return Child;
}(Parent));
exports.Child = Child;
