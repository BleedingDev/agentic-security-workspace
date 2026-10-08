import { __extends } from "tslib";
var Child = /** @class */ (function (_super) {
    __extends(Child, _super);
    function Child() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Child.prototype.value = function () { return _super.prototype.value.call(this) + 1; };
    return Child;
}(Parent));
export { Child };
