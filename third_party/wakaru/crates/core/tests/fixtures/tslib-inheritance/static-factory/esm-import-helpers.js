import { __extends } from "tslib";
var Child = /** @class */ (function (_super) {
    __extends(Child, _super);
    function Child() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Child.make = function (value) { return new Child(value); };
    return Child;
}(Parent));
export { Child };
