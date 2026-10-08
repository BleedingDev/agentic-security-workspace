declare const Parent: any;
export class Child extends Parent {
    value() { return super.value() + 1; }
}
