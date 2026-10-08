import { _ as _async_to_generator } from "@swc/helpers/_/_async_to_generator";
export function load(value) {
    return _async_to_generator(function*() {
        return yield Promise.resolve(value);
    })();
}
