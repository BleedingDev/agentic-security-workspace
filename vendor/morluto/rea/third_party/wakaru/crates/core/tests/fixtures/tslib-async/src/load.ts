export async function load(value: Promise<number>): Promise<number> {
  const result = await value;
  return result + 1;
}
