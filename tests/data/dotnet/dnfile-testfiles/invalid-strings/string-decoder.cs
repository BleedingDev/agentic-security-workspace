using System;

public class HelloWorld
{
    static char KEY = (char) 0xFF;

    static private string decode(string ct)
    {
        char[] chars = ct.ToCharArray();

        for (int i = 0; i < chars.Length; i++) {
            chars[i] = (char) (((chars[i] & 0xFF) ^ KEY) | ((chars[i] & 0xFF00) ^ (KEY << 8)));
        }

        return new string(chars);
    }

    public static void Main(string[] args)
    {
        Console.WriteLine(decode("\uFFB7\uFF9A\uFF93\uFF93\uFF90\uFFDF\uFFA8\uFF90\uFF8D\uFF93\uFF9B"));
    }
}
