namespace RL.Algorithm.Extension;

public static partial class Enumerable
{
    public static IEnumerable<(T1 First, T2 Second, T3 Third, T4 Four, T5 Five)> Zip<T1, T2, T3, T4, T5>(this IEnumerable<T1> first, IEnumerable<T2> second, IEnumerable<T3> third, IEnumerable<T4> four, IEnumerable<T5> five)
    {
        ArgumentNullException.ThrowIfNull(first, nameof(first));
        ArgumentNullException.ThrowIfNull(second, nameof(second));
        ArgumentNullException.ThrowIfNull(third, nameof(third));
        ArgumentNullException.ThrowIfNull(four, nameof(four));
        ArgumentNullException.ThrowIfNull(five, nameof(five));

        return ZipIterator(first, second, third, four, five);
    }

    private static IEnumerable<(T1 First, T2 Second, T3 Third, T4 Four, T5 Five)> ZipIterator<T1, T2, T3, T4, T5>(IEnumerable<T1> first, IEnumerable<T2> second, IEnumerable<T3> third, IEnumerable<T4> four, IEnumerable<T5> five)
    {
        using IEnumerator<T1> e1 = first.GetEnumerator();
        using IEnumerator<T2> e2 = second.GetEnumerator();
        using IEnumerator<T3> e3 = third.GetEnumerator();
        using IEnumerator<T4> e4 = four.GetEnumerator();
        using IEnumerator<T5> e5 = five.GetEnumerator();
        while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext() && e4.MoveNext() && e5.MoveNext())
        {
            yield return (e1.Current, e2.Current, e3.Current, e4.Current, e5.Current);
        }
    }
}
