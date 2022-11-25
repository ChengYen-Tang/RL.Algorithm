namespace RL.Algorithm.Common;

public abstract class Union<T1, T2>
{
    public abstract void MatchAction(Action<T1> first, Action<T2> second);
    public abstract T MatchFunc<T>(Func<T1, T> first, Func<T2, T> second);

    public static explicit operator Union<T1, T2>(T1 value)
        => new Case1(value);
    public static explicit operator Union<T1, T2>(T2 value)
        => new Case2(value);

    internal sealed class Case1 : Union<T1, T2>
    {
        public readonly T1 item;
        public Case1(T1 item) : base() { this.item = item; }
        public override T MatchFunc<T>(Func<T1, T> first, Func<T2, T> second)
            => first(item);
        public override void MatchAction(Action<T1> first, Action<T2> second)
            => first?.Invoke(item);
    }

    internal sealed class Case2 : Union<T1, T2>
    {
        public readonly T2 item;
        public Case2(T2 item) : base() { this.item = item; }
        public override T MatchFunc<T>(Func<T1, T> first, Func<T2, T> second)
            => second(item);
        public override void MatchAction(Action<T1> first, Action<T2> second)
            => second?.Invoke(item);
    }
}
