def partition(S: set, n=2):
    """Return all ways of partitioning a set into n sets."""
    S = S.copy()
    if len(S) == 1:
        return [[S] + [set()] * (n - 1)]
    partitions = []
    item = S.pop()
    lower_partitions = partition(S, n)
    for p in lower_partitions:
        for subset in p:
            a = p.copy()
            a.remove(subset)
            s = subset.copy()
            s.add(item)
            a.append(s)
            partitions.append(a)

    return partitions
