import numpy as np

class SumTree:
    '''
    Implementation consulted with https://github.com/Howuhh/prioritized_experience_replay
    '''
    def __init__(self, size: int) -> None:
        self.tree = np.full(fill_value=1e-5, shape=(size * 2 - 1,), dtype=np.float32)
        self.data = np.zeros(shape=(size), dtype=np.float32)
        self.it = 0
        self.capacity = size
        self.size = 0

    def total_sum(self):
        return self.tree[0]

    def add(self, elem, priority):
        # first add the data
        self.data[self.it] = elem
        self.update_tree(self.it, priority)
        self.it = (self.it + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_parent_idx(self, idx):
        return (idx  - 1) // 2

    def update_tree(self, idx, value):
        tree_idx = idx + self.capacity - 1
        change = value - self.tree[tree_idx]
        self.tree[tree_idx] = value # update the value of the leaf
        parent_idx = self.get_parent_idx(tree_idx)
        while parent_idx >= 0:
            self.tree[parent_idx] += change
            parent_idx = self.get_parent_idx(parent_idx)

    def get(self, p):
        idx = 0
        while p > 0 and (2 * idx + 1) < self.tree.shape[0]:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if p <= self.tree[left]:
                idx = left
            else:
                idx = right
                p -= self.tree[left]

        data_idx = idx - self.capacity + 1
        return self.tree[idx], self.data[data_idx], data_idx


def main():
    t = SumTree(size=4)
    t.add(5, 5)
    t.add(4, 1)
    t.add(3, 1)
    t.add(2, 1)
    print(t.total_sum())
    for i in range(100):
        p = np.random.uniform(low=0, high=t.total_sum())
        print(f"{t.get(p)}")

if __name__ == '__main__':
    main()