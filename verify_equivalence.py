class MockTensor:
    def __init__(self, value):
        self.value = value
    def __add__(self, other):
        return MockTensor(self.value + (other.value if isinstance(other, MockTensor) else other))
    def __sub__(self, other):
        return MockTensor(self.value - (other.value if isinstance(other, MockTensor) else other))
    def __mul__(self, other):
        return MockTensor(self.value * (other.value if isinstance(other, MockTensor) else other))
    def __rmul__(self, other):
        return self.__mul__(other)
    def __repr__(self):
        return f"MockTensor({self.value})"

def manual_lerp(start, end, weight):
    return start + (end - start) * weight

def lerp_logic(start, end, weight):
    # torch.lerp(start, end, weight) is logically start + weight * (end - start)
    return start + weight * (end - start)

def verify():
    start = MockTensor(10.0)
    end = MockTensor(20.0)
    weight = 0.3

    res_manual = manual_lerp(start, end, weight)
    res_lerp = lerp_logic(start, end, weight)

    print(f"Manual: {res_manual}")
    print(f"Lerp:   {res_lerp}")

    assert res_manual.value == res_lerp.value
    print("Verification successful!")

if __name__ == "__main__":
    verify()
