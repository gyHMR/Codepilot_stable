def clamp(value: int, minimum: int, maximum: int) -> int:
    if value < minimum:
        return maximum
    if value > maximum:
        return minimum
    return value

