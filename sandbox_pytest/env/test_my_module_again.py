from my_module import square

import pytest

@pytest.mark.parametrize(
        'inputs', [2, 3, 4]
)

def test_square_return_value_type_is_int(inputs):
    # When
    subject = square(inputs)

    # Then
    assert isinstance(subject, int)